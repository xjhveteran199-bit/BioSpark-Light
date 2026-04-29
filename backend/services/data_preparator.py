"""
Data preparation: turn raw uploads into the training-ready CSV format.

The trainer expects a labeled CSV where each row is one already-segmented
sample (signal columns s1..sN + a label column). End users typically have
*long* recordings instead — this module windows them into rows.

Three input modes:
  - Mode A: ZIP with folder-per-class, each CSV is one long recording
  - Mode B: single long CSV + a list of label time intervals
  - Mode C: generic ZIP, user provides a {filename: label} mapping

All three return a pandas DataFrame in the trainer's expected shape so the
caller can hand it straight to dataset_loader.load_labeled_dataframe().
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from backend.services.preprocess import _segment

MAX_OUTPUT_ROWS = 50_000  # hard cap on generated samples to protect memory

_TIME_COL_NAMES = {"time", "t", "timestamp", "seconds", "sec", "ms", "index"}

# OpenBCI GUI exports `.txt` files that are comma-separated but prefixed
# by 4–6 lines starting with `%` (free-form metadata). We strip those when
# reading. Pattern below also tolerates leading whitespace some boards add.
_OPENBCI_COMMENT_RE = re.compile(rb"^\s*%")
# Look for `%Sample Rate = 250 Hz` or `%Sample Rate = 250.0 Hz` in the header.
_OPENBCI_SR_RE = re.compile(
    rb"%\s*Sample\s*Rate\s*[:=]\s*(\d+(?:\.\d+)?)", re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Inspection — used to pre-fill the form before segmentation.
# ---------------------------------------------------------------------------

def inspect_upload(filename: str, file_bytes: bytes) -> dict:
    """Peek at the upload and return metadata for the prep form prefill.

    Returns:
        {
          kind: 'csv' | 'zip',
          files: [{path, columns, rows, is_csv}],   # for zips: all CSV entries
          columns: [...],                            # first CSV's columns
          row_count: int,                            # first CSV's row count
          suggested_signal_col: int,                 # heuristic: first numeric non-time column
          suggested_sampling_rate: float | None,     # if a time column exists, infer 1/dt
          class_folders: [str] | None,               # if zip with folder-per-class
        }
    """
    ext = Path(filename).suffix.lower()
    # Reject other archive formats with a clear message — only .zip is
    # parseable by Python's stdlib zipfile.
    if ext in (".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".xz"):
        raise ValueError(
            f"Archive format '{ext}' is not supported. Only .zip is supported — "
            f"please re-package the folder as a .zip (e.g. with WinRAR / 7-Zip) and try again."
        )
    if ext == ".zip":
        # Validate the actual bytes to catch mis-named files (e.g. a .rar renamed to .zip).
        if not zipfile.is_zipfile(io.BytesIO(file_bytes)):
            raise ValueError(
                "The uploaded file does not appear to be a valid .zip archive. "
                "If it is actually .rar / .7z / .tar.gz, re-package it as .zip first."
            )
        return _inspect_zip(file_bytes)
    if ext in (".csv", ".txt"):
        return _inspect_csv(file_bytes, filename)
    raise ValueError(f"Unsupported file type '{ext}'. Upload .csv or .zip.")


def _inspect_csv(file_bytes: bytes, filename: str) -> dict:
    df = _read_signal_bytes(file_bytes)
    obci = _parse_openbci_header(file_bytes)
    sr = _guess_sampling_rate(df)
    if sr is None and "sample_rate" in obci:
        sr = obci["sample_rate"]
    return {
        "kind": "csv",
        "files": [{
            "path": filename,
            "columns": list(df.columns),
            "rows": len(df),
            "is_csv": True,
        }],
        "columns": list(df.columns),
        "row_count": len(df),
        "suggested_signal_col": _guess_signal_column(df),
        "suggested_signal_col_indices": _guess_signal_columns(df),
        "suggested_sampling_rate": sr,
        "openbci_detected": bool(obci),
        "class_folders": None,
    }


def _inspect_zip(file_bytes: bytes) -> dict:
    files: list[dict] = []
    class_folders: set[str] = set()
    first_columns: list[str] = []
    first_rows = 0
    suggested_sig = 0
    suggested_sig_idxs: list[int] = []
    suggested_sr: float | None = None
    openbci_detected = False

    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if any(p.startswith("__") or p.startswith(".") for p in parts):
                continue
            if not name.lower().endswith((".csv", ".txt")):
                continue
            if len(parts) >= 2:
                class_folders.add(parts[-2])
            try:
                with zf.open(name) as fp:
                    raw = fp.read()
                df = _read_csv_or_txt(name, raw)
            except Exception as exc:  # noqa: BLE001
                files.append({"path": name, "columns": [], "rows": 0, "is_csv": True, "error": str(exc)})
                continue
            entry = {
                "path": name,
                "columns": list(df.columns),
                "rows": len(df),
                "is_csv": True,
            }
            files.append(entry)
            if not first_columns:
                first_columns = list(df.columns)
                first_rows = len(df)
                suggested_sig = _guess_signal_column(df)
                suggested_sig_idxs = _guess_signal_columns(df)
                suggested_sr = _guess_sampling_rate(df)
                obci = _parse_openbci_header(raw)
                if obci:
                    openbci_detected = True
                    if suggested_sr is None and "sample_rate" in obci:
                        suggested_sr = obci["sample_rate"]

    if not files:
        raise ValueError("ZIP contains no CSV/TXT files.")

    return {
        "kind": "zip",
        "files": files,
        "columns": first_columns,
        "row_count": first_rows,
        "suggested_signal_col": suggested_sig,
        "suggested_signal_col_indices": suggested_sig_idxs,
        "suggested_sampling_rate": suggested_sr,
        "openbci_detected": openbci_detected,
        "class_folders": sorted(class_folders) or None,
    }


# ---------------------------------------------------------------------------
# Segmentation — Mode A / B / C
# ---------------------------------------------------------------------------

def segment_long_recordings(
    zip_bytes: bytes,
    sampling_rate: float,
    segment_length_sec: float,
    overlap_ratio: float = 0.0,
    signal_col_index: int = 0,
    signal_col_indices: Sequence[int] | None = None,
    stride_sec: float | None = None,
    group_by: str = "recording",
) -> pd.DataFrame:
    """Mode A: ZIP with ``class_folder/recording.{csv,txt}``. One long
    recording per file. The immediate parent folder name becomes the
    label.

    Multi-channel: pass ``signal_col_indices=[2,3,4,5,6]`` to extract
    five columns as a 5-channel signal (column index is 0-based against
    the file's column order). Output columns are ``ch1_1..chC_S``.

    OpenBCI ``.txt`` files (with leading ``%``-prefixed comment lines)
    are accepted alongside ``.csv`` and read with a comment-aware parser.

    Stride: when the protocol structure is ``[valid][rest]`` per
    epoch (e.g. 1 s task + 4 s rest), set
    ``segment_length_sec=valid_length`` and
    ``stride_sec=epoch_length`` to skip the rest portion. When
    ``stride_sec`` is ``None`` we fall back to the legacy
    ``overlap_ratio`` semantics.
    """
    _validate_window(sampling_rate, segment_length_sec, overlap_ratio, stride_sec)
    indices = _resolve_signal_indices(signal_col_index, signal_col_indices)
    n_channels = len(indices)
    seg_len = int(round(sampling_rate * segment_length_sec))
    stride = int(round(sampling_rate * stride_sec)) if stride_sec else None

    rows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if any(p.startswith("__") or p.startswith(".") for p in parts):
                continue
            if not name.lower().endswith((".csv", ".txt")):
                continue
            if len(parts) < 2:
                continue
            label = parts[-2]
            try:
                with zf.open(name) as fp:
                    df = _read_csv_or_txt(name, fp.read())
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Failed to read '{name}': {exc}")
            sig = _extract_multi_signal(df, indices)
            trial_idx = 0
            for seg in _segment(sig, seg_len, overlap_ratio, stride=stride):
                rows.append(seg)
                labels.append(label)
                groups.append(_group_id(name, trial_idx, group_by))
                trial_idx += 1
                if len(rows) > MAX_OUTPUT_ROWS:
                    raise ValueError(
                        f"Generated more than {MAX_OUTPUT_ROWS} segments — "
                        "increase segment length or reduce overlap."
                    )

    if not rows:
        raise ValueError(
            "No segments produced. Check sampling rate, segment length, and "
            "that each class folder contains CSVs/TXTs longer than one segment."
        )

    return _rows_to_df(rows, labels, seg_len, groups, n_channels=n_channels)


def segment_with_intervals(
    csv_bytes: bytes,
    intervals: list[dict],
    sampling_rate: float,
    segment_length_sec: float,
    overlap_ratio: float = 0.0,
    signal_col_index: int = 0,
    signal_col_indices: Sequence[int] | None = None,
    stride_sec: float | None = None,
    filename: str = "upload.csv",
    group_by: str = "recording",
) -> pd.DataFrame:
    """Mode B: single long CSV/TXT + label intervals.

    intervals: [{"start_sec": float, "end_sec": float, "label": str}, ...]

    Same multi-channel + OpenBCI .txt + stride extensions as Mode A.
    """
    _validate_window(sampling_rate, segment_length_sec, overlap_ratio, stride_sec)
    if not intervals:
        raise ValueError("At least one interval is required.")
    indices = _resolve_signal_indices(signal_col_index, signal_col_indices)
    n_channels = len(indices)
    seg_len = int(round(sampling_rate * segment_length_sec))
    stride = int(round(sampling_rate * stride_sec)) if stride_sec else None

    df = _read_csv_or_txt(filename, csv_bytes)
    sig = _extract_multi_signal(df, indices)
    duration_sec = len(sig) / sampling_rate

    rows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    for i, iv in enumerate(intervals):
        try:
            start = float(iv["start_sec"])
            end = float(iv["end_sec"])
            label = str(iv["label"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Interval #{i} malformed: {exc}")
        if not label:
            raise ValueError(f"Interval #{i} has empty label.")
        if end <= start:
            raise ValueError(f"Interval #{i}: end_sec ({end}) must exceed start_sec ({start}).")
        if end > duration_sec + 1e-6:
            raise ValueError(
                f"Interval #{i}: end_sec ({end}) exceeds signal duration ({duration_sec:.2f}s)."
            )
        s_idx = int(round(start * sampling_rate))
        e_idx = int(round(end * sampling_rate))
        slice_sig = sig[s_idx:e_idx]
        trial_idx = 0
        for seg in _segment(slice_sig, seg_len, overlap_ratio, stride=stride):
            rows.append(seg)
            labels.append(label)
            groups.append(_group_id(f"interval_{i}", trial_idx, group_by))
            trial_idx += 1
            if len(rows) > MAX_OUTPUT_ROWS:
                raise ValueError(
                    f"Generated more than {MAX_OUTPUT_ROWS} segments — "
                    "increase segment length or reduce overlap."
                )

    if not rows:
        raise ValueError("No segments produced from the given intervals.")

    return _rows_to_df(rows, labels, seg_len, groups, n_channels=n_channels)


def segment_generic(
    zip_bytes: bytes,
    file_label_map: dict[str, str],
    signal_col_index: int,
    sampling_rate: float,
    segment_length_sec: float,
    overlap_ratio: float = 0.0,
    signal_col_indices: Sequence[int] | None = None,
    stride_sec: float | None = None,
    group_by: str = "recording",
) -> pd.DataFrame:
    """Mode C: generic ZIP, user provides per-file label mapping.

    Same multi-channel + OpenBCI .txt + stride extensions as Mode A.
    """
    _validate_window(sampling_rate, segment_length_sec, overlap_ratio, stride_sec)
    if not file_label_map:
        raise ValueError("file_label_map is empty — assign at least one file a label.")
    indices = _resolve_signal_indices(signal_col_index, signal_col_indices)
    n_channels = len(indices)
    seg_len = int(round(sampling_rate * segment_length_sec))
    stride = int(round(sampling_rate * stride_sec)) if stride_sec else None

    rows: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    matched_files: set[str] = set()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zip_files = {n for n in zf.namelist() if n.lower().endswith((".csv", ".txt"))}
        for fname, label in file_label_map.items():
            if not label or not str(label).strip():
                continue
            if fname not in zip_files:
                raise ValueError(f"File '{fname}' from mapping not found in ZIP.")
            with zf.open(fname) as fp:
                df = _read_csv_or_txt(fname, fp.read())
            sig = _extract_multi_signal(df, indices)
            trial_idx = 0
            for seg in _segment(sig, seg_len, overlap_ratio, stride=stride):
                rows.append(seg)
                labels.append(str(label).strip())
                groups.append(_group_id(fname, trial_idx, group_by))
                trial_idx += 1
                matched_files.add(fname)
                if len(rows) > MAX_OUTPUT_ROWS:
                    raise ValueError(
                        f"Generated more than {MAX_OUTPUT_ROWS} segments — "
                        "increase segment length or reduce overlap."
                    )

    if not rows:
        raise ValueError("No segments produced. Check labels and signal column.")

    return _rows_to_df(rows, labels, seg_len, groups, n_channels=n_channels)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _validate_window(
    sr: float,
    seg_sec: float,
    overlap: float,
    stride_sec: float | None = None,
) -> None:
    if sr <= 0:
        raise ValueError(f"sampling_rate must be > 0 (got {sr}).")
    if seg_sec <= 0:
        raise ValueError(f"segment_length_sec must be > 0 (got {seg_sec}).")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap_ratio must be in [0, 1) (got {overlap}).")
    if int(round(sr * seg_sec)) < 4:
        raise ValueError("Resulting segment length is < 4 samples — adjust SR or segment length.")
    if stride_sec is not None:
        if stride_sec <= 0:
            raise ValueError(f"stride_sec must be > 0 (got {stride_sec}).")
        if stride_sec < seg_sec:
            raise ValueError(
                f"stride_sec ({stride_sec}) must be ≥ segment_length_sec "
                f"({seg_sec}) — otherwise consecutive segments would overlap "
                "in unsigned ways. Use overlap_ratio for that case."
            )


def _looks_like_data(line: str) -> bool:
    """True if `line` looks like a row of numeric data, not a header row.

    Tolerates one trailing token that is a timestamp (``HH:MM:SS`` or
    ``HH:MM:SS.sss``) — OpenBCI's headerless export ends each row with a
    wall-clock timestamp string and we don't want to treat that lone
    non-numeric token as evidence of a header.
    """
    parts = [p.strip() for p in line.split(",")]
    if not parts or all(p == "" for p in parts):
        return False
    timestamp_re = re.compile(r"^\d{1,2}:\d{2}:\d{2}(?:\.\d+)?$")
    saw_timestamp = False
    for p in parts:
        if not p:
            continue
        # Try numeric first (covers ints, floats, signed, exponential).
        try:
            float(p)
            continue
        except ValueError:
            pass
        # Allow exactly one trailing timestamp.
        if not saw_timestamp and timestamp_re.match(p):
            saw_timestamp = True
            continue
        return False
    return True


def _read_signal_bytes(file_bytes: bytes) -> pd.DataFrame:
    """Read CSV/TXT bytes into a DataFrame, tolerating OpenBCI-style headers.

    Two OpenBCI GUI export formats are handled:
      1. Headered  : 4–6 ``%``-prefixed lines, then a comma-separated
         column-name row, then data rows (e.g. ``Sample Index, EXG
         Channel 0, ...``). We strip the ``%`` lines, pandas parses
         the rest with the column-name row as the header.
      2. Headerless: 4–6 ``%``-prefixed lines, then data rows directly
         (e.g. ``0, 0.00, 0.00, ..., 18:51:16.059``). We detect this
         by sniffing the first non-comment line — if it parses as
         all-numeric (with at most one trailing ``HH:MM:SS`` token),
         we feed it to pandas with ``header=None`` and assign synthetic
         column names ``c0..cN`` so downstream column-index lookups
         still work.

    Plain ``.csv`` files without a header row are also handled by the
    same headerless detection.
    """
    text = file_bytes.decode("utf-8", errors="replace")
    # Drop leading comment lines starting with `%`. Doing this on the
    # decoded text (not bytes) keeps line-ending handling sane on Windows.
    lines = text.splitlines()
    first_data_line = 0
    for i, line in enumerate(lines):
        if line.lstrip().startswith("%"):
            first_data_line = i + 1
            continue
        break
    cleaned = "\n".join(lines[first_data_line:])
    # Sniff the first non-comment line: if it's all numeric (with an
    # optional trailing timestamp), the file is headerless.
    probe = lines[first_data_line] if first_data_line < len(lines) else ""
    if _looks_like_data(probe):
        # Use the first row to determine column count, then assign
        # synthetic names so pd.read_csv has stable headers and the
        # caller can index by integer position as usual.
        n_cols = len([p for p in probe.split(",") if p.strip() != ""])
        names = [f"c{i}" for i in range(n_cols)]
        return pd.read_csv(io.StringIO(cleaned), header=None, names=names)
    return pd.read_csv(io.StringIO(cleaned))


def _parse_openbci_header(file_bytes: bytes) -> dict:
    """Scan the first ~4KB for OpenBCI-style metadata. Returns a dict that
    may contain ``sample_rate`` (float). Empty dict if nothing found —
    the inspect endpoint just uses this to suggest defaults."""
    head = file_bytes[:4096]
    info: dict = {}
    m = _OPENBCI_SR_RE.search(head)
    if m:
        try:
            info["sample_rate"] = float(m.group(1))
        except ValueError:
            pass
    return info


def _read_csv_or_txt(filename: str, file_bytes: bytes) -> pd.DataFrame:
    """Dispatch to the comment-aware reader for `.txt` files; pandas for `.csv`."""
    if filename.lower().endswith(".txt"):
        return _read_signal_bytes(file_bytes)
    # Plain CSV — but still tolerate `%` headers if the caller mislabels.
    text = file_bytes.decode("utf-8", errors="replace")
    if text.lstrip().startswith("%"):
        return _read_signal_bytes(file_bytes)
    return pd.read_csv(io.StringIO(text))


def _resolve_signal_indices(
    signal_col_index: int,
    signal_col_indices: Sequence[int] | None,
) -> list[int]:
    """Return the (possibly multi-channel) list of column indices to extract."""
    if signal_col_indices:
        return list(signal_col_indices)
    return [signal_col_index]


_VALID_GROUP_BY = ("recording", "trial")


def _group_id(source: str, trial_idx: int, group_by: str) -> str:
    """Build the per-segment group id used by the trainer's group-aware
    train/val/test split.

    - ``recording`` (default): one group per source recording / interval.
      Use this when each source file is a single contiguous trial.
    - ``trial``: each emitted segment becomes its own group. Useful for
      protocols where one long file contains many independent trials
      separated by rest periods (e.g. OpenBCI 1 s task / 5 s epoch x 60).
      With this, GroupShuffleSplit can scatter trials across train/val/
      test even when there is only one source file per class.
    """
    if group_by not in _VALID_GROUP_BY:
        raise ValueError(
            f"group_by must be one of {_VALID_GROUP_BY} (got {group_by!r})."
        )
    if group_by == "trial":
        return f"{source}#trial_{trial_idx}"
    return source


def _extract_signal(df: pd.DataFrame, signal_col_index: int) -> np.ndarray:
    """Pull a 1-D signal from a DataFrame by integer column index."""
    if df.empty:
        raise ValueError("CSV has no rows.")
    if signal_col_index < 0 or signal_col_index >= len(df.columns):
        raise ValueError(
            f"signal_col_index={signal_col_index} out of range (file has {len(df.columns)} columns)."
        )
    col = df.columns[signal_col_index]
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"Column '{col}' contains no numeric data.")
    return series.to_numpy(dtype=np.float32)


def _extract_multi_signal(
    df: pd.DataFrame, indices: Sequence[int]
) -> np.ndarray:
    """Pull one or more columns by integer index. Returns ``(T,)`` when
    ``len(indices) == 1`` and ``(T, C)`` when multi-channel.

    Rows where any selected column is non-numeric are dropped together so
    all channels stay time-aligned. This matters for OpenBCI exports
    where the last few rows of a recording are sometimes truncated.
    """
    if df.empty:
        raise ValueError("File has no rows.")
    n_cols = len(df.columns)
    bad = [i for i in indices if i < 0 or i >= n_cols]
    if bad:
        raise ValueError(
            f"signal_col_indices contains out-of-range entries {bad} "
            f"(file has {n_cols} columns)."
        )
    cols = [df.columns[i] for i in indices]
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if sub.empty:
        raise ValueError(
            f"Selected columns {cols} contain no aligned numeric rows."
        )
    arr = sub.to_numpy(dtype=np.float32)
    if len(indices) == 1:
        return arr[:, 0]
    return arr


def _rows_to_df(
    rows: list[np.ndarray],
    labels: list[str],
    seg_len: int,
    groups: list[str] | None = None,
    n_channels: int = 1,
) -> pd.DataFrame:
    """Stack segment rows into the trainer's flat tabular format.

    For ``n_channels == 1`` we keep the legacy ``s1..sN`` column names.
    For multi-channel we emit ``ch{c}_{i}`` columns so the dataset
    loader's channel-prefix regex auto-detects the structure and the
    trainer can reshape into ``(N, C, L)`` for the 1-D CNN.
    """
    if n_channels <= 1:
        arr = np.stack(rows).astype(np.float32)
        cols = [f"s{i+1}" for i in range(seg_len)]
    else:
        # Each row is shape (seg_len, n_channels). Stack to
        # (N, seg_len, n_channels), then transpose to (N, n_channels,
        # seg_len) and flatten so columns end up grouped by channel:
        # ch1_1..ch1_S, ch2_1..ch2_S, ...
        stacked = np.stack(rows).astype(np.float32)
        if stacked.ndim != 3 or stacked.shape[1] != seg_len or stacked.shape[2] != n_channels:
            raise ValueError(
                f"Unexpected multi-channel segment shape {stacked.shape}; "
                f"expected (N, {seg_len}, {n_channels})."
            )
        transposed = np.transpose(stacked, (0, 2, 1))
        arr = transposed.reshape(stacked.shape[0], n_channels * seg_len)
        cols = []
        for ch in range(1, n_channels + 1):
            cols.extend([f"ch{ch}_{i+1}" for i in range(seg_len)])
    out = pd.DataFrame(arr, columns=cols)
    out["label"] = labels
    # __group__ is a recording / trial id used by the trainer for group-aware
    # train/val/test splits — segments from the same source recording must
    # not leak across splits, otherwise validation accuracy is inflated.
    if groups is not None:
        out["__group__"] = groups
    return out


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))


_OPENBCI_NON_SIGNAL_HINTS = (
    "sample index", "timestamp", "accel channel", "analog channel",
    "other", "marker", "raw_uv", "ax", "ay", "az",
)


def _is_likely_non_signal(col_name: str) -> bool:
    """Heuristic for OpenBCI bookkeeping columns we should skip when
    auto-suggesting signal channels."""
    name = str(col_name).strip().lower()
    if name in _TIME_COL_NAMES:
        return True
    return any(hint in name for hint in _OPENBCI_NON_SIGNAL_HINTS)


def _is_numeric_column(df: pd.DataFrame, col: str) -> bool:
    try:
        pd.to_numeric(df[col].head(20), errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _is_useless_signal_column(df: pd.DataFrame, col: str, sample_rows: int = 500) -> bool:
    """True if the column is numeric but carries no signal:

    - Constant or near-constant (one value dominates ≥95% of samples).
      Catches disconnected channels saturated at the ADC rail (OpenBCI
      reports ``-187500.02`` for unplugged EXG inputs) and unmoving
      accelerometer axes.
    - Monotonically-increasing or 0–255 wrap-around integer
      sample-index column. OpenBCI's leading column counts samples
      mod 256 — so ``nunique`` for the first 500 rows is exactly 256
      and the diff sequence is mostly +1 with periodic ``-255`` jumps.

    Sniffs only the first ``sample_rows`` rows for speed; OpenBCI files
    can be hundreds of thousands of rows long.
    """
    try:
        s = pd.to_numeric(df[col].head(sample_rows), errors="coerce").dropna()
    except Exception:  # noqa: BLE001
        return True
    if s.empty:
        return True

    # Modal-fraction saturation check. Catches a single rail value
    # dominating the column (e.g. -187500.02 for ≥99% of rows after
    # the first sample).
    counts = s.value_counts(normalize=True)
    if counts.iloc[0] >= 0.95:
        return True

    arr = s.to_numpy()
    if arr.size >= 2:
        # Pure monotonic +1 (rare in practice — OpenBCI wraps).
        diffs = np.diff(arr)
        if np.all(diffs == 1) and float(arr[0]).is_integer():
            return True
        # OpenBCI sample-index pattern: int values in [0, 255] cycling
        # repeatedly. Diff is +1 most of the time with periodic -255
        # resets at the wrap.
        if (
            np.issubdtype(arr.dtype, np.integer)
            or (np.all(np.equal(np.mod(arr, 1), 0)) and arr.min() >= 0)
        ):
            ints = arr.astype(np.int64)
            if ints.min() >= 0 and ints.max() <= 255:
                d = np.diff(ints)
                if np.mean(d == 1) > 0.95:  # ≥95% +1 steps + occasional resets
                    return True
    return False


def _guess_signal_column(df: pd.DataFrame) -> int:
    """Pick the first numeric column whose name isn't a time/index marker."""
    for i, col in enumerate(df.columns):
        if _is_likely_non_signal(col):
            continue
        if _is_numeric_column(df, col):
            return i
    return 0


def _guess_signal_columns(df: pd.DataFrame) -> list[int]:
    """Return a list of column indices that look like signal channels.

    Strategy:
      1. If any column name contains "EXG Channel" / "EEG Channel" /
         "EMG Channel" (OpenBCI convention), pick exactly those — that's
         the strongest signal we have.
      2. Otherwise return the contiguous run of numeric, non-bookkeeping
         columns. The run must contain ≥1 entry; when nothing qualifies
         we fall back to the single best guess from
         ``_guess_signal_column``.
    """
    cols = list(df.columns)
    explicit = [
        i for i, c in enumerate(cols)
        if any(tag in str(c).lower() for tag in ("exg channel", "eeg channel", "emg channel"))
    ]
    if explicit:
        return explicit

    # Contiguous numeric, non-bookkeeping, non-useless run. "Useless"
    # filters out the OpenBCI sample-index column, zero-variance accel
    # channels, and disconnected channels saturated at the ADC rail —
    # all of which are technically numeric but carry no signal.
    runs: list[list[int]] = []
    cur: list[int] = []
    for i, c in enumerate(cols):
        if (
            _is_likely_non_signal(c)
            or not _is_numeric_column(df, c)
            or _is_useless_signal_column(df, c)
        ):
            if cur:
                runs.append(cur)
                cur = []
            continue
        cur.append(i)
    if cur:
        runs.append(cur)

    if runs:
        # Pick the longest run; ties broken by earliest start.
        runs.sort(key=lambda r: (-len(r), r[0]))
        return runs[0]

    return [_guess_signal_column(df)]


def _guess_sampling_rate(df: pd.DataFrame) -> float | None:
    """Infer SR from a time-like column, if any."""
    for col in df.columns:
        cl = str(col).strip().lower()
        if cl not in _TIME_COL_NAMES:
            continue
        try:
            t = pd.to_numeric(df[col].head(64), errors="raise").to_numpy()
        except (ValueError, TypeError):
            continue
        if len(t) < 2:
            continue
        dt = float(np.median(np.diff(t)))
        if dt <= 0:
            continue
        # Heuristic: 'ms' column → SR is 1000/dt; 'sec'/'time' → 1/dt
        if "ms" in cl:
            return round(1000.0 / dt, 3)
        return round(1.0 / dt, 3)
    return None
