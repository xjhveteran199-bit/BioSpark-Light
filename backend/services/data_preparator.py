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
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from backend.services.preprocess import _segment

MAX_OUTPUT_ROWS = 50_000  # hard cap on generated samples to protect memory

_TIME_COL_NAMES = {"time", "t", "timestamp", "seconds", "sec", "ms", "index"}


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
    df = _read_csv_bytes(file_bytes)
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
        "suggested_sampling_rate": _guess_sampling_rate(df),
        "class_folders": None,
    }


def _inspect_zip(file_bytes: bytes) -> dict:
    files: list[dict] = []
    class_folders: set[str] = set()
    first_columns: list[str] = []
    first_rows = 0
    suggested_sig = 0
    suggested_sr: float | None = None

    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if any(p.startswith("__") or p.startswith(".") for p in parts):
                continue
            if not name.lower().endswith(".csv"):
                continue
            if len(parts) >= 2:
                class_folders.add(parts[-2])
            try:
                with zf.open(name) as fp:
                    df = pd.read_csv(fp)
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
                suggested_sr = _guess_sampling_rate(df)

    if not files:
        raise ValueError("ZIP contains no CSV files.")

    return {
        "kind": "zip",
        "files": files,
        "columns": first_columns,
        "row_count": first_rows,
        "suggested_signal_col": suggested_sig,
        "suggested_sampling_rate": suggested_sr,
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
) -> pd.DataFrame:
    """Mode A: ZIP with class_folder/recording.csv. One long recording per CSV.

    The immediate parent folder name becomes the label.
    """
    _validate_window(sampling_rate, segment_length_sec, overlap_ratio)
    seg_len = int(round(sampling_rate * segment_length_sec))

    rows: list[np.ndarray] = []
    labels: list[str] = []

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            parts = Path(name).parts
            if any(p.startswith("__") or p.startswith(".") for p in parts):
                continue
            if not name.lower().endswith(".csv"):
                continue
            if len(parts) < 2:
                continue
            label = parts[-2]
            try:
                with zf.open(name) as fp:
                    df = pd.read_csv(fp)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Failed to read '{name}': {exc}")
            sig = _extract_signal(df, signal_col_index)
            for seg in _segment(sig, seg_len, overlap_ratio):
                rows.append(seg)
                labels.append(label)
                if len(rows) > MAX_OUTPUT_ROWS:
                    raise ValueError(
                        f"Generated more than {MAX_OUTPUT_ROWS} segments — "
                        "increase segment length or reduce overlap."
                    )

    if not rows:
        raise ValueError(
            "No segments produced. Check sampling rate, segment length, and "
            "that each class folder contains CSVs longer than one segment."
        )

    return _rows_to_df(rows, labels, seg_len)


def segment_with_intervals(
    csv_bytes: bytes,
    intervals: list[dict],
    sampling_rate: float,
    segment_length_sec: float,
    overlap_ratio: float = 0.0,
    signal_col_index: int = 0,
) -> pd.DataFrame:
    """Mode B: single long CSV + label intervals.

    intervals: [{"start_sec": float, "end_sec": float, "label": str}, ...]
    """
    _validate_window(sampling_rate, segment_length_sec, overlap_ratio)
    if not intervals:
        raise ValueError("At least one interval is required.")
    seg_len = int(round(sampling_rate * segment_length_sec))

    df = _read_csv_bytes(csv_bytes)
    sig = _extract_signal(df, signal_col_index)
    duration_sec = len(sig) / sampling_rate

    rows: list[np.ndarray] = []
    labels: list[str] = []
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
        for seg in _segment(slice_sig, seg_len, overlap_ratio):
            rows.append(seg)
            labels.append(label)
            if len(rows) > MAX_OUTPUT_ROWS:
                raise ValueError(
                    f"Generated more than {MAX_OUTPUT_ROWS} segments — "
                    "increase segment length or reduce overlap."
                )

    if not rows:
        raise ValueError("No segments produced from the given intervals.")

    return _rows_to_df(rows, labels, seg_len)


def segment_generic(
    zip_bytes: bytes,
    file_label_map: dict[str, str],
    signal_col_index: int,
    sampling_rate: float,
    segment_length_sec: float,
    overlap_ratio: float = 0.0,
) -> pd.DataFrame:
    """Mode C: generic ZIP, user provides per-file label mapping.

    file_label_map: {"path/in/zip.csv": "ClassName", ...}. Files not in the
    map are skipped. Each file is treated as one long recording windowed at
    the given sampling rate / segment length.
    """
    _validate_window(sampling_rate, segment_length_sec, overlap_ratio)
    if not file_label_map:
        raise ValueError("file_label_map is empty — assign at least one file a label.")
    seg_len = int(round(sampling_rate * segment_length_sec))

    rows: list[np.ndarray] = []
    labels: list[str] = []
    matched_files: set[str] = set()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zip_files = {n for n in zf.namelist() if n.lower().endswith(".csv")}
        for fname, label in file_label_map.items():
            if not label or not str(label).strip():
                continue
            if fname not in zip_files:
                raise ValueError(f"File '{fname}' from mapping not found in ZIP.")
            with zf.open(fname) as fp:
                df = pd.read_csv(fp)
            sig = _extract_signal(df, signal_col_index)
            for seg in _segment(sig, seg_len, overlap_ratio):
                rows.append(seg)
                labels.append(str(label).strip())
                matched_files.add(fname)
                if len(rows) > MAX_OUTPUT_ROWS:
                    raise ValueError(
                        f"Generated more than {MAX_OUTPUT_ROWS} segments — "
                        "increase segment length or reduce overlap."
                    )

    if not rows:
        raise ValueError("No segments produced. Check labels and signal column.")

    return _rows_to_df(rows, labels, seg_len)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _validate_window(sr: float, seg_sec: float, overlap: float) -> None:
    if sr <= 0:
        raise ValueError(f"sampling_rate must be > 0 (got {sr}).")
    if seg_sec <= 0:
        raise ValueError(f"segment_length_sec must be > 0 (got {seg_sec}).")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap_ratio must be in [0, 1) (got {overlap}).")
    if int(round(sr * seg_sec)) < 4:
        raise ValueError("Resulting segment length is < 4 samples — adjust SR or segment length.")


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


def _rows_to_df(rows: list[np.ndarray], labels: list[str], seg_len: int) -> pd.DataFrame:
    arr = np.stack(rows).astype(np.float32)
    cols = [f"s{i+1}" for i in range(seg_len)]
    out = pd.DataFrame(arr, columns=cols)
    out["label"] = labels
    return out


def _read_csv_bytes(data: bytes) -> pd.DataFrame:
    text = data.decode("utf-8", errors="replace")
    return pd.read_csv(io.StringIO(text))


def _guess_signal_column(df: pd.DataFrame) -> int:
    """Pick the first numeric column whose name isn't a time/index marker."""
    for i, col in enumerate(df.columns):
        if str(col).strip().lower() in _TIME_COL_NAMES:
            continue
        try:
            pd.to_numeric(df[col].head(20), errors="raise")
            return i
        except (ValueError, TypeError):
            continue
    return 0


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
