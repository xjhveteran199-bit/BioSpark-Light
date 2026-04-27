"""
Dataset loader for labeled biosignal training data.

Supported formats:
  - CSV with a 'label' (or 'class'/'target'/'y') column
  - ZIP archive with folder-per-class structure: class_name/sample.csv

Multi-channel support:
  - Auto-detects channel-prefixed columns (e.g. ch1_1, ch1_2, ch2_1, ch2_2)
  - Falls back to single-channel if no prefix pattern found
"""

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# Columns that are treated as time/index rather than signal data
_TIME_COL_NAMES = {"time", "t", "timestamp", "seconds", "sec", "ms", "index"}
# Columns that are treated as labels
_LABEL_COL_NAMES = {"label", "class", "target", "y"}

# Regex for channel-prefixed column names: ch1_1, ch2_10, channel1_5, CH3_001, c1_1
_CHANNEL_RE = re.compile(r"^(ch(?:annel)?[\s_-]?\d+)[\s_-](\d+)$", re.IGNORECASE)


def _signal_cols(df: pd.DataFrame, exclude: list[str]) -> list[str]:
    """Return column names that are signal data (not time, not excluded)."""
    return [
        c for c in df.columns
        if c not in exclude and c.strip().lower() not in _TIME_COL_NAMES
    ]


def _detect_channel_structure(sig_cols: list[str]) -> dict:
    """Detect channel-prefixed columns and return channel structure info.

    Returns dict with keys: detected, n_channels, samples_per_channel, channel_map.
    channel_map is an ordered dict: {prefix: [col_names_in_order]}.
    """
    channel_map: dict[str, list[tuple[int, str]]] = {}

    for col in sig_cols:
        m = _CHANNEL_RE.match(col.strip())
        if m is None:
            return {"detected": False, "n_channels": 1, "samples_per_channel": len(sig_cols), "channel_map": {}}
        prefix = m.group(1).lower()
        idx = int(m.group(2))
        channel_map.setdefault(prefix, []).append((idx, col))

    if not channel_map:
        return {"detected": False, "n_channels": 1, "samples_per_channel": len(sig_cols), "channel_map": {}}

    # Sort each channel's columns by sample index
    ordered_map: dict[str, list[str]] = {}
    counts = set()
    for prefix in sorted(channel_map.keys()):
        entries = sorted(channel_map[prefix], key=lambda t: t[0])
        ordered_map[prefix] = [col for _, col in entries]
        counts.add(len(entries))

    if len(counts) != 1:
        return {"detected": False, "n_channels": 1, "samples_per_channel": len(sig_cols), "channel_map": {}}

    n_channels = len(ordered_map)
    samples_per_channel = counts.pop()

    return {
        "detected": True,
        "n_channels": n_channels,
        "samples_per_channel": samples_per_channel,
        "channel_map": ordered_map,
    }


def _parse_labeled_csv(df: pd.DataFrame) -> dict:
    """
    Parse a DataFrame where each row is a timestep annotated with a label.
    The label column must be named 'label', 'class', 'target', or 'y'.
    """
    # Find label column (case-insensitive)
    label_col = None
    for col in df.columns:
        if col.strip().lower() in _LABEL_COL_NAMES:
            label_col = col
            break

    if label_col is None:
        raise ValueError(
            "No label column found. Expected a column named: "
            "'label', 'class', 'target', or 'y'."
        )

    sig_cols = _signal_cols(df, exclude=[label_col])
    if not sig_cols:
        raise ValueError(
            "No signal columns found after excluding the label and time columns."
        )

    labels = df[label_col].astype(str).tolist()
    class_names = sorted(set(labels))
    class_counts = {cls: labels.count(cls) for cls in class_names}

    signal_values = df[sig_cols].values
    preview_vals = signal_values[:500, 0].tolist() if len(signal_values) > 0 else []

    ch_info = _detect_channel_structure(sig_cols)

    # Data preview: first 5 rows, limited columns for readability
    preview_cols = sig_cols[:6] + (["..."] if len(sig_cols) > 6 else []) + [label_col]
    data_preview_rows = []
    for _, row in df.head(5).iterrows():
        r = {}
        for c in sig_cols[:6]:
            v = row[c]
            r[c] = round(float(v), 4) if pd.notna(v) else None
        if len(sig_cols) > 6:
            r["..."] = "..."
        r[label_col] = str(row[label_col])
        data_preview_rows.append(r)

    return {
        "format": "csv_labeled",
        "label_column": label_col,
        "signal_columns": sig_cols,
        "class_names": class_names,
        "class_counts": class_counts,
        "total_samples": len(df),
        "signal_length": ch_info["samples_per_channel"],
        "n_channels": ch_info["n_channels"],
        "total_signal_cols": len(sig_cols),
        "channel_detected": ch_info["detected"],
        "channel_map": ch_info["channel_map"],
        "preview": {
            "values": preview_vals,
            "channel_name": sig_cols[0],
        },
        "data_preview": {
            "columns": preview_cols,
            "rows": data_preview_rows,
        },
    }


def _parse_zip_dataset(zip_bytes: bytes) -> dict:
    """
    Parse a ZIP archive with folder-per-class structure.
    Expected layout: <class_name>/<any_depth>/sample.csv
    The immediate parent folder of each CSV is used as the class name.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()

        # Map class_name → list of zip entry paths
        class_files: dict[str, list[str]] = {}
        for name in names:
            parts = Path(name).parts
            # Skip macOS metadata, hidden entries, and non-CSV files
            if any(p.startswith("__") or p.startswith(".") for p in parts):
                continue
            if not name.lower().endswith(".csv"):
                continue
            if len(parts) < 2:
                continue  # CSV at root — no class folder
            class_name = parts[-2]  # immediate parent folder = class label
            class_files.setdefault(class_name, []).append(name)

        if not class_files:
            raise ValueError(
                "No class folders with CSV files found in the ZIP archive. "
                "Expected structure: class_name/sample.csv"
            )

        class_names = sorted(class_files.keys())
        class_counts = {cls: len(files) for cls, files in class_files.items()}
        total_samples = sum(class_counts.values())

        # Sample the first file to get signal shape
        first_file = class_files[class_names[0]][0]
        with zf.open(first_file) as f:
            sample_df = pd.read_csv(f)

        sig_cols = _signal_cols(sample_df, exclude=[])
        ch_info = _detect_channel_structure(sig_cols)

        preview_vals = (
            sample_df[sig_cols[0]].values[:500].tolist() if sig_cols else []
        )

        return {
            "format": "zip_folder",
            "signal_columns": sig_cols,
            "class_names": class_names,
            "class_counts": class_counts,
            "total_samples": total_samples,
            "signal_length": ch_info["samples_per_channel"],
            "n_channels": ch_info["n_channels"],
            "total_signal_cols": len(sig_cols),
            "channel_detected": ch_info["detected"],
            "channel_map": ch_info["channel_map"],
            "preview": {
                "values": preview_vals,
                "channel_name": sig_cols[0] if sig_cols else "ch0",
            },
        }


def load_labeled_dataframe(df: pd.DataFrame) -> dict:
    """Parse an in-memory labeled DataFrame.

    Same return shape as load_labeled_dataset for the CSV path. Used by the
    data preparation pipeline to avoid serializing → re-parsing CSVs.
    """
    return _parse_labeled_csv(df)


def load_labeled_dataset(filename: str, file_bytes: bytes) -> dict:
    """
    Parse a labeled biosignal dataset from raw file bytes.

    Parameters
    ----------
    filename : str
        Original filename (used to detect format via extension).
    file_bytes : bytes
        Raw file content.

    Returns
    -------
    dict with keys:
        format, class_names, class_counts, total_samples,
        signal_length, n_channels, signal_columns, preview
    """
    ext = Path(filename).suffix.lower()

    if ext == ".zip":
        return _parse_zip_dataset(file_bytes)

    if ext in (".csv", ".txt"):
        text = file_bytes.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(text))
        return _parse_labeled_csv(df)

    raise ValueError(
        f"Unsupported file format: '{ext}'. "
        "Upload a CSV file (with a 'label' column) or a ZIP archive "
        "(with folder-per-class structure)."
    )
