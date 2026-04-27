"""
Data preparation router.

Lets users turn raw uploads (zip of recordings, single long CSV, generic zip)
into a training-ready CSV by filling in metadata: sampling rate, segment
length, signal column, and (per mode) labels.

Three modes (selected client-side):
  - A: ZIP of class folders, each CSV = one long recording
  - B: single long CSV + label time intervals
  - C: generic ZIP + per-file label mapping
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field, ValidationError

from backend.services import data_preparator as dp
from backend.services import dataset_cache
from backend.services.dataset_loader import load_labeled_dataframe

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
_PREP_CACHE_MAX = 16
_prep_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PrepConfig(BaseModel):
    """Discriminated by `mode`. Frontend posts this as a JSON string in form data."""
    mode: str = Field(pattern="^[ABC]$")
    sampling_rate: float = Field(gt=0)
    segment_length_sec: float = Field(gt=0)
    overlap_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)
    signal_col_index: int = Field(default=0, ge=0)
    # Mode B
    intervals: Optional[list[dict]] = None
    # Mode C
    file_label_map: Optional[dict[str, str]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/prep/inspect")
async def inspect(file: UploadFile = File(...)):
    """Peek at the uploaded file and return metadata for form prefill."""
    filename = file.filename or "upload"
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum 200 MB.")
    try:
        info = dp.inspect_upload(filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("inspect failed for %s", filename)
        raise HTTPException(status_code=500, detail=f"Inspect failed: {exc}")
    info["filename"] = filename
    return info


@router.post("/prep/segment")
async def segment(
    file: UploadFile = File(...),
    config: str = Form(..., description="JSON-encoded PrepConfig"),
):
    """Run segmentation per the chosen mode and stash the result for training."""
    try:
        cfg_dict = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid config JSON: {exc}")
    try:
        cfg = PrepConfig(**cfg_dict)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors())

    filename = file.filename or "upload"
    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum 200 MB.")

    try:
        if cfg.mode == "A":
            if not filename.lower().endswith(".zip"):
                raise HTTPException(status_code=422, detail="Mode A expects a .zip upload.")
            df = dp.segment_long_recordings(
                file_bytes,
                sampling_rate=cfg.sampling_rate,
                segment_length_sec=cfg.segment_length_sec,
                overlap_ratio=cfg.overlap_ratio,
                signal_col_index=cfg.signal_col_index,
            )
        elif cfg.mode == "B":
            if not (filename.lower().endswith(".csv") or filename.lower().endswith(".txt")):
                raise HTTPException(status_code=422, detail="Mode B expects a .csv upload.")
            if not cfg.intervals:
                raise HTTPException(status_code=422, detail="Mode B requires `intervals`.")
            df = dp.segment_with_intervals(
                file_bytes,
                intervals=cfg.intervals,
                sampling_rate=cfg.sampling_rate,
                segment_length_sec=cfg.segment_length_sec,
                overlap_ratio=cfg.overlap_ratio,
                signal_col_index=cfg.signal_col_index,
            )
        elif cfg.mode == "C":
            if not filename.lower().endswith(".zip"):
                raise HTTPException(status_code=422, detail="Mode C expects a .zip upload.")
            if not cfg.file_label_map:
                raise HTTPException(status_code=422, detail="Mode C requires `file_label_map`.")
            df = dp.segment_generic(
                file_bytes,
                file_label_map=cfg.file_label_map,
                signal_col_index=cfg.signal_col_index,
                sampling_rate=cfg.sampling_rate,
                segment_length_sec=cfg.segment_length_sec,
                overlap_ratio=cfg.overlap_ratio,
            )
        else:
            raise HTTPException(status_code=422, detail=f"Unknown mode: {cfg.mode}")
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("segmentation failed (mode=%s)", cfg.mode)
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {exc}")

    try:
        summary = load_labeled_dataframe(df)
    except Exception as exc:  # noqa: BLE001
        logger.exception("post-segmentation summary failed")
        raise HTTPException(status_code=500, detail=f"Output validation failed: {exc}")

    csv_bytes = dp.df_to_csv_bytes(df)
    dataset_id = "prep_" + uuid.uuid4().hex[:8]
    derived_filename = f"{Path(filename).stem}_prepped.csv"

    _prep_cache[dataset_id] = {
        "filename": derived_filename,
        "summary": summary,
        "csv_bytes": csv_bytes,
        "config": cfg.model_dump(),
    }
    while len(_prep_cache) > _PREP_CACHE_MAX:
        _prep_cache.popitem(last=False)

    preview_rows = df.head(10).to_dict(orient="records")
    return {
        "dataset_id": dataset_id,
        "filename": derived_filename,
        "summary": summary,
        "preview_rows": preview_rows,
        "row_count": len(df),
    }


@router.get("/prep/{dataset_id}/download")
async def download(dataset_id: str):
    entry = _prep_cache.get(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Prepared dataset not found or evicted.")
    data = entry["csv_bytes"]
    return Response(
        content=data,
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{entry["filename"]}"',
            "Content-Length": str(len(data)),
            "Cache-Control": "no-store",
        },
    )


@router.post("/prep/{dataset_id}/promote")
async def promote(dataset_id: str):
    """Register the prepared CSV in the shared training dataset cache."""
    entry = _prep_cache.get(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Prepared dataset not found or evicted.")
    dataset_cache.put(dataset_id, {
        "filename": entry["filename"],
        "summary": entry["summary"],
        "file_bytes": entry["csv_bytes"],
    })
    return {
        "dataset_id": dataset_id,
        "filename": entry["filename"],
        **entry["summary"],
    }
