"""
Figures router — publication-quality figure generation and download.

All endpoints require a completed training job and return either
PNG (300 DPI) or SVG figures styled for Nature, IEEE, or Science journals.
"""

import asyncio
import io
import logging
import traceback
import zipfile
from functools import partial

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

logger = logging.getLogger(__name__)

router = APIRouter()

_ALL_ZIP_TIMEOUT_SEC = 180

# Lazy imports to avoid loading torch/matplotlib at startup
_trainer_module = None
_figures_module = None


def _get_trainer():
    global _trainer_module
    if _trainer_module is None:
        from backend.services import trainer as _mod
        _trainer_module = _mod
    return _trainer_module


def _get_figures():
    global _figures_module
    if _figures_module is None:
        from backend.services import publication_figures as _mod
        _figures_module = _mod
    return _figures_module


def _require_completed_job(job_id: str):
    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    return job


def _media_type(fmt: str) -> str:
    return "image/svg+xml" if fmt == "svg" else "image/png"


async def _run_in_executor(fn, *args):
    """Run a blocking function in the default thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(fn, *args))


# ---------------------------------------------------------------------------
# Figure endpoints
# ---------------------------------------------------------------------------

@router.get("/train/{job_id}/figures/training_curves")
async def get_training_curves_figure(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return publication-quality training curves (loss + accuracy)."""
    job = _require_completed_job(job_id)
    if not job.history:
        raise HTTPException(status_code=409, detail="No training history available.")

    figs = _get_figures()
    data = await _run_in_executor(figs.render_training_curves, job.history, style, fmt)

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="training_curves_{job_id}.{fmt}"'},
    )


@router.get("/train/{job_id}/figures/confusion_matrix")
async def get_confusion_matrix_figure(
    job_id: str,
    mode: str = Query("both", pattern="^(count|normalized|both)$"),
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return publication-quality confusion matrix heatmap."""
    job = _require_completed_job(job_id)
    trainer = _get_trainer()
    figs = _get_figures()

    cm_data = trainer.compute_confusion_matrix(job)
    data = await _run_in_executor(figs.render_confusion_matrix, cm_data, mode, style, fmt)

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="confusion_matrix_{job_id}.{fmt}"'},
    )


@router.get("/train/{job_id}/figures/tsne")
async def get_tsne_figure(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return publication-quality t-SNE scatter plot."""
    job = _require_completed_job(job_id)
    trainer = _get_trainer()
    figs = _get_figures()

    tsne_data = trainer.compute_tsne(job)
    data = await _run_in_executor(figs.render_tsne, tsne_data, style, fmt)

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="tsne_{job_id}.{fmt}"'},
    )


@router.get("/train/{job_id}/figures/per_class_metrics")
async def get_per_class_metrics_figure(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return publication-quality per-class Precision/Recall/F1 bar chart."""
    job = _require_completed_job(job_id)
    trainer = _get_trainer()
    figs = _get_figures()

    cm_data = trainer.compute_confusion_matrix(job)
    data = await _run_in_executor(figs.render_per_class_metrics, cm_data, style, fmt)

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="per_class_metrics_{job_id}.{fmt}"'},
    )


@router.get("/train/{job_id}/figures/architecture")
async def get_architecture_figure(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
    fmt: str = Query("png", pattern="^(png|svg)$"),
):
    """Return model architecture diagram."""
    job = _require_completed_job(job_id)
    figs = _get_figures()

    if job.model is None:
        raise HTTPException(status_code=409, detail="Model not available.")

    # Reconstruct input shape from job metadata
    input_shape = (job.n_channels, job.val_X.shape[1] // max(job.n_channels, 1))
    data = await _run_in_executor(
        figs.render_architecture_diagram, job.model, input_shape, style, fmt,
    )

    return Response(
        content=data,
        media_type=_media_type(fmt),
        headers={"Content-Disposition": f'attachment; filename="architecture_{job_id}.{fmt}"'},
    )


def _build_figure_specs(job, trainer, figs, style: str):
    """Build (name, render_fn, args) triplets, skipping ones that need missing data.

    Each spec entry is independently renderable; errors during build (e.g. cm/tsne
    computation failure) are converted into a (name, None, error_message) sentinel
    so the spec still appears in the zip as a `.failed.txt` rather than aborting.
    """
    specs = []

    # 1. training_curves — needs job.history
    if job.history:
        specs.append(("training_curves", figs.render_training_curves, [job.history, style]))
    else:
        specs.append(("training_curves", None, "No training history available."))

    # 2. confusion_matrix + 4. per_class_metrics — both need cm_data
    try:
        cm_data = trainer.compute_confusion_matrix(job)
        specs.append(("confusion_matrix", figs.render_confusion_matrix, [cm_data, "both", style]))
        specs.append(("per_class_metrics", figs.render_per_class_metrics, [cm_data, style]))
    except Exception as exc:  # noqa: BLE001
        logger.exception("compute_confusion_matrix failed for job %s", job.job_id)
        msg = f"Confusion matrix computation failed: {exc}"
        specs.append(("confusion_matrix", None, msg))
        specs.append(("per_class_metrics", None, msg))

    # 3. tsne
    try:
        tsne_data = trainer.compute_tsne(job)
        specs.append(("tsne", figs.render_tsne, [tsne_data, style]))
    except Exception as exc:  # noqa: BLE001
        logger.exception("compute_tsne failed for job %s", job.job_id)
        specs.append(("tsne", None, f"t-SNE computation failed: {exc}"))

    # 5. architecture — needs job.model and val_X for input shape
    if job.model is None:
        specs.append(("architecture", None, "Model not available."))
    elif getattr(job, "val_X", None) is None:
        specs.append(("architecture", None, "Validation tensor not retained; cannot infer input shape."))
    else:
        try:
            input_shape = (job.n_channels, job.val_X.shape[1] // max(job.n_channels, 1))
            specs.append(("architecture", figs.render_architecture_diagram,
                          [job.model, input_shape, style]))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Architecture input-shape derivation failed for job %s", job.job_id)
            specs.append(("architecture", None, f"Architecture setup failed: {exc}"))

    return specs


async def _render_one(name: str, render_fn, args, fmt: str):
    """Render a single figure, returning (filename, bytes_or_none, error_or_none)."""
    try:
        data = await _run_in_executor(render_fn, *args, fmt)
        return f"{name}.{fmt}", data, None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Render failed: %s.%s", name, fmt)
        tb = traceback.format_exc()
        return f"{name}.{fmt}", None, f"{exc}\n\n{tb}"


@router.get("/train/{job_id}/figures/all.zip")
async def download_all_figures(
    job_id: str,
    style: str = Query("nature", pattern="^(nature|ieee|science)$"),
):
    """Download all 5 figures in both PNG and SVG formats as a ZIP archive.

    Tolerant to partial failures: if any individual figure render fails,
    a `<name>.<fmt>.failed.txt` entry is written instead, and the README
    summarizes which artifacts succeeded.
    """
    try:
        job = _require_completed_job(job_id)
        trainer = _get_trainer()
        figs = _get_figures()

        specs = _build_figure_specs(job, trainer, figs, style)

        # Schedule renders concurrently. Setup-failed specs (render_fn is None)
        # become deterministic failure entries.
        pending = []
        for name, render_fn, args_or_err in specs:
            for fmt in ("png", "svg"):
                if render_fn is None:
                    pending.append(asyncio.sleep(0, result=(f"{name}.{fmt}", None, str(args_or_err))))
                else:
                    pending.append(_render_one(name, render_fn, args_or_err, fmt))

        try:
            results = await asyncio.wait_for(asyncio.gather(*pending), timeout=_ALL_ZIP_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.exception("download_all_figures timed out after %ss for job %s",
                             _ALL_ZIP_TIMEOUT_SEC, job_id)
            raise HTTPException(
                status_code=504,
                detail=f"Figure rendering exceeded {_ALL_ZIP_TIMEOUT_SEC}s. Try downloading figures individually.",
            )

        ok_count = sum(1 for _, data, _ in results if data is not None)
        total = len(results)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, data, err in results:
                if data is not None:
                    zf.writestr(filename, data)
                else:
                    zf.writestr(f"{filename}.failed.txt",
                                f"Failed to render {filename}\n\n{err}\n")

            best_acc = f"{job.best_val_acc:.4f}" if job.best_val_acc is not None else "n/a"
            zf.writestr("README.txt", (
                f"BioSpark Publication Figures\n"
                f"Job ID: {job_id}\n"
                f"Style: {style}\n"
                f"Best Val Accuracy: {best_acc}\n"
                f"Generated artifacts: {ok_count}/{total}\n\n"
                f"Files:\n"
                f"  training_curves.png/svg  — Loss + Accuracy training curves\n"
                f"  confusion_matrix.png/svg — Confusion matrix heatmap (count + normalized)\n"
                f"  tsne.png/svg             — t-SNE feature space visualization\n"
                f"  per_class_metrics.png/svg— Per-class Precision / Recall / F1 bar chart\n"
                f"  architecture.png/svg     — Model architecture diagram\n\n"
                f"Any '.failed.txt' file indicates the corresponding figure could not be\n"
                f"rendered; the file contains the error message.\n\n"
                f"Generated by BioSpark — AI-Powered Biosignal Analysis Platform\n"
            ))

        data = buf.getvalue()
        return Response(
            content=data,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="biospark_figures_{job_id}.zip"',
                "Content-Length": str(len(data)),
                "Cache-Control": "no-store",
            },
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("download_all_figures unexpected failure for job %s", job_id)
        raise HTTPException(status_code=500, detail=f"Figure bundle failed: {exc}")
