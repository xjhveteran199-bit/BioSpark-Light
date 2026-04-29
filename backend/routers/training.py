"""
Training router — BioSpark-Light (single-user, no auth).

All endpoints behave as if there is exactly one local user (user_id=0).
The warm-start chain still works — it's just always for the same uid.
"""

import asyncio
import csv
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import desc, select, update

from backend.config import CHECKPOINTS_DIR
from backend.database import async_session
from backend.models.training_history import ModelCheckpoint, TrainingRun
from backend.services.dataset_loader import load_labeled_dataset
from backend.services import dataset_cache

# Single-user local app — every job is scoped to this sentinel uid.
_DEFAULT_USER_ID = 0

# Lazy-import trainer (depends on torch — keep startup fast)
_trainer_module = None


def _get_trainer():
    global _trainer_module
    if _trainer_module is None:
        from backend.services import trainer as _mod
        _trainer_module = _mod
    return _trainer_module


router = APIRouter()

_dataset_cache = dataset_cache._dataset_cache  # alias for prep router compat

MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB


# ---------------------------------------------------------------------------
# Training Presets
# ---------------------------------------------------------------------------

_PRESETS = {
    "auto": {
        "label_en": "Smart Auto (Recommended)",
        "label_zh": "智能自动（推荐）",
        "description_en": "Auto-picks architecture, applies class weights, and stops when performance plateaus. LR search off by default — toggle it on under Advanced for an extra +2-3 min.",
        "description_zh": "自动选择网络结构 + 类别加权 + 早停。默认跳过学习率搜索，可在高级选项中开启（额外 +2-3 分钟）。",
        "time_estimate_en": "3–10 min · CPU",
        "time_estimate_zh": "约 3–10 分钟 · CPU",
        "epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "val_split": 0.2,
        "test_split": 0.2,
        "auto_mode": True,
        "lr_search": False,
        "early_stopping_patience": 12,
        "use_class_weights": True,
    },
    "fast": {
        "label_en": "Quick Test",
        "label_zh": "快速测试",
        "description_en": "20 epochs without LR search. Useful for checking data quality before committing to a full run.",
        "description_zh": "20 轮训练，跳过学习率搜索。适合快速验证数据质量。",
        "time_estimate_en": "30 sec – 2 min · CPU",
        "time_estimate_zh": "约 30 秒 – 2 分钟 · CPU",
        "epochs": 20,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "val_split": 0.2,
        "test_split": 0.2,
        "auto_mode": False,
        "lr_search": False,
        "early_stopping_patience": 8,
        "use_class_weights": True,
    },
    "thorough": {
        "label_en": "Publication Ready",
        "label_zh": "发表级别",
        "description_en": "100 epochs with full auto-optimization including LR search. Maximizes accuracy — use this for your final submitted model.",
        "description_zh": "100 轮完整自动优化（含学习率搜索），最大化准确率。适合最终投稿前的训练。",
        "time_estimate_en": "15–45 min · CPU (incl. LR search)",
        "time_estimate_zh": "约 15–45 分钟 · CPU（含学习率搜索）",
        "epochs": 100,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "val_split": 0.2,
        "test_split": 0.2,
        "auto_mode": True,
        "lr_search": True,
        "early_stopping_patience": 20,
        "use_class_weights": True,
    },
    "custom": {
        "label_en": "Custom",
        "label_zh": "自定义",
        "description_en": "Set all hyperparameters manually.",
        "description_zh": "手动设置所有超参数。",
        "time_estimate_en": "Depends on your settings",
        "time_estimate_zh": "取决于您的设置",
        "epochs": 30,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "val_split": 0.2,
        "test_split": 0.2,
        "auto_mode": False,
        "lr_search": False,
        "early_stopping_patience": 10,
        "use_class_weights": True,
    },
}


@router.get("/train/presets")
async def get_training_presets():
    return {"presets": _PRESETS}


# ---------------------------------------------------------------------------
# Phase 1 — Upload
# ---------------------------------------------------------------------------

@router.post("/train/upload")
async def upload_training_data(file: UploadFile = File(...)):
    """Upload a labeled dataset (CSV with `label` column or folder-per-class ZIP)."""
    filename = file.filename or "dataset"
    ext = _path_ext(filename)

    if ext not in ("csv", "txt", "zip"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '.{ext}'. Upload a .csv or .zip file.",
        )

    file_bytes = await file.read()

    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum 200 MB.")

    try:
        summary = load_labeled_dataset(filename, file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse dataset: {exc}")

    dataset_id = str(uuid.uuid4())[:8]
    _dataset_cache[dataset_id] = {
        "filename": filename,
        "summary": summary,
        "file_bytes": file_bytes,
    }

    return {"dataset_id": dataset_id, "filename": filename, **summary}


@router.get("/train/dataset/{dataset_id}")
async def get_dataset_info(dataset_id: str):
    entry = _dataset_cache.get(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return {"dataset_id": dataset_id, "filename": entry["filename"], **entry["summary"]}


# ---------------------------------------------------------------------------
# Phase 2 — Training
# ---------------------------------------------------------------------------

class TrainStartRequest(BaseModel):
    dataset_id: str
    preset: str = Field(default="custom", description="Preset name: auto/fast/thorough/custom")
    epochs: int = Field(default=30, ge=1, le=200)
    learning_rate: float = Field(default=1e-3, gt=0, le=1.0)
    batch_size: int = Field(default=64, ge=4, le=512)
    val_split: float = Field(default=0.2, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.2, ge=0.0, lt=0.9)
    n_channels: int = Field(default=0, ge=0, le=64)
    auto_mode: bool = Field(default=False)
    early_stopping_patience: int = Field(default=10, ge=3, le=50)
    use_class_weights: bool = Field(default=True)
    lr_search: Optional[bool] = Field(default=None)
    warm_start: bool = Field(default=False)


@router.post("/train/assess")
async def assess_dataset(dataset_id: str):
    entry = _dataset_cache.get(dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")

    try:
        from backend.services.auto_optimizer import DataQualityAssessor
        trainer = _get_trainer()
        X, y, class_names, _groups = trainer._dataset_to_tensors(
            entry["file_bytes"], entry["filename"], entry["summary"]
        )
        result = DataQualityAssessor().assess(X, y, class_names)
        return {"dataset_id": dataset_id, **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {exc}")


async def _resolve_warm_start(
    user_id: int,
    summary: dict,
) -> tuple[Optional[str], Optional[int]]:
    n_classes = len(summary.get("class_names", []) or [])  # noqa: F841
    n_channels = summary.get("n_channels", 1) or 1
    async with async_session() as db:
        stmt = (
            select(ModelCheckpoint)
            .where(ModelCheckpoint.user_id == user_id)
            .order_by(desc(ModelCheckpoint.is_active), desc(ModelCheckpoint.version))
        )
        rows = (await db.execute(stmt)).scalars().all()
        for ckpt in rows:
            shape = ckpt.input_shape or {}
            if shape.get("n_channels", 1) != n_channels:
                continue
            if not Path(ckpt.file_path).exists():
                continue
            return ckpt.file_path, ckpt.id
    return None, None


def _make_on_complete(
    user_id: int,
    job_id: str,
    summary: dict,
    config: dict,
    warm_started_from_id: Optional[int],
):
    async def _persist(job):
        try:
            import torch

            user_dir = CHECKPOINTS_DIR / str(user_id)
            user_dir.mkdir(parents=True, exist_ok=True)

            async with async_session() as db:
                last = await db.execute(
                    select(ModelCheckpoint.version)
                    .where(ModelCheckpoint.user_id == user_id)
                    .order_by(desc(ModelCheckpoint.version))
                    .limit(1)
                )
                last_version = last.scalar_one_or_none() or 0
                next_version = last_version + 1

                file_path = user_dir / f"v{next_version}.pt"

                payload = {
                    "state_dict": job.model.state_dict() if job.model is not None else {},
                    "n_classes": len(job.class_names),
                    "class_names": list(job.class_names),
                    "input_shape": {
                        "n_channels": int(job.n_channels),
                        "signal_length": int(job.signal_length or 0),
                    },
                    "arch_config": job.model.arch_config if job.model is not None else None,
                }
                torch.save(payload, str(file_path))

                run = TrainingRun(
                    user_id=user_id,
                    job_id=job_id,
                    dataset_summary=_jsonable(summary),
                    config=_jsonable(config),
                    best_val_acc=float(job.best_val_acc or 0.0),
                    status="completed",
                    warm_started_from_id=warm_started_from_id,
                    completed_at=datetime.now(timezone.utc),
                )
                db.add(run)
                await db.flush()

                await db.execute(
                    update(ModelCheckpoint)
                    .where(ModelCheckpoint.user_id == user_id, ModelCheckpoint.is_active == True)  # noqa: E712
                    .values(is_active=False)
                )

                ckpt = ModelCheckpoint(
                    user_id=user_id,
                    training_run_id=run.id,
                    version=next_version,
                    file_path=str(file_path),
                    n_classes=len(job.class_names),
                    class_names=list(job.class_names),
                    input_shape={
                        "n_channels": int(job.n_channels),
                        "signal_length": int(job.signal_length or 0),
                    },
                    best_val_acc=float(job.best_val_acc or 0.0),
                    is_active=True,
                )
                db.add(ckpt)
                await db.commit()
        except Exception:
            import traceback as _tb
            _tb.print_exc()

    return _persist


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not isinstance(v, (bytes, bytearray))}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


@router.post("/train/start")
async def start_training(req: TrainStartRequest):
    """
    Start a training job. Single-user local mode — always uses
    user_id=_DEFAULT_USER_ID for warm-start chain bookkeeping.
    """
    entry = _dataset_cache.get(req.dataset_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload first.")

    preset_cfg = _PRESETS.get(req.preset, _PRESETS["custom"])
    config = {
        "preset": req.preset,
        "epochs": req.epochs if req.preset == "custom" else preset_cfg["epochs"],
        "learning_rate": req.learning_rate if req.preset == "custom" else preset_cfg["learning_rate"],
        "batch_size": req.batch_size if req.preset == "custom" else preset_cfg["batch_size"],
        "val_split": req.val_split if req.preset == "custom" else preset_cfg["val_split"],
        "test_split": (
            req.test_split if req.preset == "custom"
            else preset_cfg.get("test_split", 0.2)
        ),
        "n_channels": req.n_channels,
        "auto_mode": req.auto_mode if req.preset == "custom" else preset_cfg["auto_mode"],
        "early_stopping_patience": req.early_stopping_patience if req.preset == "custom" else preset_cfg["early_stopping_patience"],
        "use_class_weights": req.use_class_weights if req.preset == "custom" else preset_cfg["use_class_weights"],
        "lr_search": (
            req.lr_search if req.lr_search is not None
            else preset_cfg.get("lr_search", False)
        ),
        "warm_start": req.warm_start,
    }

    job_id = str(uuid.uuid4())[:8]
    user_id = _DEFAULT_USER_ID

    warm_path: Optional[str] = None
    warm_from_id: Optional[int] = None
    if req.warm_start:
        try:
            warm_path, warm_from_id = await _resolve_warm_start(user_id, entry["summary"])
        except Exception:
            warm_path, warm_from_id = None, None

    on_complete = _make_on_complete(
        user_id=user_id,
        job_id=job_id,
        summary=entry["summary"],
        config=config,
        warm_started_from_id=warm_from_id,
    )

    try:
        _get_trainer().training_manager.start(
            job_id=job_id,
            file_bytes=entry["file_bytes"],
            filename=entry["filename"],
            summary=entry["summary"],
            config=config,
            user_id=user_id,
            warm_start_path=warm_path,
            warm_started_from_id=warm_from_id,
            on_complete=on_complete,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {exc}")

    return {
        "job_id": job_id,
        "status": "started",
        "config": config,
        "warm_started_from_id": warm_from_id,
    }


@router.get("/train/{job_id}/status")
async def get_training_status(job_id: str):
    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    return {
        "job_id": job_id,
        "status": job.status,
        "best_val_acc": job.best_val_acc,
        "history": job.history,
        "error": job.error,
        "class_names": job.class_names,
        "config": job.config,
    }


@router.websocket("/train/ws/{job_id}")
async def training_websocket(ws: WebSocket, job_id: str):
    await ws.accept()

    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        await ws.send_json({"type": "error", "message": "Job not found."})
        await ws.close()
        return

    loop = asyncio.get_event_loop()

    for past_metric in job.history:
        await ws.send_json(past_metric)

    if job.status == "completed":
        await ws.send_json({
            "type": "complete",
            "best_val_acc": round(job.best_val_acc, 5),
            "total_epochs": len(job.history),
        })
        await ws.close()
        return
    if job.status == "failed":
        await ws.send_json({"type": "error", "message": job.error or "Training failed"})
        await ws.close()
        return

    queue: asyncio.Queue = asyncio.Queue()

    async def _on_metric(payload: dict):
        await queue.put(payload)

    job.register_callback(_on_metric, loop)

    try:
        while True:
            payload = await queue.get()
            await ws.send_json(payload)
            if payload.get("type") in ("complete", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        job.unregister_callback(_on_metric)


# ---------------------------------------------------------------------------
# Phase 3 — Post-training visualizations
# ---------------------------------------------------------------------------

@router.get("/train/{job_id}/confusion_matrix")
async def get_confusion_matrix(job_id: str):
    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    try:
        return _get_trainer().compute_confusion_matrix(job)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/train/{job_id}/tsne")
async def get_tsne(job_id: str, perplexity: float = 30.0):
    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    try:
        return _get_trainer().compute_tsne(job, perplexity=perplexity)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# (Grad-CAM endpoint deliberately omitted from BioSpark-Light — strip XAI for v1.)


# ---------------------------------------------------------------------------
# Phase 4 — Export endpoints
# ---------------------------------------------------------------------------

def _require_completed_job(job_id: str):
    job = _get_trainer().training_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Training job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Training not complete (status: {job.status}).")
    return job


@router.get("/train/{job_id}/export/model")
async def export_model(job_id: str):
    job = _require_completed_job(job_id)
    buf = io.BytesIO()
    import torch
    torch.save(job.model.state_dict(), buf)
    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="biospark_model_{job_id}.pt"'},
    )


@router.get("/train/{job_id}/export/history")
async def export_history(job_id: str):
    job = _require_completed_job(job_id)
    payload = {
        "job_id": job_id,
        "config": job.config,
        "class_names": job.class_names,
        "best_val_acc": job.best_val_acc,
        "history": job.history,
    }
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="training_history_{job_id}.json"'},
    )


@router.get("/train/{job_id}/export/confusion_matrix_csv")
async def export_confusion_matrix_csv(job_id: str):
    job = _require_completed_job(job_id)
    data = _get_trainer().compute_confusion_matrix(job)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["# Confusion Matrix (rows=true, cols=predicted)"])
    writer.writerow([""] + data["class_names"])
    for i, row in enumerate(data["matrix"]):
        writer.writerow([data["class_names"][i]] + row)
    writer.writerow([])
    writer.writerow(["# Per-Class Metrics"])
    writer.writerow(["Class", "Precision", "Recall", "F1", "Support"])
    for c in data["per_class"]:
        writer.writerow([c["class"], c["precision"], c["recall"], c["f1"], c["support"]])
    writer.writerow(["Overall Accuracy", data["accuracy"], "", "", ""])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="confusion_matrix_{job_id}.csv"'},
    )


@router.get("/train/{job_id}/export/tsne_csv")
async def export_tsne_csv(job_id: str):
    job = _require_completed_job(job_id)
    data = _get_trainer().compute_tsne(job)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["x", "y", "label"])
    for x, y, label in zip(data["x"], data["y"], data["labels"]):
        writer.writerow([round(x, 6), round(y, 6), label])

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="tsne_{job_id}.csv"'},
    )


@router.get("/train/{job_id}/interpret")
async def interpret_results(job_id: str):
    job = _require_completed_job(job_id)

    try:
        cm_data = _get_trainer().compute_confusion_matrix(job)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot compute confusion matrix: {exc}")

    acc = cm_data["accuracy"]
    per_class = cm_data["per_class"]
    n_epochs = len(job.history)
    early_stopped = job.config.get("early_stopping_patience") is not None and n_epochs < job.config.get("epochs", 9999)

    if acc >= 0.92:
        readiness = "excellent"
        readiness_en = "Excellent results — your accuracy exceeds 92%, which is competitive in top-tier biosignal classification papers."
        readiness_zh = "优秀成绩——准确率超过 92%，在顶级生物信号分类论文中具有竞争力。"
    elif acc >= 0.85:
        readiness = "strong"
        readiness_en = "Strong results — accuracy above 85% is publishable. Consider k-fold validation to report confidence intervals."
        readiness_zh = "良好成绩——准确率高于 85% 可达到发表标准，建议进行 k 折交叉验证以报告置信区间。"
    elif acc >= 0.75:
        readiness = "moderate"
        readiness_en = "Moderate results — solid foundation. Try the 'Publication Ready' preset or add more labeled data to push above 85%."
        readiness_zh = "中等成绩——有一定基础。尝试\"发表级别\"预设或增加标注数据，争取超过 85%。"
    else:
        readiness = "weak"
        readiness_en = "Results need improvement. Check your data labels, consider collecting more samples, or use the Smart Auto preset."
        readiness_zh = "结果有待提升。请检查数据标签，考虑增加样本量，或使用\"智能自动\"预设。"

    worst = min(per_class, key=lambda c: c["f1"])
    worst_en = (
        f"Class '{worst['class']}' has the lowest F1 score ({worst['f1']*100:.1f}%). "
        f"It has {worst['support']} validation samples — collecting more data for this class would help."
    )
    worst_zh = (
        f"类别 '{worst['class']}' 的 F1 分数最低（{worst['f1']*100:.1f}%），"
        f"验证集中有 {worst['support']} 个样本。为该类别收集更多数据将有助于改善。"
    )

    if job.history:
        last = job.history[-1]
        overfit_gap = last.get("train_acc", 0) - last.get("val_acc", 0)
        if overfit_gap > 0.15:
            dynamics_en = f"Overfitting detected: training accuracy ({last['train_acc']*100:.1f}%) is much higher than validation ({last['val_acc']*100:.1f}%). Try adding more data augmentation or increasing dropout."
            dynamics_zh = f"检测到过拟合：训练准确率（{last['train_acc']*100:.1f}%）远高于验证准确率（{last['val_acc']*100:.1f}%）。建议增加数据增强或提高 Dropout。"
        else:
            dynamics_en = "Training and validation curves are well-aligned — no significant overfitting detected."
            dynamics_zh = "训练曲线与验证曲线吻合良好——未检测到明显过拟合。"
    else:
        dynamics_en = dynamics_zh = ""

    return {
        "job_id": job_id,
        "accuracy": acc,
        "readiness": readiness,
        "readiness_en": readiness_en,
        "readiness_zh": readiness_zh,
        "worst_class": worst["class"],
        "worst_class_f1": worst["f1"],
        "worst_class_advice_en": worst_en,
        "worst_class_advice_zh": worst_zh,
        "dynamics_en": dynamics_en,
        "dynamics_zh": dynamics_zh,
        "epochs_used": n_epochs,
        "early_stopped": early_stopped,
        "next_steps_en": [
            "Run k-fold cross-validation to get confidence intervals for your paper.",
            "Use statistical testing to compare against a baseline method.",
            "Export publication-quality figures for your manuscript.",
        ],
        "next_steps_zh": [
            "运行 k 折交叉验证，为论文获取置信区间。",
            "使用统计检验与基线方法进行比较。",
            "导出发表级别图表用于论文。",
        ],
    }


def _path_ext(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").lower()
