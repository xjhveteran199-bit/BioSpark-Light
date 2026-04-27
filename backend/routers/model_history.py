"""
Per-user model history (Self-Improving model) — BioSpark-Light variant.

Single-user local app: every request is scoped to user_id=0. Endpoints kept
identical to the web version so the frontend module (`my_models.js`) does
not need to change.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.training_history import ModelCheckpoint, TrainingRun

router = APIRouter()

_DEFAULT_USER_ID = 0


@router.get("/models/history")
async def list_model_history(db: AsyncSession = Depends(get_db)):
    """List training runs + checkpoints for the local user."""
    user_id = _DEFAULT_USER_ID

    runs = (
        await db.execute(
            select(TrainingRun)
            .where(TrainingRun.user_id == user_id)
            .order_by(desc(TrainingRun.created_at))
        )
    ).scalars().all()

    ckpts = (
        await db.execute(
            select(ModelCheckpoint)
            .where(ModelCheckpoint.user_id == user_id)
            .order_by(desc(ModelCheckpoint.version))
        )
    ).scalars().all()

    return {
        "user_id": user_id,
        "runs": [
            {
                "id": r.id,
                "job_id": r.job_id,
                "best_val_acc": r.best_val_acc,
                "status": r.status,
                "warm_started_from_id": r.warm_started_from_id,
                "config": r.config,
                "dataset_summary": r.dataset_summary,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            }
            for r in runs
        ],
        "checkpoints": [
            {
                "id": c.id,
                "training_run_id": c.training_run_id,
                "version": c.version,
                "n_classes": c.n_classes,
                "class_names": c.class_names,
                "input_shape": c.input_shape,
                "best_val_acc": c.best_val_acc,
                "is_active": c.is_active,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in ckpts
        ],
    }


@router.post("/models/{checkpoint_id}/activate")
async def activate_checkpoint(checkpoint_id: int, db: AsyncSession = Depends(get_db)):
    user_id = _DEFAULT_USER_ID

    target = (
        await db.execute(
            select(ModelCheckpoint).where(
                ModelCheckpoint.id == checkpoint_id,
                ModelCheckpoint.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found.")

    await db.execute(
        update(ModelCheckpoint)
        .where(
            ModelCheckpoint.user_id == user_id,
            ModelCheckpoint.is_active == True,  # noqa: E712
        )
        .values(is_active=False)
    )
    target.is_active = True
    await db.commit()
    return {"id": checkpoint_id, "is_active": True}


@router.delete("/models/{checkpoint_id}")
async def delete_checkpoint(checkpoint_id: int, db: AsyncSession = Depends(get_db)):
    user_id = _DEFAULT_USER_ID

    target = (
        await db.execute(
            select(ModelCheckpoint).where(
                ModelCheckpoint.id == checkpoint_id,
                ModelCheckpoint.user_id == user_id,
            )
        )
    ).scalar_one_or_none()
    if target is None:
        raise HTTPException(status_code=404, detail="Checkpoint not found.")

    try:
        p = Path(target.file_path)
        if p.exists():
            p.unlink()
    except Exception:
        pass

    was_active = target.is_active
    await db.delete(target)
    await db.commit()

    if was_active:
        latest = (
            await db.execute(
                select(ModelCheckpoint)
                .where(ModelCheckpoint.user_id == user_id)
                .order_by(desc(ModelCheckpoint.version))
                .limit(1)
            )
        ).scalar_one_or_none()
        if latest is not None:
            latest.is_active = True
            await db.commit()

    return {"id": checkpoint_id, "deleted": True}
