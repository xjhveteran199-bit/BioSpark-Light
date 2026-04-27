"""
Database models for self-improving model history.

TrainingRun: one row per training job (per user), tracks dataset summary,
config, achieved accuracy, and the warm-start checkpoint it was seeded from
(if any).

ModelCheckpoint: persisted model artifact on disk, with input shape & class
metadata for warm-start compatibility checks. At most one row per user is
flagged is_active=True (the default warm-start source).
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)

    dataset_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    best_val_acc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False)

    warm_started_from_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_checkpoints.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    checkpoint = relationship(
        "ModelCheckpoint",
        back_populates="training_run",
        foreign_keys="ModelCheckpoint.training_run_id",
        uselist=False,
    )


class ModelCheckpoint(Base):
    __tablename__ = "model_checkpoints"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    training_run_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("training_runs.id", ondelete="SET NULL"),
        nullable=True,
    )

    version: Mapped[int] = mapped_column(Integer, nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)

    n_classes: Mapped[int] = mapped_column(Integer, nullable=False)
    class_names: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    input_shape: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    best_val_acc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    training_run = relationship(
        "TrainingRun",
        back_populates="checkpoint",
        foreign_keys=[training_run_id],
    )
