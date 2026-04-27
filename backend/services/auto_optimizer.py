"""
Auto-optimization utilities for the training pipeline.

Provides:
- LR range test (Smith 2017) to find optimal learning rate
- Class weight computation for imbalanced datasets
- Rule-based architecture selection based on data properties
- Early stopping with best-model checkpointing
"""

import copy
import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# LR Range Test
# ---------------------------------------------------------------------------

def lr_range_test(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    start_lr: float = 1e-6,
    end_lr: float = 1.0,
    num_iter: int = 100,
    on_progress: Optional[Callable[[int, int, float, float], None]] = None,
) -> float:
    """Run a mini LR range test (Smith 2017) and return the suggested LR.

    Linearly sweeps the learning rate over *num_iter* mini-batches while
    recording loss.  Returns the LR at the steepest negative loss gradient
    (the "elbow"), which is typically a good starting point for training.

    The model's ``state_dict`` is saved before the test and restored
    afterwards so that the actual training starts from untouched weights.

    ``on_progress(iter_idx, total, current_lr, smoothed_loss)`` is invoked
    every 5 iterations (and on the last one) so the caller can stream a
    progress bar to the UI — without it, the ~2-3 minute test feels like a
    hang on a busy CPU.
    """
    saved_state = copy.deepcopy(model.state_dict())

    optimizer = torch.optim.SGD(model.parameters(), lr=start_lr)
    lr_mult = (end_lr / start_lr) ** (1.0 / max(num_iter - 1, 1))

    lrs, losses = [], []
    smoothed_loss = 0.0
    best_loss = float("inf")
    current_lr = start_lr
    batch_iter = iter(train_loader)

    model.train()
    for i in range(num_iter):
        # Get next batch (cycle if dataset is small)
        try:
            xb, yb = next(batch_iter)
        except StopIteration:
            batch_iter = iter(train_loader)
            xb, yb = next(batch_iter)

        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()

        # Exponential smoothing
        if i == 0:
            smoothed_loss = loss_val
        else:
            smoothed_loss = 0.95 * smoothed_loss + 0.05 * loss_val

        # Stop if loss diverges (> 4x best)
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        if smoothed_loss > best_loss * 4 and i > 10:
            break

        lrs.append(current_lr)
        losses.append(smoothed_loss)

        # Stream progress every 5 iters and on the last one — keeps the UI alive.
        if on_progress is not None and (i % 5 == 0 or i == num_iter - 1):
            try:
                on_progress(i + 1, num_iter, current_lr, smoothed_loss)
            except Exception:
                pass  # progress callback must never break training

        # Step LR
        current_lr *= lr_mult
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

    # Restore original weights
    model.load_state_dict(saved_state)

    # Find the LR with steepest negative gradient
    if len(losses) < 5:
        return 1e-3  # fallback

    gradients = []
    for i in range(1, len(losses)):
        grad = (losses[i] - losses[i - 1]) / (math.log10(lrs[i]) - math.log10(lrs[i - 1]))
        gradients.append(grad)

    # Find the point with the most negative gradient (steepest descent)
    min_grad_idx = int(np.argmin(gradients))
    # Use a LR slightly before the steepest point (more conservative)
    suggested_idx = max(0, min_grad_idx - 2)
    suggested_lr = lrs[suggested_idx]

    # Clamp to reasonable range
    suggested_lr = max(1e-5, min(suggested_lr, 0.1))

    return suggested_lr


# ---------------------------------------------------------------------------
# Class Weight Computation
# ---------------------------------------------------------------------------

def compute_class_weights(
    y: np.ndarray,
    n_classes: int,
    strategy: str = "balanced",
) -> torch.Tensor:
    """Compute class weights for CrossEntropyLoss to handle imbalanced data.

    Strategies:
        - ``"balanced"``: weight = n_samples / (n_classes * class_count).
          Inverse-frequency weighting (same as sklearn).
        - ``"sqrt"``: weight = sqrt(balanced_weight). Less aggressive
          rebalancing, often better for very skewed distributions.
        - ``"none"``: uniform weights (no rebalancing).

    Returns a 1-D float tensor of shape ``(n_classes,)``.
    """
    if strategy == "none":
        return torch.ones(n_classes, dtype=torch.float32)

    n_samples = len(y)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    # Avoid division by zero for classes not present
    counts = np.maximum(counts, 1.0)

    weights = n_samples / (n_classes * counts)

    if strategy == "sqrt":
        weights = np.sqrt(weights)

    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Architecture Selection
# ---------------------------------------------------------------------------

def select_architecture(
    n_classes: int,
    n_channels: int,
    signal_length: int,
    n_samples: int,
) -> dict:
    """Rule-based architecture config selector.

    Returns a dict consumed by Signal1DCNN(arch_config=...) with keys:
        - ``kernel_sizes``: list of 3 ints (one per conv block)
        - ``channels``: list of 3 ints (output channels per conv block)
        - ``dropout1``: float (after global avg pool)
        - ``dropout2``: float (between FC layers)
        - ``fc_hidden``: int (hidden FC layer size)
    """
    # Default (medium signals)
    kernel_sizes = [5, 5, 3]
    channels = [32, 64, 128]
    dropout1 = 0.3
    dropout2 = 0.2
    fc_hidden = 64

    # Short signals: smaller kernels
    if signal_length < 64:
        kernel_sizes = [3, 3, 3]
    # Long signals: larger first kernel, wider network
    elif signal_length >= 256:
        kernel_sizes = [7, 5, 3]
        channels = [64, 128, 256]
        fc_hidden = 128

    # Many classes: wider network
    if n_classes > 10:
        channels = [max(c, v) for c, v in zip(channels, [64, 128, 256])]
        fc_hidden = max(fc_hidden, 128)

    # Small dataset: increase regularization
    if n_samples < 200:
        dropout1 = 0.5
        dropout2 = 0.3

    # Very large dataset: reduce regularization
    if n_samples > 5000:
        dropout1 = 0.2
        dropout2 = 0.1

    return {
        "kernel_sizes": kernel_sizes,
        "channels": channels,
        "dropout1": dropout1,
        "dropout2": dropout2,
        "fc_hidden": fc_hidden,
    }


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Track validation loss and stop training when it stops improving.

    Call ``.step(val_loss, model)`` each epoch.  When patience is
    exceeded, ``.should_stop`` becomes ``True``.  The best model weights
    are always available via ``.best_state``.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Update state.  Returns ``True`` if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Data Quality Assessment
# ---------------------------------------------------------------------------

class DataQualityAssessor:
    """Assess the quality of a labeled training dataset before training.

    Checks for common issues that would hurt model performance:
    class imbalance, insufficient samples, flat/constant channels, and
    NaN/Inf values.  Returns structured issues so the frontend can display
    plain-language warnings before the user starts training.
    """

    def assess(self, X: np.ndarray, y: np.ndarray, class_names: list) -> dict:
        """Run all quality checks.

        Parameters
        ----------
        X : ndarray of shape (N, features)
        y : ndarray of shape (N,) with integer class indices
        class_names : list of class label strings

        Returns
        -------
        dict with keys: issues, quality_score (0-100), ready_to_train,
                        sample_count, class_distribution
        """
        issues = []
        n_samples = len(X)
        n_classes = len(class_names)

        # ── NaN / Inf check ──
        if np.any(~np.isfinite(X)):
            issues.append({
                "severity": "error",
                "code": "nan_inf",
                "message_en": "Dataset contains NaN or Inf values. Clean the data before training.",
                "message_zh": "数据集包含 NaN 或 Inf 值，请在训练前清理数据。",
            })

        # ── Sample count ──
        if n_samples < 50:
            issues.append({
                "severity": "error",
                "code": "too_few_samples",
                "message_en": f"Only {n_samples} samples — need at least 50 for meaningful training.",
                "message_zh": f"仅有 {n_samples} 个样本，至少需要 50 个才能有效训练。",
            })
        elif n_samples < 300:
            issues.append({
                "severity": "warning",
                "code": "small_dataset",
                "message_en": f"Only {n_samples} samples. Results may not generalize well. Consider collecting more data.",
                "message_zh": f"仅有 {n_samples} 个样本，结果可能泛化性不足，建议收集更多数据。",
            })

        # ── Class imbalance ──
        counts = np.bincount(y, minlength=n_classes)
        valid_counts = counts[counts > 0]
        if len(valid_counts) > 1:
            ratio = int(valid_counts.max()) / max(int(valid_counts.min()), 1)
            if ratio > 10:
                issues.append({
                    "severity": "warning",
                    "code": "severe_imbalance",
                    "message_en": f"Severe class imbalance ({ratio:.0f}× ratio). Class weighting is strongly recommended.",
                    "message_zh": f"严重类别不平衡（比例 {ratio:.0f}×），强烈建议启用类别权重。",
                })
            elif ratio > 5:
                issues.append({
                    "severity": "warning",
                    "code": "moderate_imbalance",
                    "message_en": f"Moderate class imbalance ({ratio:.0f}× ratio). Class weighting is recommended.",
                    "message_zh": f"存在类别不平衡（比例 {ratio:.0f}×），建议启用类别权重。",
                })

        # ── Missing classes (some classes have 0 samples) ──
        empty_classes = [class_names[i] for i, c in enumerate(counts) if c == 0]
        if empty_classes:
            issues.append({
                "severity": "error",
                "code": "empty_classes",
                "message_en": f"Classes with 0 samples: {empty_classes}. Remove them or add data.",
                "message_zh": f"以下类别样本数为 0：{empty_classes}，请删除或补充数据。",
            })

        # ── Flat / constant channels (std ≈ 0 across samples) ──
        if X.shape[0] > 1:
            col_stds = X.std(axis=0)
            flat_frac = float((col_stds < 1e-6).mean())
            if flat_frac > 0.5:
                issues.append({
                    "severity": "warning",
                    "code": "flat_signal",
                    "message_en": f"{flat_frac*100:.0f}% of signal columns are nearly constant — the signal may not contain discriminative information.",
                    "message_zh": f"{flat_frac*100:.0f}% 的信号列接近常数，信号可能不含有效判别信息。",
                })
            elif flat_frac > 0.1:
                issues.append({
                    "severity": "info",
                    "code": "partial_flat",
                    "message_en": f"{flat_frac*100:.0f}% of signal columns have very low variance.",
                    "message_zh": f"{flat_frac*100:.0f}% 的信号列方差极低。",
                })

        # ── Quality score (0-100) ──
        deductions = sum(
            30 if i["severity"] == "error" else
            15 if i["severity"] == "warning" else 5
            for i in issues
        )
        quality_score = max(0, 100 - deductions)

        # ── Per-class distribution ──
        class_dist = {class_names[i]: int(counts[i]) for i in range(n_classes)}

        return {
            "issues": issues,
            "quality_score": quality_score,
            "ready_to_train": all(i["severity"] != "error" for i in issues),
            "sample_count": n_samples,
            "n_classes": n_classes,
            "class_distribution": class_dist,
        }
