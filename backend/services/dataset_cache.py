"""
Shared in-memory cache for parsed training datasets.

Lifted out of the training router so that other routers (e.g. data prep)
can register datasets that the trainer can pick up by `dataset_id`.

Each entry is a dict with keys:
    - filename: str
    - summary:  dict (matches load_labeled_dataset's output shape)
    - file_bytes: bytes (the raw CSV/ZIP that the trainer re-parses)
"""

from __future__ import annotations

from typing import Any

_dataset_cache: dict[str, dict[str, Any]] = {}


def put(dataset_id: str, entry: dict[str, Any]) -> None:
    _dataset_cache[dataset_id] = entry


def get(dataset_id: str) -> dict[str, Any] | None:
    return _dataset_cache.get(dataset_id)


def has(dataset_id: str) -> bool:
    return dataset_id in _dataset_cache


def pop(dataset_id: str) -> dict[str, Any] | None:
    return _dataset_cache.pop(dataset_id, None)


def all_ids() -> list[str]:
    return list(_dataset_cache.keys())
