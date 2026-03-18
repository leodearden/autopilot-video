"""VRAM-aware GPU model scheduler with context-manager API.

Manages a registry of ML models and their VRAM requirements,
loading/unloading them on demand with LRU eviction when GPU
memory is constrained.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = ["GPUScheduler", "ModelSpec", "SchedulerError"]

logger = logging.getLogger(__name__)


class SchedulerError(Exception):
    """Raised for all GPU scheduler failures."""


@dataclass
class ModelSpec:
    """Specification for a GPU model managed by the scheduler.

    Attributes:
        load_fn: Callable that loads the model and returns the model object.
        unload_fn: Callable that takes a loaded model object and unloads it.
        vram_bytes: Estimated VRAM usage in bytes.
        warmup_fn: Optional callable to warm up the model after loading.
    """

    load_fn: Callable[[], Any]
    unload_fn: Callable[[Any], None]
    vram_bytes: int
    warmup_fn: Callable[[Any], None] | None = None


class GPUScheduler:
    """VRAM-aware GPU model scheduler with LRU eviction."""

    pass
