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

    def __init__(
        self,
        total_vram: int,
        device: int = 0,
        vram_query: Callable[[], tuple[int, int]] | None = None,
    ) -> None:
        self._device = device
        self._total_vram = total_vram
        self._vram_query = vram_query
        self._registry: dict[str, ModelSpec] = {}
        self._loaded: dict[str, Any] = {}
        self._lru: list[str] = []  # oldest first
        self._lock = threading.Lock()

    @property
    def device(self) -> int:
        """GPU device index."""
        return self._device

    @property
    def loaded_models(self) -> set[str]:
        """Names of currently loaded models."""
        return set(self._loaded)

    def register(self, name: str, spec: ModelSpec) -> None:
        """Register a model spec under *name*."""
        with self._lock:
            if name in self._registry:
                raise SchedulerError(f"Model '{name}' already registered")
            self._registry[name] = spec

    def unregister(self, name: str) -> None:
        """Remove a model from the registry (unloading first if needed)."""
        with self._lock:
            if name not in self._registry:
                raise SchedulerError(f"Model '{name}' not registered")
            if name in self._loaded:
                spec = self._registry[name]
                try:
                    spec.unload_fn(self._loaded[name])
                except Exception:
                    logger.exception("Error unloading '%s' during unregister", name)
                del self._loaded[name]
                self._lru.remove(name)
            del self._registry[name]

    @contextmanager
    def model(self, name: str):  # type: ignore[override]
        """Context manager that ensures *name* is loaded and yields the model."""
        with self._lock:
            if name not in self._registry:
                raise SchedulerError(f"Model '{name}' not registered")
            model_obj = self._loaded.get(name)
            if model_obj is not None:
                # Already loaded — update LRU
                self._lru.remove(name)
                self._lru.append(name)
            else:
                model_obj = self._registry[name].load_fn()
                self._loaded[name] = model_obj
                self._lru.append(name)
        yield model_obj
