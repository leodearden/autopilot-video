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

    @property
    def vram_used(self) -> int:
        """Total VRAM bytes consumed by loaded models."""
        return sum(
            self._registry[n].vram_bytes for n in self._loaded
        )

    @property
    def vram_free(self) -> int:
        """VRAM bytes still available within the budget."""
        return self._total_vram - self.vram_used

    def _evict_for(self, needed: int) -> None:
        """Evict LRU models until *needed* bytes are free.

        Must be called while holding ``self._lock``.
        """
        while self.vram_free < needed:
            if not self._lru:
                raise SchedulerError(
                    f"Cannot free {needed} bytes: only "
                    f"{self.vram_free} available after evicting all models"
                )
            victim = self._lru.pop(0)
            spec = self._registry[victim]
            obj = self._loaded.pop(victim)
            logger.info(
                "Evicting model '%s' (%d bytes) to free VRAM",
                victim,
                spec.vram_bytes,
            )
            try:
                spec.unload_fn(obj)
            except Exception:
                logger.exception("Error unloading '%s' during eviction", victim)

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
                spec = self._registry[name]
                self._evict_for(spec.vram_bytes)
                model_obj = spec.load_fn()
                self._loaded[name] = model_obj
                self._lru.append(name)
                logger.info(
                    "Loaded model '%s' (%d bytes VRAM)", name, spec.vram_bytes
                )
                if spec.warmup_fn is not None:
                    logger.debug("Warming up model '%s'", name)
                    spec.warmup_fn(model_obj)
        yield model_obj

    def force_unload_all(self) -> None:
        """Unload every loaded model and reset LRU tracking."""
        with self._lock:
            count = len(self._loaded)
            for name in list(self._loaded):
                spec = self._registry[name]
                obj = self._loaded.pop(name)
                try:
                    spec.unload_fn(obj)
                except Exception:
                    logger.exception(
                        "Error unloading '%s' during force_unload_all", name
                    )
            self._lru.clear()
            logger.info("Force-unloaded %d model(s)", count)
