"""VRAM-aware GPU model scheduler with context-manager API.

Manages a registry of ML models and their VRAM requirements,
loading/unloading them on demand with LRU eviction when GPU
memory is constrained.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
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
        self._in_use: dict[str, int] = {}  # reference counts for yielded models
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
        Models with a non-zero reference count (currently yielded to a caller)
        are skipped and never evicted.
        """
        while self.vram_free < needed:
            # Find the first evictable (not in-use) model in LRU order
            victim = None
            for candidate in self._lru:
                if self._in_use.get(candidate, 0) == 0:
                    victim = candidate
                    break
            if victim is None:
                raise SchedulerError(
                    f"Cannot free {needed} bytes: only "
                    f"{self.vram_free} available after evicting all models"
                )
            self._lru.remove(victim)
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
        """Context manager that ensures *name* is loaded and yields the model.

        The model's reference count is incremented before yielding and
        decremented in the ``finally`` block, both under the lock.  This
        prevents ``_evict_for`` from evicting a model that is currently
        in use by another thread.
        """
        with self._lock:
            if name not in self._registry:
                raise SchedulerError(f"Model '{name}' not registered")
            if name in self._loaded:
                # Already loaded — update LRU
                model_obj = self._loaded[name]
                self._lru.remove(name)
                self._lru.append(name)
            else:
                spec = self._registry[name]
                self._evict_for(spec.vram_bytes)
                try:
                    model_obj = spec.load_fn()
                except Exception as exc:
                    raise SchedulerError(
                        f"Failed to load model '{name}': {exc}"
                    ) from exc
                self._loaded[name] = model_obj
                self._lru.append(name)
                logger.info(
                    "Loaded model '%s' (%d bytes VRAM)", name, spec.vram_bytes
                )
                if spec.warmup_fn is not None:
                    logger.debug("Warming up model '%s'", name)
                    try:
                        spec.warmup_fn(model_obj)
                    except Exception as exc:
                        # Clean up: unload the model since warmup failed
                        del self._loaded[name]
                        self._lru.remove(name)
                        try:
                            spec.unload_fn(model_obj)
                        except Exception:
                            logger.exception(
                                "Error unloading '%s' after warmup failure",
                                name,
                            )
                        raise SchedulerError(
                            f"Warmup failed for model '{name}': {exc}"
                        ) from exc
            # Increment reference count while still holding the lock
            self._in_use[name] = self._in_use.get(name, 0) + 1
        try:
            yield model_obj
        finally:
            with self._lock:
                self._in_use[name] -= 1
                if self._in_use[name] == 0:
                    del self._in_use[name]

    def force_unload_all(self) -> None:
        """Unload every loaded model and reset LRU tracking.

        Models that are currently in use (yielded to a caller) are skipped
        with a warning to avoid corrupting active inference.
        """
        with self._lock:
            skipped = 0
            unloaded = 0
            for name in list(self._loaded):
                if self._in_use.get(name, 0) > 0:
                    logger.warning(
                        "Skipping in-use model '%s' during force_unload_all "
                        "(ref_count=%d)",
                        name,
                        self._in_use[name],
                    )
                    skipped += 1
                    continue
                spec = self._registry[name]
                obj = self._loaded.pop(name)
                self._lru.remove(name)
                try:
                    spec.unload_fn(obj)
                except Exception:
                    logger.exception(
                        "Error unloading '%s' during force_unload_all", name
                    )
                unloaded += 1
            if skipped == 0:
                self._lru.clear()
            logger.info(
                "Force-unloaded %d model(s), skipped %d in-use",
                unloaded,
                skipped,
            )
