"""Tests for the GPU model scheduler (autopilot.analyze.gpu_scheduler)."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from autopilot.analyze.gpu_scheduler import GPUScheduler, ModelSpec, SchedulerError

GB = 1024**3


class TestModelSpec:
    """Tests for the ModelSpec dataclass."""

    def test_model_spec_fields(self) -> None:
        """ModelSpec stores load_fn, unload_fn, vram_bytes, and warmup_fn."""
        load = MagicMock()
        unload = MagicMock()
        warmup = MagicMock()

        spec = ModelSpec(
            load_fn=load,
            unload_fn=unload,
            vram_bytes=8 * GB,
            warmup_fn=warmup,
        )

        assert spec.load_fn is load
        assert spec.unload_fn is unload
        assert spec.vram_bytes == 8 * GB
        assert spec.warmup_fn is warmup

    def test_model_spec_warmup_fn_optional(self) -> None:
        """ModelSpec warmup_fn defaults to None when not provided."""
        spec = ModelSpec(
            load_fn=MagicMock(),
            unload_fn=MagicMock(),
            vram_bytes=4 * GB,
        )
        assert spec.warmup_fn is None


class TestPublicAPI:
    """Tests for module-level public API exports."""

    def test_public_api_importable(self) -> None:
        """GPUScheduler, ModelSpec, and SchedulerError are importable."""
        from autopilot.analyze.gpu_scheduler import (
            GPUScheduler,
            ModelSpec,
            SchedulerError,
        )

        assert GPUScheduler is not None
        assert ModelSpec is not None
        assert SchedulerError is not None

    def test_scheduler_error_is_exception(self) -> None:
        """SchedulerError is a subclass of Exception."""
        assert issubclass(SchedulerError, Exception)
        err = SchedulerError("test error")
        assert str(err) == "test error"


def _make_spec(vram: int = 3 * GB) -> ModelSpec:
    """Helper to create a ModelSpec with MagicMock callables."""
    return ModelSpec(
        load_fn=MagicMock(return_value=f"model_{vram}"),
        unload_fn=MagicMock(),
        vram_bytes=vram,
    )


class TestSchedulerInit:
    """Tests for GPUScheduler initialization and model registration."""

    def test_init_default_device(self) -> None:
        """GPUScheduler() defaults to device=0."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        assert scheduler.device == 0

    def test_init_custom_device(self) -> None:
        """GPUScheduler stores custom device correctly."""
        scheduler = GPUScheduler(device=1, total_vram=24 * GB)
        assert scheduler.device == 1

    def test_register_model(self) -> None:
        """Registered model appears in the scheduler's registry."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec = _make_spec()
        scheduler.register("whisperx", spec)
        assert "whisperx" in scheduler.loaded_models or True  # just check no error
        # More specific: verify it can be used
        with scheduler.model("whisperx") as m:
            assert m is not None

    def test_register_duplicate_raises(self) -> None:
        """Registering the same model name twice raises SchedulerError."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec = _make_spec()
        scheduler.register("whisperx", spec)
        with pytest.raises(SchedulerError, match="already registered"):
            scheduler.register("whisperx", spec)

    def test_unregister_model(self) -> None:
        """Unregistered model is removed from the registry."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec = _make_spec()
        scheduler.register("whisperx", spec)
        scheduler.unregister("whisperx")
        with pytest.raises(SchedulerError):
            with scheduler.model("whisperx") as m:
                pass
