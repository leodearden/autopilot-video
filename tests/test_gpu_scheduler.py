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
