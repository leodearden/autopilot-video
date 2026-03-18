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


class TestContextManager:
    """Tests for model loading via the context manager."""

    def test_context_manager_loads_model(self) -> None:
        """Context manager calls load_fn and yields the loaded model."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        sentinel = object()
        spec = ModelSpec(
            load_fn=MagicMock(return_value=sentinel),
            unload_fn=MagicMock(),
            vram_bytes=3 * GB,
        )
        scheduler.register("test", spec)
        with scheduler.model("test") as m:
            assert m is sentinel
        spec.load_fn.assert_called_once()

    def test_context_manager_model_stays_loaded(self) -> None:
        """After exiting the context, the model remains loaded (LRU cache)."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec = _make_spec()
        scheduler.register("test", spec)
        with scheduler.model("test"):
            pass
        assert "test" in scheduler.loaded_models

    def test_context_manager_reuses_loaded_model(self) -> None:
        """Re-entering context for a loaded model does not call load_fn again."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec = _make_spec()
        scheduler.register("test", spec)
        with scheduler.model("test"):
            pass
        with scheduler.model("test"):
            pass
        spec.load_fn.assert_called_once()

    def test_context_manager_unknown_model_raises(self) -> None:
        """Using scheduler.model() with an unregistered name raises SchedulerError."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        with pytest.raises(SchedulerError, match="not registered"):
            with scheduler.model("nonexistent"):
                pass


class TestLRUEviction:
    """Tests for LRU eviction when VRAM is insufficient."""

    def test_evicts_lru_when_vram_insufficient(self) -> None:
        """Loading a model that won't fit evicts the least-recently-used model."""
        scheduler = GPUScheduler(total_vram=10 * GB)
        spec_a = _make_spec(vram=3 * GB)
        spec_b = _make_spec(vram=8 * GB)
        scheduler.register("a", spec_a)
        scheduler.register("b", spec_b)

        with scheduler.model("a"):
            pass  # a loaded (3GB used, 7GB free)

        with scheduler.model("b"):
            pass  # b needs 8GB, only 7GB free → evict a

        spec_a.unload_fn.assert_called_once()
        assert "a" not in scheduler.loaded_models
        assert "b" in scheduler.loaded_models

    def test_evicts_multiple_models_if_needed(self) -> None:
        """Evicts multiple LRU models until enough VRAM is free."""
        scheduler = GPUScheduler(total_vram=10 * GB)
        specs = {}
        for name in ("s1", "s2", "s3"):
            specs[name] = _make_spec(vram=3 * GB)
            scheduler.register(name, specs[name])
        spec_large = _make_spec(vram=9 * GB)
        scheduler.register("large", spec_large)

        # Load all 3 small models (9GB used)
        for name in ("s1", "s2", "s3"):
            with scheduler.model(name):
                pass

        # Load large (needs 9GB, only 1GB free → must evict all 3 small)
        with scheduler.model("large"):
            pass

        for name in ("s1", "s2", "s3"):
            specs[name].unload_fn.assert_called_once()
        assert scheduler.loaded_models == {"large"}

    def test_no_eviction_when_vram_sufficient(self) -> None:
        """No eviction when VRAM budget is sufficient for all models."""
        scheduler = GPUScheduler(total_vram=10 * GB)
        spec_a = _make_spec(vram=3 * GB)
        spec_b = _make_spec(vram=3 * GB)
        scheduler.register("a", spec_a)
        scheduler.register("b", spec_b)

        with scheduler.model("a"):
            pass
        with scheduler.model("b"):
            pass

        spec_a.unload_fn.assert_not_called()
        spec_b.unload_fn.assert_not_called()
        assert scheduler.loaded_models == {"a", "b"}

    def test_eviction_order_is_lru(self) -> None:
        """Eviction starts with the least-recently-used model."""
        scheduler = GPUScheduler(total_vram=10 * GB)
        spec_a = _make_spec(vram=3 * GB)
        spec_b = _make_spec(vram=3 * GB)
        spec_c = _make_spec(vram=3 * GB)
        spec_d = _make_spec(vram=4 * GB)
        scheduler.register("a", spec_a)
        scheduler.register("b", spec_b)
        scheduler.register("c", spec_c)
        scheduler.register("d", spec_d)

        # Load A, B, C in order (9GB used, LRU order: A, B, C)
        for name in ("a", "b", "c"):
            with scheduler.model(name):
                pass

        # Load D (needs 4GB, only 1GB free → evict A first, then B if needed)
        # After evicting A: 6GB used, 4GB free → D fits, stop evicting
        with scheduler.model("d"):
            pass

        spec_a.unload_fn.assert_called_once()
        spec_b.unload_fn.assert_not_called()
        spec_c.unload_fn.assert_not_called()
        assert "a" not in scheduler.loaded_models
        assert scheduler.loaded_models == {"b", "c", "d"}


class TestLRUReaccess:
    """Tests for LRU ordering updates on re-access."""

    def test_lru_order_updated_on_reaccess(self) -> None:
        """Re-accessing a model moves it to most-recently-used, changing eviction order."""
        scheduler = GPUScheduler(total_vram=10 * GB)
        spec_a = _make_spec(vram=3 * GB)
        spec_b = _make_spec(vram=3 * GB)
        spec_c = _make_spec(vram=3 * GB)
        spec_d = _make_spec(vram=4 * GB)
        scheduler.register("a", spec_a)
        scheduler.register("b", spec_b)
        scheduler.register("c", spec_c)
        scheduler.register("d", spec_d)

        # Load A, B, C (9GB, LRU order: A, B, C)
        for name in ("a", "b", "c"):
            with scheduler.model(name):
                pass

        # Re-access A → LRU order: B, C, A
        with scheduler.model("a"):
            pass

        # Load D (needs 4GB, 1GB free) → should evict B (now LRU), not A
        with scheduler.model("d"):
            pass

        spec_b.unload_fn.assert_called_once()
        spec_a.unload_fn.assert_not_called()
        assert scheduler.loaded_models == {"a", "c", "d"}

    def test_model_too_large_for_total_vram_raises(self) -> None:
        """A model larger than total_vram raises SchedulerError immediately."""
        scheduler = GPUScheduler(total_vram=10 * GB)
        spec = _make_spec(vram=12 * GB)
        scheduler.register("huge", spec)
        with pytest.raises(SchedulerError, match="Cannot free"):
            with scheduler.model("huge"):
                pass


class TestForceUnload:
    """Tests for force_unload_all()."""

    def test_force_unload_all_unloads_everything(self) -> None:
        """force_unload_all calls unload_fn for every loaded model."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        specs = {}
        for name in ("a", "b", "c"):
            specs[name] = _make_spec(vram=3 * GB)
            scheduler.register(name, specs[name])
            with scheduler.model(name):
                pass

        scheduler.force_unload_all()

        for name in ("a", "b", "c"):
            specs[name].unload_fn.assert_called_once()
        assert scheduler.loaded_models == set()

    def test_force_unload_all_when_empty(self) -> None:
        """force_unload_all on an empty scheduler raises no error."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        scheduler.force_unload_all()  # no error

    def test_force_unload_all_clears_lru(self) -> None:
        """After force_unload_all, LRU tracking is reset."""
        scheduler = GPUScheduler(total_vram=10 * GB)
        spec_a = _make_spec(vram=3 * GB)
        spec_b = _make_spec(vram=8 * GB)
        scheduler.register("a", spec_a)
        scheduler.register("b", spec_b)

        with scheduler.model("a"):
            pass
        scheduler.force_unload_all()

        # Now loading b should NOT evict anything (LRU is empty)
        with scheduler.model("b"):
            pass
        spec_a.unload_fn.assert_called_once()  # only from force_unload
        assert scheduler.loaded_models == {"b"}

    def test_can_reload_after_force_unload(self) -> None:
        """Models can be reloaded after force_unload_all."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec = _make_spec(vram=3 * GB)
        scheduler.register("test", spec)

        with scheduler.model("test"):
            pass
        scheduler.force_unload_all()
        with scheduler.model("test"):
            pass

        assert spec.load_fn.call_count == 2


class TestWarmup:
    """Tests for warmup function support."""

    def test_warmup_called_after_load(self) -> None:
        """warmup_fn is called with the loaded model after load_fn."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        sentinel = object()
        warmup = MagicMock()
        spec = ModelSpec(
            load_fn=MagicMock(return_value=sentinel),
            unload_fn=MagicMock(),
            vram_bytes=3 * GB,
            warmup_fn=warmup,
        )
        scheduler.register("test", spec)
        with scheduler.model("test"):
            pass
        warmup.assert_called_once_with(sentinel)

    def test_warmup_not_called_on_cached_access(self) -> None:
        """warmup_fn is NOT called when model is already loaded."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        warmup = MagicMock()
        spec = ModelSpec(
            load_fn=MagicMock(return_value="m"),
            unload_fn=MagicMock(),
            vram_bytes=3 * GB,
            warmup_fn=warmup,
        )
        scheduler.register("test", spec)
        with scheduler.model("test"):
            pass
        with scheduler.model("test"):
            pass
        warmup.assert_called_once()  # only on first load

    def test_warmup_none_skipped(self) -> None:
        """When warmup_fn is None, no warmup is attempted."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec = ModelSpec(
            load_fn=MagicMock(return_value="m"),
            unload_fn=MagicMock(),
            vram_bytes=3 * GB,
            warmup_fn=None,
        )
        scheduler.register("test", spec)
        with scheduler.model("test") as m:
            assert m == "m"  # loads without error


class TestThreadSafety:
    """Tests for thread safety under concurrent access."""

    def test_concurrent_model_access(self) -> None:
        """Multiple threads accessing the same model only call load_fn once."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        sentinel = object()
        spec = ModelSpec(
            load_fn=MagicMock(return_value=sentinel),
            unload_fn=MagicMock(),
            vram_bytes=3 * GB,
        )
        scheduler.register("model_a", spec)

        results: list[object] = []
        errors: list[Exception] = []

        def access_model() -> None:
            try:
                for _ in range(5):
                    with scheduler.model("model_a") as m:
                        results.append(m)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_model) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(r is sentinel for r in results)
        spec.load_fn.assert_called_once()

    def test_concurrent_different_models(self) -> None:
        """Two threads loading different models both succeed."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec_a = _make_spec(vram=3 * GB)
        spec_b = _make_spec(vram=3 * GB)
        scheduler.register("a", spec_a)
        scheduler.register("b", spec_b)

        errors: list[Exception] = []

        def load_model(name: str) -> None:
            try:
                with scheduler.model(name):
                    pass
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=load_model, args=("a",))
        t2 = threading.Thread(target=load_model, args=("b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        assert scheduler.loaded_models == {"a", "b"}


class TestProperties:
    """Tests for device and VRAM introspection properties."""

    def test_device_property(self) -> None:
        """GPUScheduler.device returns the configured device index."""
        scheduler = GPUScheduler(device=2, total_vram=24 * GB)
        assert scheduler.device == 2

    def test_loaded_models_returns_names(self) -> None:
        """loaded_models returns set of currently loaded model names."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec_a = _make_spec(vram=3 * GB)
        spec_b = _make_spec(vram=3 * GB)
        scheduler.register("a", spec_a)
        scheduler.register("b", spec_b)
        with scheduler.model("a"):
            pass
        with scheduler.model("b"):
            pass
        assert scheduler.loaded_models == {"a", "b"}

    def test_loaded_models_empty_initially(self) -> None:
        """loaded_models is empty when no models are loaded."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        assert scheduler.loaded_models == set()

    def test_vram_used_property(self) -> None:
        """vram_used returns the sum of loaded models' vram_bytes."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec_a = _make_spec(vram=3 * GB)
        spec_b = _make_spec(vram=8 * GB)
        scheduler.register("a", spec_a)
        scheduler.register("b", spec_b)
        with scheduler.model("a"):
            pass
        with scheduler.model("b"):
            pass
        assert scheduler.vram_used == 11 * GB

    def test_vram_free_property(self) -> None:
        """vram_free returns total_vram minus vram_used."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec = _make_spec(vram=8 * GB)
        scheduler.register("m", spec)
        with scheduler.model("m"):
            pass
        assert scheduler.vram_free == 16 * GB


class TestLogging:
    """Tests for structured logging integration."""

    def test_load_logs_model_name_and_vram(self, caplog: pytest.LogCaptureFixture) -> None:
        """Loading a model logs its name and VRAM usage."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        spec = _make_spec(vram=3 * GB)
        scheduler.register("whisperx", spec)
        with caplog.at_level("INFO", logger="autopilot.analyze.gpu_scheduler"):
            with scheduler.model("whisperx"):
                pass
        assert any("whisperx" in r.message for r in caplog.records)
        assert any("3" in r.message for r in caplog.records)

    def test_eviction_logs_evicted_model(self, caplog: pytest.LogCaptureFixture) -> None:
        """Evicting a model logs which model was evicted."""
        scheduler = GPUScheduler(total_vram=10 * GB)
        spec_a = _make_spec(vram=3 * GB)
        spec_b = _make_spec(vram=8 * GB)
        scheduler.register("a", spec_a)
        scheduler.register("b", spec_b)
        with scheduler.model("a"):
            pass
        with caplog.at_level("INFO", logger="autopilot.analyze.gpu_scheduler"):
            with scheduler.model("b"):
                pass
        assert any("a" in r.message and "evict" in r.message.lower() for r in caplog.records)

    def test_force_unload_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """force_unload_all logs the number of unloaded models."""
        scheduler = GPUScheduler(total_vram=24 * GB)
        for name in ("a", "b"):
            scheduler.register(name, _make_spec(vram=3 * GB))
            with scheduler.model(name):
                pass
        with caplog.at_level("INFO", logger="autopilot.analyze.gpu_scheduler"):
            scheduler.force_unload_all()
        assert any("2" in r.message for r in caplog.records)
