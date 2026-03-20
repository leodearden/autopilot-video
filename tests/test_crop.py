"""Tests for auto-crop viewport computation (autopilot.render.crop)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest


class TestPublicAPI:
    """Verify public API surface: imports, signatures, and __all__ exports."""

    def test_crop_error_importable_and_is_exception(self) -> None:
        """CropError should be importable and be an Exception subclass."""
        from autopilot.render.crop import CropError

        assert issubclass(CropError, Exception)

    def test_compute_crop_path_importable(self) -> None:
        """compute_crop_path should be importable."""
        from autopilot.render.crop import compute_crop_path

        assert callable(compute_crop_path)

    def test_compute_crop_path_signature(self) -> None:
        """compute_crop_path should accept (media_id, target_aspect, db, config, edl_entry)."""
        from autopilot.render.crop import compute_crop_path

        sig = inspect.signature(compute_crop_path)
        param_names = list(sig.parameters.keys())
        assert param_names == [
            "media_id",
            "target_aspect",
            "db",
            "config",
            "edl_entry",
        ]

    def test_compute_crop_path_return_annotation(self) -> None:
        """compute_crop_path should be annotated to return np.ndarray."""
        from autopilot.render.crop import compute_crop_path

        sig = inspect.signature(compute_crop_path)
        # With PEP 563 (from __future__ import annotations), annotations are strings
        assert sig.return_annotation in (np.ndarray, "np.ndarray")

    def test_all_exports(self) -> None:
        """__all__ should export CropError and compute_crop_path."""
        from autopilot.render import crop

        assert hasattr(crop, "__all__")
        assert set(crop.__all__) == {"CropError", "compute_crop_path"}


class TestComputeCropDimensions:
    """Tests for _compute_crop_dimensions helper."""

    def test_16_9_from_square_source(self) -> None:
        """4096x4096 source with '16:9' -> maximize width, compute height."""
        from autopilot.render.crop import _compute_crop_dimensions

        crop_w, crop_h = _compute_crop_dimensions(4096, 4096, "16:9")
        assert crop_w == 4096
        assert crop_h == 4096 * 9 // 16  # 2304

    def test_9_16_from_square_source(self) -> None:
        """4096x4096 source with '9:16' -> maximize height, compute width."""
        from autopilot.render.crop import _compute_crop_dimensions

        crop_w, crop_h = _compute_crop_dimensions(4096, 4096, "9:16")
        assert crop_h == 4096
        assert crop_w == 4096 * 9 // 16  # 2304

    def test_16_9_from_landscape_source(self) -> None:
        """1920x1080 source with '16:9' -> full frame (already 16:9)."""
        from autopilot.render.crop import _compute_crop_dimensions

        crop_w, crop_h = _compute_crop_dimensions(1920, 1080, "16:9")
        assert crop_w == 1920
        assert crop_h == 1080

    def test_unknown_aspect_raises_crop_error(self) -> None:
        """Unknown aspect string raises CropError."""
        from autopilot.render.crop import CropError, _compute_crop_dimensions

        with pytest.raises(CropError, match="aspect"):
            _compute_crop_dimensions(4096, 4096, "potato")
