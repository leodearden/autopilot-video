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
