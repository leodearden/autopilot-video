"""Tests for autopilot.render.moviepy_render — MoviePy complex-path rendering."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify public API surface: imports, signatures, and __all__ exports."""

    def test_complex_render_error_importable_and_is_exception(self) -> None:
        """ComplexRenderError should be importable and be an Exception subclass."""
        from autopilot.render.moviepy_render import ComplexRenderError

        assert issubclass(ComplexRenderError, Exception)

    def test_render_complex_importable(self) -> None:
        """render_complex should be importable and callable."""
        from autopilot.render.moviepy_render import render_complex

        assert callable(render_complex)

    def test_render_complex_signature(self) -> None:
        """render_complex should accept (edl_entry, crop_path, output_path, config)."""
        from autopilot.render.moviepy_render import render_complex

        sig = inspect.signature(render_complex)
        param_names = list(sig.parameters.keys())
        assert param_names == ["edl_entry", "crop_path", "output_path", "config"]

    def test_all_exports(self) -> None:
        """__all__ should export ComplexRenderError and render_complex."""
        from autopilot.render import moviepy_render

        assert hasattr(moviepy_render, "__all__")
        assert set(moviepy_render.__all__) == {"ComplexRenderError", "render_complex"}
