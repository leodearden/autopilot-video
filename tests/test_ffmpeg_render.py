"""Tests for autopilot.render.ffmpeg_render — FFmpeg fast-path rendering."""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify public API surface: imports, signatures, and __all__ exports."""

    def test_render_error_importable_and_is_exception(self) -> None:
        """RenderError should be importable and be an Exception subclass."""
        from autopilot.render.ffmpeg_render import RenderError

        assert issubclass(RenderError, Exception)

    def test_render_simple_importable(self) -> None:
        """render_simple should be importable and callable."""
        from autopilot.render.ffmpeg_render import render_simple

        assert callable(render_simple)

    def test_render_simple_signature(self) -> None:
        """render_simple should accept (edl_entry, crop_path, output_path, config)."""
        from autopilot.render.ffmpeg_render import render_simple

        sig = inspect.signature(render_simple)
        param_names = list(sig.parameters.keys())
        assert param_names == ["edl_entry", "crop_path", "output_path", "config"]

    def test_all_exports(self) -> None:
        """__all__ should export RenderError and render_simple."""
        from autopilot.render import ffmpeg_render

        assert hasattr(ffmpeg_render, "__all__")
        assert set(ffmpeg_render.__all__) == {"RenderError", "render_simple"}
