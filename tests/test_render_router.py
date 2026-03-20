"""Tests for autopilot.render.router — render routing and assembly."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify public API surface: imports, signatures, and __all__ exports."""

    def test_routing_error_importable_and_is_exception(self) -> None:
        """RoutingError should be importable and be an Exception subclass."""
        from autopilot.render.router import RoutingError

        assert issubclass(RoutingError, Exception)

    def test_route_and_render_importable(self) -> None:
        """route_and_render should be importable and callable."""
        from autopilot.render.router import route_and_render

        assert callable(route_and_render)

    def test_route_and_render_signature(self) -> None:
        """route_and_render should accept (narrative_id, db, config)."""
        from autopilot.render.router import route_and_render

        sig = inspect.signature(route_and_render)
        param_names = list(sig.parameters.keys())
        assert param_names == ["narrative_id", "db", "config"]

    def test_all_exports(self) -> None:
        """__all__ should export RoutingError and route_and_render."""
        from autopilot.render import router

        assert hasattr(router, "__all__")
        assert set(router.__all__) == {"RoutingError", "route_and_render"}


# ---------------------------------------------------------------------------
# _classify_clip helper
# ---------------------------------------------------------------------------


class TestClassifyClip:
    """Verify _classify_clip returns 'fast' or 'slow' correctly."""

    def test_center_crop_returns_fast(self) -> None:
        """Clip with crop mode 'center' should classify as 'fast'."""
        from autopilot.render.router import _classify_clip

        clip = {"clip_id": "c1"}
        crop_modes = {"c1": "center"}
        assert _classify_clip(clip, crop_modes) == "fast"

    def test_manual_offset_returns_fast(self) -> None:
        """Clip with crop mode 'manual_offset' should classify as 'fast'."""
        from autopilot.render.router import _classify_clip

        clip = {"clip_id": "c1"}
        crop_modes = {"c1": "manual_offset"}
        assert _classify_clip(clip, crop_modes) == "fast"

    def test_stabilize_only_returns_fast(self) -> None:
        """Clip with crop mode 'stabilize_only' should classify as 'fast'."""
        from autopilot.render.router import _classify_clip

        clip = {"clip_id": "c1"}
        crop_modes = {"c1": "stabilize_only"}
        assert _classify_clip(clip, crop_modes) == "fast"

    def test_auto_subject_returns_slow(self) -> None:
        """Clip with crop mode 'auto_subject' should classify as 'slow'."""
        from autopilot.render.router import _classify_clip

        clip = {"clip_id": "c1"}
        crop_modes = {"c1": "auto_subject"}
        assert _classify_clip(clip, crop_modes) == "slow"

    def test_pip_overlay_returns_slow(self) -> None:
        """Clip with 'pip' overlay should classify as 'slow'."""
        from autopilot.render.router import _classify_clip

        clip = {"clip_id": "c1", "overlay": "pip"}
        crop_modes = {"c1": "center"}
        assert _classify_clip(clip, crop_modes) == "slow"

    def test_no_crop_mode_defaults_fast(self) -> None:
        """Clip with no crop_modes entry should default to 'fast'."""
        from autopilot.render.router import _classify_clip

        clip = {"clip_id": "c1"}
        crop_modes = {}  # no entry for c1
        assert _classify_clip(clip, crop_modes) == "fast"
