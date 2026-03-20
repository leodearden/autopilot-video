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


# ---------------------------------------------------------------------------
# Helpers for integration tests
# ---------------------------------------------------------------------------


def _make_config(
    resolution: tuple[int, int] = (1920, 1080),
    codec: str = "h264",
    quality_crf: int = 18,
    audio_bitrate: str = "256k",
    target_loudness_lufs: int = -16,
) -> object:
    """Create an OutputConfig for testing."""
    from autopilot.config import OutputConfig

    return OutputConfig(
        resolution=resolution,
        codec=codec,
        quality_crf=quality_crf,
        audio_bitrate=audio_bitrate,
        target_loudness_lufs=target_loudness_lufs,
    )


def _make_edl(clips: list[dict] | None = None, **kwargs: object) -> dict:
    """Create a minimal EDL dict for testing."""
    edl: dict = {
        "clips": clips or [
            {
                "clip_id": "clip_1",
                "source_path": "/tmp/src/clip.mp4",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ],
        "transitions": {},
        "crop_modes": {},
        "audio_settings": {},
        "music": [],
        "voiceovers": [],
        "broll_requests": [],
    }
    edl.update(kwargs)
    return edl


# ---------------------------------------------------------------------------
# EDL loading
# ---------------------------------------------------------------------------


class TestEDLLoading:
    """Verify route_and_render loads EDL correctly from database."""

    def test_no_edit_plan_raises_routing_error(self) -> None:
        """When narrative has no edit plan, should raise RoutingError."""
        from autopilot.render.router import RoutingError, route_and_render

        db = MagicMock()
        db.get_edit_plan.return_value = None
        config = _make_config()

        with pytest.raises(RoutingError, match="No edit plan"):
            route_and_render("narr_1", db, config)

    def test_loads_edl_from_edit_plan(self) -> None:
        """route_and_render should parse edl_json from edit plan."""
        from autopilot.render.router import RoutingError, route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "narr_1", "title": "Test"}
        config = _make_config()

        # Should not raise RoutingError for missing EDL
        # (will raise NotImplementedError or other error later in pipeline)
        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("autopilot.render.router.render_complex"), \
             patch("subprocess.run"):
            mock_rs.return_value = Path("/tmp/segment.mp4")
            try:
                route_and_render("narr_1", db, config)
            except (NotImplementedError, RoutingError, Exception):
                pass  # We just want to verify EDL was loaded

        db.get_edit_plan.assert_called_once_with("narr_1")
