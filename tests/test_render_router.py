"""Tests for autopilot.render.router — render routing and assembly."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from autopilot.config import OutputConfig

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
) -> OutputConfig:
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
        """route_and_render should parse edl_json from edit plan and use clip data."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "narr_1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("autopilot.render.router.render_complex"), \
             patch("subprocess.run"):
            mock_rs.return_value = Path("/tmp/segment.mp4")
            route_and_render("narr_1", db, config)

        db.get_edit_plan.assert_called_once_with("narr_1")
        # Verify the parsed EDL clip was passed to render_simple
        mock_rs.assert_called_once()
        rendered_clip = mock_rs.call_args[0][0]
        assert rendered_clip["clip_id"] == "clip_1"


# ---------------------------------------------------------------------------
# Clip dispatching
# ---------------------------------------------------------------------------


class TestWorkDirCleanup:
    """Verify temporary work directory is cleaned up."""

    def test_work_dir_cleaned_up_on_success(self, tmp_path: Path) -> None:
        """After successful render, temporary work dir should be cleaned up."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        work = tmp_path / "render_work"
        work.mkdir()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("subprocess.run"), \
             patch("tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__ = MagicMock(return_value=str(work))
            mock_td.return_value.__exit__ = MagicMock(return_value=False)
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config)

        # TemporaryDirectory used as context manager (via with statement)
        mock_td.return_value.__enter__.assert_called_once()
        mock_td.return_value.__exit__.assert_called_once()

    def test_work_dir_cleaned_up_on_error(self, tmp_path: Path) -> None:
        """On render failure, temporary work dir should still be cleaned up."""
        from autopilot.render.ffmpeg_render import RenderError
        from autopilot.render.router import RoutingError, route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        config = _make_config()

        work = tmp_path / "render_work"
        work.mkdir()

        with patch("autopilot.render.router.render_simple",
                    side_effect=RenderError("fail")), \
             patch("subprocess.run"), \
             patch("tempfile.TemporaryDirectory") as mock_td:
            mock_td.return_value.__enter__ = MagicMock(return_value=str(work))
            mock_td.return_value.__exit__ = MagicMock(return_value=False)
            with pytest.raises(RoutingError):
                route_and_render("n1", db, config)

        # __exit__ must still be called for cleanup even on error
        mock_td.return_value.__exit__.assert_called_once()


class TestClipDispatching:
    """Verify route_and_render dispatches to correct renderers."""

    def test_fast_clip_calls_render_simple(self) -> None:
        """A clip classified 'fast' should call render_simple."""
        from autopilot.render.router import route_and_render

        edl = _make_edl(crop_modes={"clip_1": "center"})
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("autopilot.render.router.render_complex") as mock_rc, \
             patch("subprocess.run"):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config)

        mock_rs.assert_called_once()
        mock_rc.assert_not_called()

    def test_slow_clip_calls_render_complex(self) -> None:
        """A clip classified 'slow' should call render_complex."""
        from autopilot.render.router import route_and_render

        edl = _make_edl(crop_modes={"clip_1": "auto_subject"})
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("autopilot.render.router.render_complex") as mock_rc, \
             patch("subprocess.run"):
            mock_rc.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config)

        mock_rc.assert_called_once()
        mock_rs.assert_not_called()

    def test_mixed_clips_dispatch_correctly(self) -> None:
        """Multiple clips with different modes should dispatch to correct renderers."""
        from autopilot.render.router import route_and_render

        clips = [
            {"clip_id": "c1", "source_path": "/a.mp4", "in_timecode": "00:00:00.000",
             "out_timecode": "00:00:05.000", "track": 1},
            {"clip_id": "c2", "source_path": "/b.mp4", "in_timecode": "00:00:00.000",
             "out_timecode": "00:00:05.000", "track": 1},
        ]
        edl = _make_edl(clips=clips, crop_modes={"c1": "center", "c2": "auto_subject"})
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("autopilot.render.router.render_complex") as mock_rc, \
             patch("subprocess.run"):
            mock_rs.return_value = Path("/tmp/seg1.mp4")
            mock_rc.return_value = Path("/tmp/seg2.mp4")
            route_and_render("n1", db, config)

        mock_rs.assert_called_once()
        mock_rc.assert_called_once()


# ---------------------------------------------------------------------------
# Final concatenation
# ---------------------------------------------------------------------------


class TestFinalConcatenation:
    """Verify route_and_render concatenates segments and mixes audio."""

    def test_concat_uses_ffmpeg(self) -> None:
        """Final concatenation should use ffmpeg concat demuxer."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("subprocess.run") as mock_run:
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config)

        # The second call to subprocess.run should be the concat
        assert mock_run.call_count >= 1
        concat_cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in concat_cmd[0]
        cmd_str = " ".join(concat_cmd)
        assert "concat" in cmd_str

    def test_output_path_contains_narrative_title(self) -> None:
        """Final output should be at output/{narrative_title}/final.mp4."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "My Video"}
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("subprocess.run"):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            result = route_and_render("n1", db, config)

        assert "My Video" in str(result)
        assert result.name == "final.mp4"

    def test_music_tracks_added_as_inputs(self) -> None:
        """Music tracks from EDL should be added as ffmpeg inputs."""
        from autopilot.render.router import route_and_render

        edl = _make_edl(music=[{"path": "/tmp/music.mp3", "level": -6}])
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("subprocess.run") as mock_run:
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config)

        concat_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(concat_cmd)
        assert "/tmp/music.mp3" in cmd_str
        assert "amix" in cmd_str


# ---------------------------------------------------------------------------
# Subtitle support
# ---------------------------------------------------------------------------


class TestSubtitleSupport:
    """Verify route_and_render generates and applies subtitles."""

    def test_subtitles_applied_when_transcript_available(self) -> None:
        """When ASR transcript exists, subtitles should be generated."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        segments = [
            {"start": 0.0, "end": 2.5, "text": "Hello world"},
            {"start": 2.5, "end": 5.0, "text": "Testing"},
        ]
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = {"segments_json": json.dumps(segments)}
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("subprocess.run") as mock_run:
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config)

        concat_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(concat_cmd)
        assert "subtitles=" in cmd_str

    def test_no_subtitles_when_no_transcript(self) -> None:
        """When no ASR transcript, no subtitle filter should appear."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("subprocess.run") as mock_run:
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config)

        concat_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(concat_cmd)
        assert "subtitles=" not in cmd_str


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify route_and_render error handling."""

    def test_render_simple_failure_raises_routing_error(self) -> None:
        """RenderError from render_simple should be wrapped in RoutingError."""
        from autopilot.render.ffmpeg_render import RenderError
        from autopilot.render.router import RoutingError, route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        config = _make_config()

        with (
            patch(
                "autopilot.render.router.render_simple",
                side_effect=RenderError("ffmpeg failed"),
            ),
            pytest.raises(RoutingError, match="clip_1"),
        ):
            route_and_render("n1", db, config)

    def test_corrupt_edl_json_raises_routing_error(self) -> None:
        """Corrupt edl_json should raise RoutingError."""
        from autopilot.render.router import RoutingError, route_and_render

        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": "not valid json{{{"}
        config = _make_config()

        with pytest.raises(RoutingError, match="Corrupt edl_json"):
            route_and_render("n1", db, config)

    def test_missing_narrative_still_works(self) -> None:
        """Even if narrative not found, should use 'untitled' default."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = None
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, \
             patch("subprocess.run"):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            result = route_and_render("n1", db, config)

        assert "untitled" in str(result)
