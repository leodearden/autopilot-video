"""Tests for autopilot.render.router — render routing and assembly."""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
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
        assert param_names == ["narrative_id", "db", "config", "output_dir"]

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
        "clips": clips
        or [
            {
                "clip_id": "clip_1",
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
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = None
        config = _make_config()

        with pytest.raises(RoutingError, match="No edit plan"):
            route_and_render("narr_1", db, config, Path("/tmp/test_output"))

    def test_loads_edl_from_edit_plan(self) -> None:
        """route_and_render should parse edl_json from edit plan and use clip data."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "narr_1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("autopilot.render.router.render_complex"),
            patch("subprocess.run"),
        ):
            mock_rs.return_value = Path("/tmp/segment.mp4")
            route_and_render("narr_1", db, config, Path("/tmp/test_output"))

        db.get_edit_plan.assert_called_once_with("narr_1")
        # Verify the parsed EDL clip was passed to render_simple
        mock_rs.assert_called_once()
        rendered_clip = mock_rs.call_args[0][0]
        assert rendered_clip["clip_id"] == "clip_1"
        # Regression guard: source_path must be the resolved mock value
        assert rendered_clip["source_path"] == "/fake/source.mp4"


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
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        work = tmp_path / "render_work"
        work.mkdir()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run"),
            patch("tempfile.TemporaryDirectory") as mock_td,
        ):
            mock_td.return_value.__enter__ = MagicMock(return_value=str(work))
            mock_td.return_value.__exit__ = MagicMock(return_value=False)
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        # TemporaryDirectory used as context manager (via with statement)
        mock_td.return_value.__enter__.assert_called_once()
        mock_td.return_value.__exit__.assert_called_once()

    def test_work_dir_cleaned_up_on_error(self, tmp_path: Path) -> None:
        """On render failure, temporary work dir should still be cleaned up."""
        from autopilot.render.ffmpeg_render import RenderError
        from autopilot.render.router import RoutingError, route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        config = _make_config()

        work = tmp_path / "render_work"
        work.mkdir()

        with (
            patch("autopilot.render.router.render_simple", side_effect=RenderError("fail")),
            patch("subprocess.run"),
            patch("tempfile.TemporaryDirectory") as mock_td,
        ):
            mock_td.return_value.__enter__ = MagicMock(return_value=str(work))
            mock_td.return_value.__exit__ = MagicMock(return_value=False)
            with pytest.raises(RoutingError):
                route_and_render("n1", db, config, Path("/tmp/test_output"))

        # __exit__ must still be called for cleanup even on error
        mock_td.return_value.__exit__.assert_called_once()


class TestClipDispatching:
    """Verify route_and_render dispatches to correct renderers."""

    def test_fast_clip_calls_render_simple(self) -> None:
        """A clip classified 'fast' should call render_simple."""
        from autopilot.render.router import route_and_render

        edl = _make_edl(crop_modes={"clip_1": "center"})
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("autopilot.render.router.render_complex") as mock_rc,
            patch("subprocess.run"),
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        mock_rs.assert_called_once()
        mock_rc.assert_not_called()

    def test_slow_clip_calls_render_complex(self) -> None:
        """A clip classified 'slow' should call render_complex."""
        from autopilot.render.router import route_and_render

        edl = _make_edl(crop_modes={"clip_1": "auto_subject"})
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("autopilot.render.router.render_complex") as mock_rc,
            patch("subprocess.run"),
        ):
            mock_rc.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        mock_rc.assert_called_once()
        mock_rs.assert_not_called()

    def test_mixed_clips_dispatch_correctly(self) -> None:
        """Multiple clips with different modes should dispatch to correct renderers."""
        from autopilot.render.router import route_and_render

        clips = [
            {
                "clip_id": "c1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:05.000",
                "track": 1,
            },
            {
                "clip_id": "c2",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:05.000",
                "track": 1,
            },
        ]
        edl = _make_edl(clips=clips, crop_modes={"c1": "center", "c2": "auto_subject"})
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        db.get_media.side_effect = lambda mid: {"file_path": f"/resolved/{mid}.mp4"}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("autopilot.render.router.render_complex") as mock_rc,
            patch("subprocess.run"),
        ):
            mock_rs.return_value = Path("/tmp/seg1.mp4")
            mock_rc.return_value = Path("/tmp/seg2.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        mock_rs.assert_called_once()
        mock_rc.assert_called_once()


# ---------------------------------------------------------------------------
# Crop path loading from DB
# ---------------------------------------------------------------------------


class TestCropPathLoading:
    """Verify crop_path is loaded from DB for slow-path clips."""

    def test_crop_path_loaded_from_db_for_slow_clip(self) -> None:
        """Slow-path clip should have crop_path loaded via db.get_crop_path."""
        from autopilot.render.router import route_and_render

        edl = _make_edl(
            crop_modes=[
                {"clip_id": "clip_1", "mode": "auto_subject", "subject_track_id": "face_0"}
            ],
        )
        crop_data = np.full((30, 2), [100, 50], dtype=np.float64)
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        db.get_crop_path.return_value = {"path_data": crop_data.tolist()}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple"),
            patch("autopilot.render.router.render_complex") as mock_rc,
            patch("subprocess.run"),
        ):
            mock_rc.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        mock_rc.assert_called_once()
        # The crop_path arg (index 1) should be an ndarray loaded from DB
        crop_arg = mock_rc.call_args[0][1]
        assert isinstance(crop_arg, np.ndarray)

    def test_slow_clip_no_crop_raises_routing_error(self) -> None:
        """Slow-path clip with no crop data should raise RoutingError."""
        from autopilot.render.router import RoutingError, route_and_render

        edl = _make_edl(
            crop_modes=[
                {"clip_id": "clip_1", "mode": "auto_subject", "subject_track_id": "face_0"}
            ],
        )
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_crop_path.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple"),
            patch("autopilot.render.router.render_complex"),
            patch("subprocess.run"),
        ):
            with pytest.raises(RoutingError, match="crop"):
                route_and_render("n1", db, config, Path("/tmp/test_output"))


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
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

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
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "My Video"}
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, patch("subprocess.run"):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            result = route_and_render("n1", db, config, Path("/tmp/test_output"))

        assert "My Video" in str(result)
        assert result.name == "final.mp4"

    def test_music_tracks_added_as_inputs(self) -> None:
        """Music tracks from EDL should be added as ffmpeg inputs."""
        from autopilot.render.router import route_and_render

        edl = _make_edl(music=[{"path": "/tmp/music.mp3", "level": -6}])
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        concat_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(concat_cmd)
        assert "/tmp/music.mp3" in cmd_str
        assert "amix" in cmd_str

    def test_amix_has_output_label_and_map(self) -> None:
        """amix filter_complex must end with [aout] and use -map for routing."""
        from autopilot.render.router import route_and_render

        edl = _make_edl(music=[{"path": "/tmp/music.mp3", "level": -6}])
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        concat_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(concat_cmd)
        # filter_complex must have [aout] output label
        assert "[aout]" in cmd_str
        # Must have -map for both video and audio streams
        assert "-map" in cmd_str


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
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = {"segments_json": json.dumps(segments)}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        concat_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(concat_cmd)
        assert "subtitles=" in cmd_str

    def test_no_subtitles_when_no_transcript(self) -> None:
        """When no ASR transcript, no subtitle filter should appear."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        concat_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(concat_cmd)
        assert "subtitles=" not in cmd_str


# ---------------------------------------------------------------------------
# Subtitles + audio mixing in single filter_complex
# ---------------------------------------------------------------------------


class TestSubtitlesWithAudioMixing:
    """Verify subtitles are merged into filter_complex when audio mixing is active."""

    def test_subtitles_in_filter_complex_with_audio_mixing(self) -> None:
        """When both amix and subtitles active, single filter_complex, no -vf."""
        from autopilot.render.router import route_and_render

        segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        edl = _make_edl(music=[{"path": "/tmp/music.mp3", "level": -6}])
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = {"segments_json": json.dumps(segments)}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        concat_cmd = mock_run.call_args[0][0]

        # 1. Only ONE -filter_complex (no standalone -vf)
        assert concat_cmd.count("-filter_complex") == 1
        assert "-vf" not in concat_cmd

        # 2. filter_complex contains video node with subtitles
        fc_idx = concat_cmd.index("-filter_complex")
        fc_val = concat_cmd[fc_idx + 1]
        assert "subtitles=" in fc_val
        assert "[vout]" in fc_val
        assert "[aout]" in fc_val

        # 3. -map references both [vout] and [aout]
        cmd_str = " ".join(concat_cmd)
        assert "[vout]" in cmd_str
        assert "[aout]" in cmd_str


# ---------------------------------------------------------------------------
# Stream copy vs filter conflict
# ---------------------------------------------------------------------------


class TestStreamCopyVsFilter:
    """Verify -c copy is not used when video filters are present."""

    def test_no_stream_copy_when_subtitles_present(self) -> None:
        """When subtitles are active (no audio mixing), must NOT use -c copy."""
        from autopilot.render.router import route_and_render

        segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        edl = _make_edl()  # no music -> no audio mixing
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = {"segments_json": json.dumps(segments)}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        concat_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(concat_cmd)
        # Subtitles should be present
        assert "subtitles=" in cmd_str
        # -c copy must NOT be present (incompatible with -vf)
        # Check adjacent pairs for "-c" followed by "copy"
        for i, arg in enumerate(concat_cmd[:-1]):
            if arg == "-c" and concat_cmd[i + 1] == "copy":
                pytest.fail("-c copy found in command alongside subtitles filter")

    def test_stream_copy_when_no_filters(self) -> None:
        """With no subtitles and no audio mixing, -c copy SHOULD be used."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()  # no music, no voiceovers
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None  # no subtitles
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        concat_cmd = mock_run.call_args[0][0]
        # Should have -c copy
        found_copy = False
        for i, arg in enumerate(concat_cmd[:-1]):
            if arg == "-c" and concat_cmd[i + 1] == "copy":
                found_copy = True
                break
        assert found_copy, f"-c copy not found in command: {concat_cmd}"


# ---------------------------------------------------------------------------
# Transcript lookup by media_id
# ---------------------------------------------------------------------------


class TestTranscriptByMediaId:
    """Verify transcripts are fetched per-clip by media_id, not narrative_id."""

    def test_transcript_fetched_by_media_id(self) -> None:
        """db.get_transcript should be called with media_id (=clip_id), not narrative_id."""
        from autopilot.render.router import route_and_render

        segments = [{"start": 0.0, "end": 2.5, "text": "Hello"}]
        edl = _make_edl()
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = {"segments_json": json.dumps(segments)}
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, patch("subprocess.run"):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        # Should have been called with clip_id ("clip_1"), not "n1"
        transcript_calls = db.get_transcript.call_args_list
        called_ids = [c[0][0] for c in transcript_calls]
        assert "clip_1" in called_ids
        assert "n1" not in called_ids

    def test_multiple_clips_transcripts_combined(self) -> None:
        """With 2 clips, transcripts from both should be combined in SRT."""
        from autopilot.render.router import route_and_render

        clips = [
            {
                "clip_id": "c1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:05.000",
                "track": 1,
            },
            {
                "clip_id": "c2",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:05.000",
                "track": 1,
            },
        ]
        edl = _make_edl(clips=clips)
        seg1 = [{"start": 0.0, "end": 2.0, "text": "First clip"}]
        seg2 = [{"start": 0.0, "end": 2.0, "text": "Second clip"}]

        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_media.side_effect = lambda mid: {"file_path": f"/resolved/{mid}.mp4"}

        def _get_transcript(media_id: str):
            if media_id == "c1":
                return {"segments_json": json.dumps(seg1)}
            elif media_id == "c2":
                return {"segments_json": json.dumps(seg2)}
            return None

        db.get_transcript.side_effect = _get_transcript
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        # Verify subtitles filter is in the concat command
        concat_cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(concat_cmd)
        assert "subtitles=" in cmd_str


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify route_and_render error handling."""

    def test_concat_timeout_error_chains_cause(self) -> None:
        """RoutingError.__cause__ should be the original TimeoutExpired exception."""
        import subprocess as _subprocess

        from autopilot.render.router import RoutingError, route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        timeout_exc = _subprocess.TimeoutExpired(cmd="ffmpeg", timeout=1800)
        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run", side_effect=timeout_exc),
            pytest.raises(RoutingError) as exc_info,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        assert exc_info.value.__cause__ is timeout_exc

    def test_concat_passes_timeout_to_subprocess(self) -> None:
        """Concat subprocess.run call should include timeout=CONCAT_TIMEOUT_SECONDS."""
        from autopilot.render.router import CONCAT_TIMEOUT_SECONDS, route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run") as mock_run,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        # The concat subprocess.run call should have timeout kwarg
        assert mock_run.call_args[1].get("timeout") == CONCAT_TIMEOUT_SECONDS

    def test_concat_timeout_raises_routing_error(self) -> None:
        """TimeoutExpired on concat subprocess should be wrapped in RoutingError."""
        import subprocess as _subprocess

        from autopilot.render.router import RoutingError, route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch(
                "subprocess.run",
                side_effect=_subprocess.TimeoutExpired(
                    cmd="ffmpeg",
                    timeout=1800,
                ),
            ),
            pytest.raises(RoutingError) as exc_info,
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        msg = str(exc_info.value).lower()
        assert "timeout" in msg or "timed out" in msg

    def test_render_simple_failure_raises_routing_error(self) -> None:
        """RenderError from render_simple should be wrapped in RoutingError."""
        from autopilot.render.ffmpeg_render import RenderError
        from autopilot.render.router import RoutingError, route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
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
            route_and_render("n1", db, config, Path("/tmp/test_output"))

    def test_corrupt_edl_json_raises_routing_error(self) -> None:
        """Corrupt edl_json should raise RoutingError."""
        from autopilot.render.router import RoutingError, route_and_render

        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": "not valid json{{{"}
        config = _make_config()

        with pytest.raises(RoutingError, match="Corrupt edl_json"):
            route_and_render("n1", db, config, Path("/tmp/test_output"))

    def test_missing_narrative_still_works(self) -> None:
        """Even if narrative not found, should use 'untitled' default."""
        from autopilot.render.router import route_and_render

        edl = _make_edl()
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = None
        db.get_transcript.return_value = None
        config = _make_config()

        with patch("autopilot.render.router.render_simple") as mock_rs, patch("subprocess.run"):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            result = route_and_render("n1", db, config, Path("/tmp/test_output"))

        assert "untitled" in str(result)


# ---------------------------------------------------------------------------
# Source path resolution from clip_id
# ---------------------------------------------------------------------------


class TestSourcePathResolution:
    """Verify route_and_render resolves source_path via db.get_media when missing."""

    def test_db_get_media_called_for_clip_without_source_path(self) -> None:
        """When clip has clip_id but no source_path, db.get_media should be called."""
        from autopilot.render.router import route_and_render

        clips = [
            {
                "clip_id": "clip_1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        db.get_media.return_value = {"file_path": "/resolved/clip.mp4"}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run"),
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        db.get_media.assert_called_with("clip_1")

    def test_resolved_source_path_passed_to_renderer(self) -> None:
        """The resolved source_path should be set on the clip dict passed to renderer."""
        from autopilot.render.router import route_and_render

        clips = [
            {
                "clip_id": "clip_1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        db.get_media.return_value = {"file_path": "/resolved/clip.mp4"}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run"),
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        rendered_clip = mock_rs.call_args[0][0]
        assert rendered_clip["source_path"] == "/resolved/clip.mp4"

    def test_missing_media_raises_routing_error(self) -> None:
        """When db.get_media returns None, RoutingError should be raised."""
        from autopilot.render.router import RoutingError, route_and_render

        clips = [
            {
                "clip_id": "clip_1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_media.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple"),
            patch("autopilot.render.router.render_complex"),
            patch("subprocess.run"),
        ):
            with pytest.raises(RoutingError, match="No media record"):
                route_and_render("n1", db, config, Path("/tmp/test_output"))

    def test_source_path_resolved_for_slow_clip(self) -> None:
        """Slow-path (auto_subject) clip without source_path should also be resolved."""
        from autopilot.render.router import route_and_render

        clips = [
            {
                "clip_id": "clip_1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        crop_data = np.full((30, 2), [100, 50], dtype=np.float64)
        edl = _make_edl(
            clips=clips,
            crop_modes=[
                {"clip_id": "clip_1", "mode": "auto_subject", "subject_track_id": 0}
            ],
        )
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        db.get_media.return_value = {"file_path": "/resolved/slow_clip.mp4"}
        db.get_crop_path.return_value = {"path_data": crop_data.tolist()}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple"),
            patch("autopilot.render.router.render_complex") as mock_rc,
            patch("subprocess.run"),
        ):
            mock_rc.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        rendered_clip = mock_rc.call_args[0][0]
        assert rendered_clip["source_path"] == "/resolved/slow_clip.mp4"
        db.get_media.assert_called_with("clip_1")

    def test_source_path_coerced_from_pathlib_path(self) -> None:
        """When db.get_media returns a Path object for file_path, it should be coerced to str."""
        from autopilot.render.router import route_and_render

        clips = [
            {
                "clip_id": "clip_1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        db.get_media.return_value = {"file_path": Path("/resolved/clip.mp4")}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run"),
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        rendered_clip = mock_rs.call_args[0][0]
        assert rendered_clip["source_path"] == "/resolved/clip.mp4"
        assert isinstance(rendered_clip["source_path"], str)

    def test_missing_file_path_key_raises_routing_error(self) -> None:
        """When db.get_media returns a dict without file_path, RoutingError should be raised."""
        from autopilot.render.router import RoutingError, route_and_render

        clips = [
            {
                "clip_id": "clip_1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_media.return_value = {}
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple"),
            patch("autopilot.render.router.render_complex"),
            patch("subprocess.run"),
        ):
            with pytest.raises(RoutingError, match="file_path"):
                route_and_render("n1", db, config, Path("/tmp/test_output"))

    def test_multiple_clips_each_resolved(self) -> None:
        """Each clip without source_path should trigger its own db.get_media call."""
        from autopilot.render.router import route_and_render

        clips = [
            {
                "clip_id": "c1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:05.000",
                "track": 1,
            },
            {
                "clip_id": "c2",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:05.000",
                "track": 1,
            },
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None

        def _get_media(media_id: str):
            return {"file_path": f"/resolved/{media_id}.mp4"}

        db.get_media.side_effect = _get_media
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run"),
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        called_ids = [c[0][0] for c in db.get_media.call_args_list]
        assert "c1" in called_ids
        assert "c2" in called_ids
        assert db.get_media.call_count == 2

    def test_existing_source_path_not_overwritten(self) -> None:
        """Clip with source_path already set should NOT trigger db.get_media."""
        from autopilot.render.router import route_and_render

        clips = [
            {
                "clip_id": "clip_1",
                "source_path": "/already/set.mp4",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_media.return_value = {"file_path": "/fake/source.mp4"}
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run"),
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        db.get_media.assert_not_called()
        rendered_clip = mock_rs.call_args[0][0]
        assert rendered_clip["source_path"] == "/already/set.mp4"

    def test_no_clip_id_and_no_source_path_raises_routing_error(self) -> None:
        """Clip with neither clip_id nor source_path should raise RoutingError early."""
        from autopilot.render.router import RoutingError, route_and_render

        clips = [
            {
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple"),
            patch("subprocess.run"),
        ):
            with pytest.raises(RoutingError, match="no clip_id"):
                route_and_render("n1", db, config, Path("/tmp/test_output"))

    def test_media_missing_file_path_raises_routing_error(self) -> None:
        """Media record without file_path key should raise RoutingError, not KeyError."""
        from autopilot.render.router import RoutingError, route_and_render

        clips = [
            {
                "clip_id": "clip_1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        db.get_media.return_value = {"media_id": "clip_1"}  # no file_path key
        config = _make_config()

        with (
            patch("autopilot.render.router.render_simple"),
            patch("subprocess.run"),
        ):
            with pytest.raises(RoutingError, match="missing file_path"):
                route_and_render("n1", db, config, Path("/tmp/test_output"))

    def test_source_path_resolution_does_not_mutate_original_clip(self) -> None:
        """Resolving source_path should NOT mutate the original clip dict."""
        from autopilot.render.router import route_and_render

        clips = [
            {
                "clip_id": "clip_1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }
        ]
        edl = _make_edl(clips=clips)
        db = MagicMock()
        db.get_edit_plan.return_value = {"edl_json": json.dumps(edl)}
        db.get_narrative.return_value = {"narrative_id": "n1", "title": "Test"}
        db.get_transcript.return_value = None
        db.get_media.return_value = {"file_path": "/resolved/clip.mp4"}
        config = _make_config()

        # Capture the deserialized clips to verify they aren't mutated
        parsed_clips: list[dict] = []
        _real_loads = json.loads

        def _capturing_loads(s: object, *a: object, **kw: object) -> object:
            result = _real_loads(str(s), *a, **kw)  # type: ignore[arg-type]
            if isinstance(result, dict) and "clips" in result:
                parsed_clips.extend(result["clips"])
            return result

        with (
            patch("autopilot.render.router.json.loads", side_effect=_capturing_loads),
            patch("autopilot.render.router.render_simple") as mock_rs,
            patch("subprocess.run"),
        ):
            mock_rs.return_value = Path("/tmp/seg.mp4")
            route_and_render("n1", db, config, Path("/tmp/test_output"))

        # The deserialized clip dict should NOT have source_path added
        assert len(parsed_clips) == 1
        assert "source_path" not in parsed_clips[0]
        # But the renderer should still have received the resolved source_path
        rendered_clip = mock_rs.call_args[0][0]
        assert rendered_clip["source_path"] == "/resolved/clip.mp4"
