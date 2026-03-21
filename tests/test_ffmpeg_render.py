"""Tests for autopilot.render.ffmpeg_render — FFmpeg fast-path rendering."""

from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from autopilot.config import OutputConfig

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    resolution: tuple[int, int] = (1920, 1080),
    codec: str = "h264",
    quality_crf: int = 18,
    audio_bitrate: str = "256k",
    target_loudness_lufs: int = -16,
) -> OutputConfig:
    """Create an OutputConfig for testing."""
    return OutputConfig(
        resolution=resolution,
        codec=codec,
        quality_crf=quality_crf,
        audio_bitrate=audio_bitrate,
        target_loudness_lufs=target_loudness_lufs,
    )


def _make_edl_entry(
    source_path: str = "/tmp/src/clip.mp4",
    in_timecode: str = "00:00:00.000",
    out_timecode: str = "00:00:10.000",
    **kwargs: object,
) -> dict:
    """Create a minimal EDL entry dict for testing."""
    entry: dict = {
        "clip_id": "clip_1",
        "source_path": source_path,
        "in_timecode": in_timecode,
        "out_timecode": out_timecode,
        "track": 1,
    }
    entry.update(kwargs)
    return entry


# ---------------------------------------------------------------------------
# FFmpeg command construction
# ---------------------------------------------------------------------------


class TestRenderSimpleCommand:
    """Verify render_simple builds correct FFmpeg commands."""

    def test_calls_subprocess_run(self, tmp_path: Path) -> None:
        """render_simple should call subprocess.run."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        mock_run.assert_called_once()

    def test_ffmpeg_binary(self, tmp_path: Path) -> None:
        """render_simple should invoke 'ffmpeg' as the command."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"

    def test_input_file(self, tmp_path: Path) -> None:
        """render_simple should include -i with source path from edl_entry."""
        config = _make_config()
        edl_entry = _make_edl_entry(source_path="/data/clip.mp4")
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        assert "-i" in cmd
        i_idx = cmd.index("-i")
        assert cmd[i_idx + 1] == "/data/clip.mp4"

    def test_nvenc_codec(self, tmp_path: Path) -> None:
        """render_simple should use h264_nvenc video codec."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        assert "-c:v" in cmd
        cv_idx = cmd.index("-c:v")
        assert cmd[cv_idx + 1] == "h264_nvenc"

    def test_crf_from_config(self, tmp_path: Path) -> None:
        """render_simple should use quality_crf from config."""
        config = _make_config(quality_crf=23)
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        assert "-crf" in cmd
        crf_idx = cmd.index("-crf")
        assert cmd[crf_idx + 1] == "23"

    def test_loudnorm_audio_filter(self, tmp_path: Path) -> None:
        """render_simple should apply loudnorm at config's target LUFS."""
        config = _make_config(target_loudness_lufs=-14)
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        assert "-af" in cmd
        af_idx = cmd.index("-af")
        assert "loudnorm=I=-14" in cmd[af_idx + 1]

    def test_aac_audio_at_config_bitrate(self, tmp_path: Path) -> None:
        """render_simple should encode audio as AAC at config bitrate."""
        config = _make_config(audio_bitrate="192k")
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        assert "-c:a" in cmd
        ca_idx = cmd.index("-c:a")
        assert cmd[ca_idx + 1] == "aac"
        assert "-b:a" in cmd
        ba_idx = cmd.index("-b:a")
        assert cmd[ba_idx + 1] == "192k"

    def test_overwrite_flag(self, tmp_path: Path) -> None:
        """render_simple should include -y overwrite flag."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        assert "-y" in cmd

    def test_output_path_is_last_arg(self, tmp_path: Path) -> None:
        """render_simple should put output path as last argument."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == str(output)

    def test_returns_output_path(self, tmp_path: Path) -> None:
        """render_simple should return the output_path."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run"):
            from autopilot.render.ffmpeg_render import render_simple

            result = render_simple(edl_entry, None, output, config)

        assert result == output

    def test_timecode_seek(self, tmp_path: Path) -> None:
        """render_simple should use -ss for seek and -t for duration."""
        config = _make_config()
        edl_entry = _make_edl_entry(
            in_timecode="00:01:30.000",
            out_timecode="00:02:00.000",
        )
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        assert "-ss" in cmd
        ss_idx = cmd.index("-ss")
        assert cmd[ss_idx + 1] == "00:01:30.000"
        # Uses -t (duration) instead of -to (absolute time)
        assert "-t" in cmd
        t_idx = cmd.index("-t")
        assert float(cmd[t_idx + 1]) == pytest.approx(30.0)

    def test_check_true(self, tmp_path: Path) -> None:
        """render_simple should call subprocess.run with check=True."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        assert mock_run.call_args[1].get("check") is True


# ---------------------------------------------------------------------------
# Static crop
# ---------------------------------------------------------------------------


class TestRenderSimpleStaticCrop:
    """Verify render_simple applies static crop filter when crop_path provided."""

    def test_static_crop_in_vf(self, tmp_path: Path) -> None:
        """When crop_path is static (all rows same), add crop filter to ffmpeg."""
        config = _make_config(resolution=(1920, 1080))
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"
        # Static crop: all rows identical (x=100, y=50)
        crop_path = np.full((30, 2), [100, 50], dtype=np.float64)

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, crop_path, output, config)

        cmd = mock_run.call_args[0][0]
        # Should have a video filter with crop
        cmd_str = " ".join(cmd)
        assert "crop=" in cmd_str

    def test_static_crop_values(self, tmp_path: Path) -> None:
        """Static crop filter should include correct x, y and output resolution w, h."""
        config = _make_config(resolution=(1280, 720))
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"
        crop_path = np.full((30, 2), [200, 100], dtype=np.float64)

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, crop_path, output, config)

        cmd = mock_run.call_args[0][0]
        # Find the vf or filter_complex arg containing crop
        cmd_str = " ".join(cmd)
        # crop=w:h:x:y format
        assert "crop=1280:720:200:100" in cmd_str

    def test_single_frame_crop(self, tmp_path: Path) -> None:
        """A single-frame crop_path should still be treated as static."""
        config = _make_config(resolution=(1920, 1080))
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"
        crop_path = np.array([[300, 150]], dtype=np.float64)

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, crop_path, output, config)

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "crop=" in cmd_str


# ---------------------------------------------------------------------------
# No crop and error handling
# ---------------------------------------------------------------------------


class TestRenderSimpleNoCrop:
    """Verify render_simple handles None crop_path correctly."""

    def test_no_crop_filter_when_none(self, tmp_path: Path) -> None:
        """When crop_path is None, no crop filter should appear."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "crop=" not in cmd_str


class TestRenderSimpleErrors:
    """Verify render_simple error handling."""

    def test_subprocess_timeout_raises_render_error(self, tmp_path: Path) -> None:
        """TimeoutExpired should be wrapped in RenderError with clip_id and timeout."""
        from autopilot.render.ffmpeg_render import RenderError, render_simple

        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(
                    cmd="ffmpeg", timeout=600,
                ),
            ),
            pytest.raises(RenderError, match="clip_1") as exc_info,
        ):
            render_simple(edl_entry, None, output, config)

        assert "timeout" in str(exc_info.value).lower() or "timed out" in str(exc_info.value).lower()
        assert "600" in str(exc_info.value)

    def test_subprocess_error_wrapped_in_render_error(self, tmp_path: Path) -> None:
        """CalledProcessError should be wrapped in RenderError."""
        from autopilot.render.ffmpeg_render import RenderError, render_simple

        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"

        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
            ),
            pytest.raises(RenderError) as exc_info,
        ):
            render_simple(edl_entry, None, output, config)

        assert "clip_1" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Xfade transitions
# ---------------------------------------------------------------------------


class TestRenderSimpleTransitions:
    """Verify render_simple handles xfade transitions."""

    @staticmethod
    def _get_vf_value(cmd: list[str]) -> str | None:
        """Extract the -vf filter argument value, or None if not present."""
        if "-vf" in cmd:
            return cmd[cmd.index("-vf") + 1]
        return None

    def test_crossfade_transition_adds_fade(self, tmp_path: Path) -> None:
        """When edl_entry has transition with type='crossfade', fade filter added."""
        config = _make_config()
        edl_entry = _make_edl_entry(
            transition={"type": "crossfade", "duration": 0.5},
        )
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        vf = self._get_vf_value(cmd)
        assert vf is not None, "Expected -vf flag with fade filter"
        assert "fade" in vf.lower()

    def test_crossfade_duration(self, tmp_path: Path) -> None:
        """Crossfade transition should include specified duration."""
        config = _make_config()
        edl_entry = _make_edl_entry(
            transition={"type": "crossfade", "duration": 1.0},
        )
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        vf = self._get_vf_value(cmd)
        assert vf is not None, "Expected -vf flag with fade filter"
        assert "duration=1.0" in vf or "d=1.0" in vf

    def test_no_transition_no_fade(self, tmp_path: Path) -> None:
        """Without transition in edl_entry, no fade filter should appear."""
        config = _make_config()
        edl_entry = _make_edl_entry()  # no transition key
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]
        vf = self._get_vf_value(cmd)
        # Either no -vf at all, or if present, no fade filter
        if vf is not None:
            assert "fade" not in vf.lower()


# ---------------------------------------------------------------------------
# PTS reset and duration-based timing
# ---------------------------------------------------------------------------


class TestRenderSimpleTiming:
    """Verify PTS reset and -t duration instead of -to."""

    def test_fade_out_uses_zero_based_pts(self, tmp_path: Path) -> None:
        """For clip with in_tc='00:01:00.000', out_tc='00:01:30.000' and 0.5s crossfade:
        1. -vf chain includes setpts=PTS-STARTPTS
        2. fade-out st=29.5 (zero-based, not 89.5)
        3. -t 30.0 instead of -to 00:01:30.000
        """
        config = _make_config()
        edl_entry = _make_edl_entry(
            in_timecode="00:01:00.000",
            out_timecode="00:01:30.000",
            transition={"type": "crossfade", "duration": 0.5},
        )
        output = tmp_path / "out.mp4"

        with patch("subprocess.run") as mock_run:
            from autopilot.render.ffmpeg_render import render_simple

            render_simple(edl_entry, None, output, config)

        cmd = mock_run.call_args[0][0]

        # 1. -vf chain includes setpts=PTS-STARTPTS
        vf_idx = cmd.index("-vf")
        vf_val = cmd[vf_idx + 1]
        assert "setpts=PTS-STARTPTS" in vf_val

        # 2. fade-out st=29.5 (zero-based: 30.0 - 0.5)
        assert "st=29.5" in vf_val

        # 3. Uses -t (duration) instead of -to (absolute time)
        assert "-t" in cmd
        t_idx = cmd.index("-t")
        assert float(cmd[t_idx + 1]) == pytest.approx(30.0)
        assert "-to" not in cmd
