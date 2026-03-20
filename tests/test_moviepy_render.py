"""Tests for autopilot.render.moviepy_render — MoviePy complex-path rendering."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from autopilot.config import OutputConfig

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
    from autopilot.config import OutputConfig

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
# render_complex with per-frame crop
# ---------------------------------------------------------------------------


class TestRenderComplex:
    """Verify render_complex applies per-frame crop from crop_path."""

    def test_opens_source_video(self, tmp_path: Path) -> None:
        """render_complex should open the source video via VideoFileClip."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"
        crop_path = np.full((30, 2), [100, 50], dtype=np.float64)

        mock_clip = MagicMock()
        mock_clip.subclipped.return_value = mock_clip
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
        mock_clip.duration = 10.0
        mock_clip.fps = 30
        mock_clip.size = (1920, 1080)

        with patch(
            "autopilot.render.moviepy_render.VideoFileClip",
            return_value=mock_clip,
        ) as mock_vfc:
            from autopilot.render.moviepy_render import render_complex

            render_complex(edl_entry, crop_path, output, config)

        mock_vfc.assert_called_once_with("/tmp/src/clip.mp4")

    def test_subclips_by_timecode(self, tmp_path: Path) -> None:
        """render_complex should subclip by in/out timecodes."""
        config = _make_config()
        edl_entry = _make_edl_entry(
            in_timecode="00:00:05.000",
            out_timecode="00:00:15.000",
        )
        output = tmp_path / "out.mp4"
        crop_path = np.full((30, 2), [100, 50], dtype=np.float64)

        mock_clip = MagicMock()
        mock_clip.subclipped.return_value = mock_clip
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
        mock_clip.duration = 10.0
        mock_clip.fps = 30
        mock_clip.size = (1920, 1080)

        with patch(
            "autopilot.render.moviepy_render.VideoFileClip",
            return_value=mock_clip,
        ):
            from autopilot.render.moviepy_render import render_complex

            render_complex(edl_entry, crop_path, output, config)

        mock_clip.subclipped.assert_called_once_with(5.0, 15.0)

    def test_writes_output_file(self, tmp_path: Path) -> None:
        """render_complex should write output to the specified path."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"
        crop_path = np.full((30, 2), [100, 50], dtype=np.float64)

        mock_clip = MagicMock()
        mock_clip.subclipped.return_value = mock_clip
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
        mock_clip.duration = 10.0
        mock_clip.fps = 30
        mock_clip.size = (1920, 1080)

        with patch(
            "autopilot.render.moviepy_render.VideoFileClip",
            return_value=mock_clip,
        ):
            from autopilot.render.moviepy_render import render_complex

            render_complex(edl_entry, crop_path, output, config)

        # Should call write_videofile on the processed clip
        mock_clip.write_videofile.assert_called_once()
        call_args = mock_clip.write_videofile.call_args
        assert str(output) in str(call_args)

    def test_returns_output_path(self, tmp_path: Path) -> None:
        """render_complex should return the output_path."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"
        crop_path = np.full((30, 2), [100, 50], dtype=np.float64)

        mock_clip = MagicMock()
        mock_clip.subclipped.return_value = mock_clip
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
        mock_clip.duration = 10.0
        mock_clip.fps = 30
        mock_clip.size = (1920, 1080)

        with patch(
            "autopilot.render.moviepy_render.VideoFileClip",
            return_value=mock_clip,
        ):
            from autopilot.render.moviepy_render import render_complex

            result = render_complex(edl_entry, crop_path, output, config)

        assert result == output


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRenderComplexEdgeCases:
    """Verify render_complex handles edge cases correctly."""

    def test_empty_crop_path_raises_error(self, tmp_path: Path) -> None:
        """Empty crop_path (shape (0,2)) should raise ComplexRenderError."""
        from autopilot.render.moviepy_render import ComplexRenderError

        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"
        crop_path = np.empty((0, 2), dtype=np.float64)

        mock_clip = MagicMock()
        mock_clip.subclipped.return_value = mock_clip
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
        mock_clip.duration = 10.0
        mock_clip.fps = 30
        mock_clip.size = (1920, 1080)

        with (
            patch(
                "autopilot.render.moviepy_render.VideoFileClip",
                return_value=mock_clip,
            ),
            pytest.raises(ComplexRenderError, match="Empty crop_path"),
        ):
            from autopilot.render.moviepy_render import render_complex

            render_complex(edl_entry, crop_path, output, config)

    def test_output_directory_creation(self, tmp_path: Path) -> None:
        """render_complex should create output directory if it doesn't exist."""
        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "deep" / "nested" / "out.mp4"
        crop_path = np.full((30, 2), [100, 50], dtype=np.float64)

        mock_clip = MagicMock()
        mock_clip.subclipped.return_value = mock_clip
        mock_clip.with_effects.return_value = mock_clip
        mock_clip.transform.return_value = mock_clip
        mock_clip.duration = 10.0
        mock_clip.fps = 30
        mock_clip.size = (1920, 1080)

        with patch(
            "autopilot.render.moviepy_render.VideoFileClip",
            return_value=mock_clip,
        ):
            from autopilot.render.moviepy_render import render_complex

            render_complex(edl_entry, crop_path, output, config)

        assert output.parent.exists()

    def test_moviepy_error_wrapped(self, tmp_path: Path) -> None:
        """MoviePy exceptions should be wrapped in ComplexRenderError."""
        from autopilot.render.moviepy_render import ComplexRenderError

        config = _make_config()
        edl_entry = _make_edl_entry()
        output = tmp_path / "out.mp4"
        crop_path = np.full((30, 2), [100, 50], dtype=np.float64)

        with (
            patch(
                "autopilot.render.moviepy_render.VideoFileClip",
                side_effect=OSError("file not found"),
            ),
            pytest.raises(ComplexRenderError, match="MoviePy render failed"),
        ):
            from autopilot.render.moviepy_render import render_complex

            render_complex(edl_entry, crop_path, output, config)
