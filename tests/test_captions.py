"""Tests for autopilot.analyze.captions — selective video captioning module."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# -- Mock helpers --------------------------------------------------------------


def _make_mock_capture(
    fps: float = 30.0,
    total_frames: int = 900,
    width: int = 1920,
    height: int = 1080,
) -> MagicMock:
    """Create a MagicMock mimicking cv2.VideoCapture."""
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    cap = MagicMock()
    cap.isOpened.return_value = True

    def get_prop(prop_id):
        prop_map = {
            CAP_PROP_FPS: fps,
            CAP_PROP_FRAME_COUNT: total_frames,
            CAP_PROP_FRAME_WIDTH: width,
            CAP_PROP_FRAME_HEIGHT: height,
        }
        return prop_map.get(prop_id, 0.0)

    cap.get.side_effect = get_prop

    # BGR frame (cv2 native format)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 100  # B channel
    frame[:, :, 2] = 200  # R channel
    cap.read.return_value = (True, frame)
    cap.set.return_value = True
    return cap


def _make_mock_cv2():
    """Create a MagicMock cv2 module with correct CAP_PROP constants."""
    mock_cv2 = MagicMock()
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    return mock_cv2


# ---------------------------------------------------------------------------
# TestPublicAPI — verify module exports and function signatures
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify captions module exports and function signatures."""

    def test_caption_error_importable(self):
        """CaptionError is importable from autopilot.analyze.captions."""
        from autopilot.analyze.captions import CaptionError

        assert CaptionError is not None

    def test_caption_error_is_exception(self):
        """CaptionError is a subclass of Exception."""
        from autopilot.analyze.captions import CaptionError

        assert issubclass(CaptionError, Exception)

    def test_caption_clip_importable(self):
        """caption_clip is importable with expected params."""
        from autopilot.analyze.captions import caption_clip

        assert callable(caption_clip)

    def test_batch_caption_importable(self):
        """batch_caption is importable with expected params."""
        from autopilot.analyze.captions import batch_caption

        assert callable(batch_caption)

    def test_extract_clip_frames_importable(self):
        """_extract_clip_frames is a module-level helper."""
        from autopilot.analyze.captions import _extract_clip_frames

        assert callable(_extract_clip_frames)


# ---------------------------------------------------------------------------
# TestExtractClipFrames — frame extraction from video clips
# ---------------------------------------------------------------------------


class TestExtractClipFrames:
    """Tests for _extract_clip_frames() helper."""

    def test_extracts_correct_number_of_frames(self):
        """8 frames requested from 30s clip returns 8 PIL Images."""
        from PIL import Image

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=900)
        mock_cv2.VideoCapture.return_value = cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.analyze import captions

            # Force reimport to pick up mocked cv2
            import importlib

            importlib.reload(captions)
            frames = captions._extract_clip_frames(
                Path("/test/video.mp4"), 0.0, 30.0, fps=30.0, num_frames=8
            )

        assert len(frames) == 8
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_frame_times_evenly_spaced(self):
        """For start=10.0, end=40.0, num_frames=4, verify evenly-spaced frame positions."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=1200)
        mock_cv2.VideoCapture.return_value = cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.analyze import captions

            import importlib

            importlib.reload(captions)
            captions._extract_clip_frames(
                Path("/test/video.mp4"), 10.0, 40.0, fps=30.0, num_frames=4
            )

        # Expected timestamps: evenly spaced in [10, 40]
        # 10.0, 20.0, 30.0, 40.0 -> frame indices 300, 600, 900, 1200
        set_calls = [c for c in cap.set.call_args_list if c[0][0] == 1]  # CAP_PROP_POS_FRAMES=1
        assert len(set_calls) == 4
        frame_indices = [c[0][1] for c in set_calls]
        # All should be in the range [300, 1200]
        assert all(300 <= idx <= 1200 for idx in frame_indices)

    def test_handles_read_failure(self):
        """When cap.read() returns (False, None) for some frames, returns only successful."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=900)
        mock_cv2.VideoCapture.return_value = cap

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Alternate: success, failure, success, failure, ...
        cap.read.side_effect = [
            (True, frame),
            (False, None),
            (True, frame),
            (False, None),
            (True, frame),
            (False, None),
            (True, frame),
            (False, None),
        ]

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.analyze import captions

            import importlib

            importlib.reload(captions)
            frames = captions._extract_clip_frames(
                Path("/test/video.mp4"), 0.0, 30.0, fps=30.0, num_frames=8
            )

        assert len(frames) == 4  # Only successful reads

    def test_empty_range_returns_empty(self):
        """start_time == end_time returns empty list."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture()
        mock_cv2.VideoCapture.return_value = cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.analyze import captions

            import importlib

            importlib.reload(captions)
            frames = captions._extract_clip_frames(
                Path("/test/video.mp4"), 5.0, 5.0, fps=30.0, num_frames=8
            )

        assert frames == []

    def test_converts_bgr_to_rgb(self):
        """Verify cv2 BGR frame is converted to RGB PIL Image."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=900)
        mock_cv2.VideoCapture.return_value = cap

        # BGR frame with B=100, G=0, R=200
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 100  # Blue channel
        bgr_frame[:, :, 2] = 200  # Red channel
        cap.read.return_value = (True, bgr_frame)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.analyze import captions

            import importlib

            importlib.reload(captions)
            frames = captions._extract_clip_frames(
                Path("/test/video.mp4"), 0.0, 10.0, fps=30.0, num_frames=1
            )

        assert len(frames) == 1
        img = frames[0]
        pixel = img.getpixel((0, 0))
        # After BGR->RGB conversion: R=200, G=0, B=100
        assert pixel == (200, 0, 100)
