"""Tests for shot boundary detection (autopilot.analyze.scenes)."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPublicAPI:
    """Tests for module-level public API imports."""

    def test_shot_detection_error_importable(self) -> None:
        """ShotDetectionError and detect_shots are importable from autopilot.analyze.scenes."""
        from autopilot.analyze.scenes import ShotDetectionError, detect_shots

        assert ShotDetectionError is not None
        assert detect_shots is not None

    def test_shot_detection_error_is_exception(self) -> None:
        """ShotDetectionError is a subclass of Exception with message."""
        from autopilot.analyze.scenes import ShotDetectionError

        assert issubclass(ShotDetectionError, Exception)
        err = ShotDetectionError("test error")
        assert str(err) == "test error"

    def test_detect_shots_signature(self) -> None:
        """detect_shots has correct signature: media_id, video_path, db, scheduler (no config)."""
        from autopilot.analyze.scenes import detect_shots

        sig = inspect.signature(detect_shots)
        param_names = list(sig.parameters.keys())
        assert param_names == ["media_id", "video_path", "db", "scheduler"]
        # With __future__ annotations, return type is stringified
        assert sig.return_annotation in (None, "None")

    def test_all_exports(self) -> None:
        """__all__ exports ShotDetectionError and detect_shots."""
        from autopilot.analyze import scenes

        assert set(scenes.__all__) == {"ShotDetectionError", "detect_shots"}

    def test_module_constants(self) -> None:
        """Module constants are accessible with correct values."""
        from autopilot.analyze.scenes import (
            PYSCENEDETECT_THRESHOLD,
            TRANSNETV2_INPUT_HEIGHT,
            TRANSNETV2_INPUT_WIDTH,
        )

        assert TRANSNETV2_INPUT_WIDTH == 48
        assert TRANSNETV2_INPUT_HEIGHT == 27
        assert PYSCENEDETECT_THRESHOLD == 27.0


class TestIdempotency:
    """Tests for idempotency guard — skip when boundaries already exist."""

    def test_skips_when_boundaries_exist(self, catalog_db, tmp_path) -> None:
        """When boundaries already exist for media_id, detect_shots returns early."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")

        # Insert media + existing boundaries
        catalog_db.insert_media("vid1", str(video_file))
        catalog_db.upsert_boundaries("vid1", "[]", "transnetv2")

        scheduler = MagicMock()

        # Should return without calling scheduler
        detect_shots("vid1", video_file, catalog_db, scheduler)

        scheduler.model.assert_not_called()

    def test_proceeds_when_no_boundaries(self, catalog_db, tmp_path) -> None:
        """When no boundaries exist, detect_shots proceeds past the guard."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")

        catalog_db.insert_media("vid2", str(video_file))

        scheduler = MagicMock()

        # Should NOT return early — will raise because no real model/video
        # but the point is it gets past the idempotency check
        with pytest.raises(Exception):
            detect_shots("vid2", video_file, catalog_db, scheduler)


class TestInputValidation:
    """Tests for video path validation."""

    def test_nonexistent_path_raises(self, catalog_db, tmp_path) -> None:
        """Nonexistent video_path raises ShotDetectionError matching 'not found'."""
        from autopilot.analyze.scenes import ShotDetectionError, detect_shots

        bad_path = tmp_path / "does_not_exist.mp4"
        catalog_db.insert_media("vid3", str(bad_path))
        scheduler = MagicMock()

        with pytest.raises(ShotDetectionError, match="not found"):
            detect_shots("vid3", bad_path, catalog_db, scheduler)

    def test_validation_before_heavy_imports(self, catalog_db, tmp_path) -> None:
        """Validation happens before heavy imports — scheduler not called on bad path."""
        from autopilot.analyze.scenes import ShotDetectionError, detect_shots

        bad_path = tmp_path / "nope.mp4"
        catalog_db.insert_media("vid4", str(bad_path))
        scheduler = MagicMock()

        with pytest.raises(ShotDetectionError):
            detect_shots("vid4", bad_path, catalog_db, scheduler)

        scheduler.model.assert_not_called()


def _make_mock_cv2():
    """Create a MagicMock cv2 module with correct CAP_PROP constants."""
    mock_cv2 = MagicMock()
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.INTER_AREA = 3
    return mock_cv2


def _make_mock_capture(
    fps: float = 30.0,
    total_frames: int = 300,
    width: int = 1920,
    height: int = 1080,
) -> MagicMock:
    """Create a MagicMock mimicking cv2.VideoCapture for scenes tests."""
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

    # Default: read() returns frames of original resolution
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    return cap


class TestFrameReading:
    """Tests for _read_and_downsample_frames helper."""

    def test_returns_correct_shape(self) -> None:
        """Returns numpy array of shape (N, 27, 48, 3) for N-frame video."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=10, width=1920, height=1080)
        mock_cv2.VideoCapture.return_value = cap

        # Make cv2.resize return a 48x27 frame
        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        from autopilot.analyze.scenes import _read_and_downsample_frames

        frames, fps, total = _read_and_downsample_frames(
            Path("/fake/video.mp4"), mock_cv2
        )

        assert frames.shape == (10, 27, 48, 3)
        assert fps == 30.0
        assert total == 10

    def test_returns_fps_and_total_frames(self) -> None:
        """Returns correct fps float and total_frames int."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=24.0, total_frames=150)
        mock_cv2.VideoCapture.return_value = cap

        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        from autopilot.analyze.scenes import _read_and_downsample_frames

        frames, fps, total = _read_and_downsample_frames(
            Path("/fake/video.mp4"), mock_cv2
        )

        assert isinstance(fps, float)
        assert fps == 24.0
        assert isinstance(total, int)
        assert total == 150

    def test_skips_bad_frames(self) -> None:
        """Skips bad frames (cap.read returns False) without crashing."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=5)
        mock_cv2.VideoCapture.return_value = cap

        good_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Frames: good, bad, good, bad, good -> 3 good frames
        cap.read.side_effect = [
            (True, good_frame),
            (False, None),
            (True, good_frame),
            (False, None),
            (True, good_frame),
        ]

        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        from autopilot.analyze.scenes import _read_and_downsample_frames

        frames, fps, total = _read_and_downsample_frames(
            Path("/fake/video.mp4"), mock_cv2
        )

        assert frames.shape[0] == 3  # Only good frames
        assert total == 5  # Total is from metadata

    def test_downsamples_to_transnet_resolution(self) -> None:
        """Correctly downsamples from arbitrary resolution to 48x27 via cv2.resize."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=1, width=3840, height=2160)
        mock_cv2.VideoCapture.return_value = cap

        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        from autopilot.analyze.scenes import _read_and_downsample_frames

        frames, _, _ = _read_and_downsample_frames(
            Path("/fake/video.mp4"), mock_cv2
        )

        # cv2.resize should have been called with (48, 27) and INTER_AREA
        mock_cv2.resize.assert_called_once()
        call_args = mock_cv2.resize.call_args
        assert call_args[0][1] == (48, 27)  # (width, height) tuple

    def test_raises_when_video_cannot_open(self) -> None:
        """Raises ShotDetectionError when video cannot be opened."""
        from autopilot.analyze.scenes import (
            ShotDetectionError,
            _read_and_downsample_frames,
        )

        mock_cv2 = _make_mock_cv2()
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        with pytest.raises(ShotDetectionError, match="Failed to open"):
            _read_and_downsample_frames(Path("/fake/video.mp4"), mock_cv2)

    def test_cap_release_called_on_error(self) -> None:
        """cap.release() called even on error (finally block)."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=2)
        mock_cv2.VideoCapture.return_value = cap

        # Make read raise an error on second call
        good_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cap.read.side_effect = [
            (True, good_frame),
            RuntimeError("read error"),
        ]
        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        from autopilot.analyze.scenes import _read_and_downsample_frames

        with pytest.raises(RuntimeError, match="read error"):
            _read_and_downsample_frames(Path("/fake/video.mp4"), mock_cv2)

        cap.release.assert_called_once()


class TestTransNetV2BoundaryConversion:
    """Tests for _transnetv2_to_boundaries helper."""

    def test_converts_scenes_to_boundaries(self) -> None:
        """Converts TransNetV2 scenes array to boundary dicts."""
        from autopilot.analyze.scenes import _transnetv2_to_boundaries

        scenes = np.array([[0, 99], [100, 299]])
        result = _transnetv2_to_boundaries(scenes)

        assert len(result) == 2
        assert result[0] == {
            "start_frame": 0,
            "end_frame": 99,
            "transition_type": "cut",
        }
        assert result[1] == {
            "start_frame": 100,
            "end_frame": 299,
            "transition_type": "cut",
        }

    def test_single_scene(self) -> None:
        """Single-scene input returns one-element list."""
        from autopilot.analyze.scenes import _transnetv2_to_boundaries

        scenes = np.array([[0, 299]])
        result = _transnetv2_to_boundaries(scenes)

        assert len(result) == 1
        assert result[0]["start_frame"] == 0
        assert result[0]["end_frame"] == 299

    def test_empty_scenes(self) -> None:
        """Empty scenes array returns empty list."""
        from autopilot.analyze.scenes import _transnetv2_to_boundaries

        scenes = np.empty((0, 2), dtype=np.int64)
        result = _transnetv2_to_boundaries(scenes)

        assert result == []

    def test_output_values_are_plain_ints(self) -> None:
        """Output values are plain ints not numpy types (JSON-serializable)."""
        from autopilot.analyze.scenes import _transnetv2_to_boundaries

        scenes = np.array([[0, 99], [100, 299]])
        result = _transnetv2_to_boundaries(scenes)

        for boundary in result:
            assert type(boundary["start_frame"]) is int
            assert type(boundary["end_frame"]) is int
            # Verify JSON-serializable
            json.dumps(boundary)


def _make_mock_timecode(frame: int) -> MagicMock:
    """Create a MagicMock mimicking a PySceneDetect FrameTimecode."""
    tc = MagicMock()
    tc.get_frames.return_value = frame
    return tc


class TestPySceneDetectBoundaryConversion:
    """Tests for _pyscenedetect_to_boundaries helper."""

    def test_converts_scene_list_to_boundaries(self) -> None:
        """Converts PySceneDetect SceneList to boundary dicts using get_frames()."""
        from autopilot.analyze.scenes import _pyscenedetect_to_boundaries

        scenes = [
            (_make_mock_timecode(0), _make_mock_timecode(100)),
            (_make_mock_timecode(100), _make_mock_timecode(300)),
        ]
        result = _pyscenedetect_to_boundaries(scenes)

        assert len(result) == 2
        assert result[0] == {
            "start_frame": 0,
            "end_frame": 99,
            "transition_type": "cut",
        }
        assert result[1] == {
            "start_frame": 100,
            "end_frame": 299,
            "transition_type": "cut",
        }

    def test_empty_scene_list(self) -> None:
        """Empty scene list returns empty list."""
        from autopilot.analyze.scenes import _pyscenedetect_to_boundaries

        result = _pyscenedetect_to_boundaries([])
        assert result == []

    def test_single_scene(self) -> None:
        """Single scene converted correctly."""
        from autopilot.analyze.scenes import _pyscenedetect_to_boundaries

        scenes = [(_make_mock_timecode(0), _make_mock_timecode(500))]
        result = _pyscenedetect_to_boundaries(scenes)

        assert len(result) == 1
        assert result[0]["start_frame"] == 0
        assert result[0]["end_frame"] == 499

    def test_end_frame_is_get_frames_minus_one(self) -> None:
        """end_frame is end.get_frames()-1 since get_frames() returns next scene start."""
        from autopilot.analyze.scenes import _pyscenedetect_to_boundaries

        scenes = [(_make_mock_timecode(50), _make_mock_timecode(150))]
        result = _pyscenedetect_to_boundaries(scenes)

        # end.get_frames() returns 150, but the end frame of this scene is 149
        assert result[0]["end_frame"] == 149
