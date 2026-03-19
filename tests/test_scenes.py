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


def _make_mock_transnetv2_model(scenes: np.ndarray | None = None) -> MagicMock:
    """Create a MagicMock TransNetV2 model.

    Args:
        scenes: Optional scenes array to return from predictions_to_scenes.
            Defaults to [[0, 99], [100, 299]].
    """
    model = MagicMock()
    predictions = np.random.rand(300, 1)
    model.predict_frames.return_value = (predictions, None)
    if scenes is None:
        scenes = np.array([[0, 99], [100, 299]])
    model.predictions_to_scenes.return_value = scenes
    return model


def _make_mock_scheduler(model: MagicMock) -> MagicMock:
    """Create a MagicMock scheduler with context manager protocol."""
    scheduler = MagicMock()
    scheduler.model.return_value.__enter__ = MagicMock(return_value=model)
    scheduler.model.return_value.__exit__ = MagicMock(return_value=False)
    return scheduler


class TestTransNetV2Pipeline:
    """Tests for the TransNetV2 detection path in detect_shots."""

    def _setup(self, catalog_db, tmp_path):
        """Common setup: create video file, insert media, mock cv2 + model."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-t2v", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=300)
        mock_cv2.VideoCapture.return_value = cap
        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        return video_file, mock_cv2

    def test_scheduler_model_called(self, catalog_db, tmp_path) -> None:
        """scheduler.model('transnetv2') is called as context manager."""
        from autopilot.analyze.scenes import detect_shots

        video_file, mock_cv2 = self._setup(catalog_db, tmp_path)
        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_shots("vid-t2v", video_file, catalog_db, scheduler)

        scheduler.model.assert_called_once_with("transnetv2")

    def test_model_receives_frames(self, catalog_db, tmp_path) -> None:
        """Model receives downsampled frames array."""
        from autopilot.analyze.scenes import detect_shots

        video_file, mock_cv2 = self._setup(catalog_db, tmp_path)
        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_shots("vid-t2v", video_file, catalog_db, scheduler)

        model.predict_frames.assert_called_once()
        frames_arg = model.predict_frames.call_args[0][0]
        assert isinstance(frames_arg, np.ndarray)
        assert frames_arg.shape[1:] == (27, 48, 3)

    def test_predictions_to_scenes_called(self, catalog_db, tmp_path) -> None:
        """model.predict_frames then model.predictions_to_scenes are called."""
        from autopilot.analyze.scenes import detect_shots

        video_file, mock_cv2 = self._setup(catalog_db, tmp_path)
        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_shots("vid-t2v", video_file, catalog_db, scheduler)

        model.predict_frames.assert_called_once()
        model.predictions_to_scenes.assert_called_once()

    def test_boundaries_stored_in_db(self, catalog_db, tmp_path) -> None:
        """Boundaries stored via db.upsert_boundaries with method='transnetv2'."""
        from autopilot.analyze.scenes import detect_shots

        video_file, mock_cv2 = self._setup(catalog_db, tmp_path)
        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_shots("vid-t2v", video_file, catalog_db, scheduler)

        row = catalog_db.get_boundaries("vid-t2v", method="transnetv2")
        assert row is not None
        boundaries = json.loads(row["boundaries_json"])
        assert isinstance(boundaries, list)
        assert len(boundaries) == 2

    def test_stored_boundaries_json_structure(self, catalog_db, tmp_path) -> None:
        """Stored boundaries_json is valid JSON list of dicts with correct keys."""
        from autopilot.analyze.scenes import detect_shots

        video_file, mock_cv2 = self._setup(catalog_db, tmp_path)
        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_shots("vid-t2v", video_file, catalog_db, scheduler)

        row = catalog_db.get_boundaries("vid-t2v", method="transnetv2")
        boundaries = json.loads(row["boundaries_json"])
        for b in boundaries:
            assert "start_frame" in b
            assert "end_frame" in b
            assert "transition_type" in b
            assert b["transition_type"] == "cut"

    def test_zero_frame_video(self, catalog_db, tmp_path) -> None:
        """Handles zero-frame video gracefully (stores empty list)."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "empty.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-empty", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=0)
        mock_cv2.VideoCapture.return_value = cap

        model = _make_mock_transnetv2_model(scenes=np.empty((0, 2), dtype=np.int64))
        scheduler = _make_mock_scheduler(model)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_shots("vid-empty", video_file, catalog_db, scheduler)

        row = catalog_db.get_boundaries("vid-empty", method="transnetv2")
        assert row is not None
        boundaries = json.loads(row["boundaries_json"])
        assert boundaries == []


def _make_mock_scenedetect(scenes=None):
    """Create mock scenedetect module and detectors submodule.

    Args:
        scenes: Optional scene list to return from detect(). Defaults to 2 scenes.
    """
    mock_sd = MagicMock()
    mock_detectors = MagicMock()

    if scenes is None:
        scenes = [
            (_make_mock_timecode(0), _make_mock_timecode(100)),
            (_make_mock_timecode(100), _make_mock_timecode(300)),
        ]
    mock_sd.detect.return_value = scenes

    return mock_sd, mock_detectors


class TestPySceneDetectPipeline:
    """Tests for the PySceneDetect detection path."""

    def test_scenedetect_detect_called(self, catalog_db, tmp_path) -> None:
        """scenedetect.detect called with str(video_path) and AdaptiveDetector."""
        from autopilot.analyze.scenes import _run_pyscenedetect

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-psd", str(video_file))

        mock_sd, mock_detectors = _make_mock_scenedetect()

        with patch.dict(sys.modules, {
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            _run_pyscenedetect("vid-psd", video_file, catalog_db)

        mock_sd.detect.assert_called_once()
        call_args = mock_sd.detect.call_args
        assert call_args[0][0] == str(video_file)

    def test_adaptive_detector_threshold(self, catalog_db, tmp_path) -> None:
        """AdaptiveDetector created with adaptive_threshold=27.0."""
        from autopilot.analyze.scenes import _run_pyscenedetect

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-psd2", str(video_file))

        mock_sd, mock_detectors = _make_mock_scenedetect()

        with patch.dict(sys.modules, {
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            _run_pyscenedetect("vid-psd2", video_file, catalog_db)

        mock_detectors.AdaptiveDetector.assert_called_once_with(
            adaptive_threshold=27.0
        )

    def test_boundaries_stored_with_pyscenedetect_method(self, catalog_db, tmp_path) -> None:
        """Stores results via db.upsert_boundaries with method='pyscenedetect'."""
        from autopilot.analyze.scenes import _run_pyscenedetect

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-psd3", str(video_file))

        mock_sd, mock_detectors = _make_mock_scenedetect()

        with patch.dict(sys.modules, {
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            _run_pyscenedetect("vid-psd3", video_file, catalog_db)

        row = catalog_db.get_boundaries("vid-psd3", method="pyscenedetect")
        assert row is not None
        boundaries = json.loads(row["boundaries_json"])
        assert isinstance(boundaries, list)

    def test_stored_boundaries_are_valid_json(self, catalog_db, tmp_path) -> None:
        """Stored boundaries_json is valid JSON."""
        from autopilot.analyze.scenes import _run_pyscenedetect

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-psd4", str(video_file))

        mock_sd, mock_detectors = _make_mock_scenedetect()

        with patch.dict(sys.modules, {
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            _run_pyscenedetect("vid-psd4", video_file, catalog_db)

        row = catalog_db.get_boundaries("vid-psd4", method="pyscenedetect")
        boundaries = json.loads(row["boundaries_json"])
        assert len(boundaries) == 2
        for b in boundaries:
            assert "start_frame" in b
            assert "end_frame" in b
            assert "transition_type" in b


class TestFallbackBehavior:
    """Tests for TransNetV2 -> PySceneDetect fallback logic."""

    def test_transnetv2_success_skips_pyscenedetect(self, catalog_db, tmp_path) -> None:
        """When TransNetV2 succeeds, PySceneDetect is NOT called."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-fb1", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=300)
        mock_cv2.VideoCapture.return_value = cap
        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        mock_sd, mock_detectors = _make_mock_scenedetect()

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            detect_shots("vid-fb1", video_file, catalog_db, scheduler)

        # scenedetect.detect should not have been called
        mock_sd.detect.assert_not_called()

    def test_transnetv2_failure_runs_fallback(self, catalog_db, tmp_path) -> None:
        """When TransNetV2 raises exception, PySceneDetect fallback runs."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-fb2", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=300)
        mock_cv2.VideoCapture.return_value = cap
        # Make frame reading fail
        cap.isOpened.return_value = False

        scheduler = MagicMock()
        mock_sd, mock_detectors = _make_mock_scenedetect()

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            detect_shots("vid-fb2", video_file, catalog_db, scheduler)

        # Fallback should have stored with pyscenedetect method
        row = catalog_db.get_boundaries("vid-fb2", method="pyscenedetect")
        assert row is not None

    def test_double_failure_raises(self, catalog_db, tmp_path) -> None:
        """When both methods fail, ShotDetectionError is raised."""
        from autopilot.analyze.scenes import ShotDetectionError, detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-fb3", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=300)
        mock_cv2.VideoCapture.return_value = cap
        cap.isOpened.return_value = False

        scheduler = MagicMock()

        # Make scenedetect fail too
        mock_sd = MagicMock()
        mock_sd.detect.side_effect = RuntimeError("scenedetect failed")
        mock_detectors = MagicMock()

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            with pytest.raises(ShotDetectionError):
                detect_shots("vid-fb3", video_file, catalog_db, scheduler)

    def test_double_failure_wraps_cause(self, catalog_db, tmp_path) -> None:
        """Double failure wraps fallback exception via __cause__."""
        from autopilot.analyze.scenes import ShotDetectionError, detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-fb4", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture()
        mock_cv2.VideoCapture.return_value = cap
        cap.isOpened.return_value = False

        scheduler = MagicMock()
        mock_sd = MagicMock()
        fallback_err = RuntimeError("pyscene broke")
        mock_sd.detect.side_effect = fallback_err
        mock_detectors = MagicMock()

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            with pytest.raises(ShotDetectionError) as exc_info:
                detect_shots("vid-fb4", video_file, catalog_db, scheduler)

        assert exc_info.value.__cause__ is fallback_err

    def test_no_partial_results_on_double_failure(self, catalog_db, tmp_path) -> None:
        """No partial results stored in DB on double failure."""
        from autopilot.analyze.scenes import ShotDetectionError, detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-fb5", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture()
        mock_cv2.VideoCapture.return_value = cap
        cap.isOpened.return_value = False

        scheduler = MagicMock()
        mock_sd = MagicMock()
        mock_sd.detect.side_effect = RuntimeError("fail")
        mock_detectors = MagicMock()

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            with pytest.raises(ShotDetectionError):
                detect_shots("vid-fb5", video_file, catalog_db, scheduler)

        # No boundaries should exist
        assert catalog_db.get_boundaries("vid-fb5") == []


class TestLogging:
    """Tests for structured log output."""

    def test_info_on_start(self, catalog_db, tmp_path, caplog) -> None:
        """INFO log on detection start with media_id."""
        import logging

        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-log1", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=300)
        mock_cv2.VideoCapture.return_value = cap
        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.scenes"):
            with patch.dict(sys.modules, {"cv2": mock_cv2}):
                detect_shots("vid-log1", video_file, catalog_db, scheduler)

        assert any("vid-log1" in r.message and "Starting" in r.message for r in caplog.records)

    def test_info_on_completion(self, catalog_db, tmp_path, caplog) -> None:
        """INFO log on successful completion with method name and boundary count."""
        import logging

        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-log2", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=300)
        mock_cv2.VideoCapture.return_value = cap
        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.scenes"):
            with patch.dict(sys.modules, {"cv2": mock_cv2}):
                detect_shots("vid-log2", video_file, catalog_db, scheduler)

        assert any(
            "vid-log2" in r.message and "Completed" in r.message and "transnetv2" in r.message
            for r in caplog.records
        )

    def test_info_on_idempotent_skip(self, catalog_db, tmp_path, caplog) -> None:
        """INFO log mentioning 'skipping' for idempotent call."""
        import logging

        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-log3", str(video_file))
        catalog_db.upsert_boundaries("vid-log3", "[]", "transnetv2")

        scheduler = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.scenes"):
            detect_shots("vid-log3", video_file, catalog_db, scheduler)

        assert any("skipping" in r.message.lower() for r in caplog.records)

    def test_warning_on_transnetv2_failure(self, catalog_db, tmp_path, caplog) -> None:
        """WARNING log containing media_id when TransNetV2 fails and fallback triggers."""
        import logging

        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("vid-log4", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture()
        mock_cv2.VideoCapture.return_value = cap
        cap.isOpened.return_value = False

        scheduler = MagicMock()
        mock_sd, mock_detectors = _make_mock_scenedetect()

        with caplog.at_level(logging.WARNING, logger="autopilot.analyze.scenes"):
            with patch.dict(sys.modules, {
                "cv2": mock_cv2,
                "scenedetect": mock_sd,
                "scenedetect.detectors": mock_detectors,
            }):
                detect_shots("vid-log4", video_file, catalog_db, scheduler)

        assert any(
            "vid-log4" in r.message and r.levelno == logging.WARNING
            for r in caplog.records
        )


class TestIntegration:
    """End-to-end integration tests."""

    def test_transnetv2_e2e(self, catalog_db, tmp_path) -> None:
        """Full TransNetV2 path: mocked video -> boundaries in DB."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("int-1", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=300)
        mock_cv2.VideoCapture.return_value = cap
        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        scenes = np.array([[0, 99], [100, 199], [200, 299]])
        model = _make_mock_transnetv2_model(scenes=scenes)
        scheduler = _make_mock_scheduler(model)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_shots("int-1", video_file, catalog_db, scheduler)

        row = catalog_db.get_boundaries("int-1", method="transnetv2")
        assert row is not None
        boundaries = json.loads(row["boundaries_json"])
        assert len(boundaries) == 3
        assert boundaries[0]["start_frame"] == 0
        assert boundaries[0]["end_frame"] == 99
        assert boundaries[0]["transition_type"] == "cut"
        assert boundaries[2]["start_frame"] == 200
        assert boundaries[2]["end_frame"] == 299

    def test_idempotent_second_call(self, catalog_db, tmp_path) -> None:
        """Second call to same media_id is idempotent (scheduler not called again)."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("int-2", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=300)
        mock_cv2.VideoCapture.return_value = cap
        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        # First call
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_shots("int-2", video_file, catalog_db, scheduler)

        # Reset mock to track second call
        scheduler.model.reset_mock()

        # Second call should be idempotent
        detect_shots("int-2", video_file, catalog_db, scheduler)
        scheduler.model.assert_not_called()

    def test_fallback_stores_pyscenedetect(self, catalog_db, tmp_path) -> None:
        """Fallback path stores boundaries with method='pyscenedetect'."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("int-3", str(video_file))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture()
        mock_cv2.VideoCapture.return_value = cap
        cap.isOpened.return_value = False

        scheduler = MagicMock()
        mock_sd, mock_detectors = _make_mock_scenedetect()

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            detect_shots("int-3", video_file, catalog_db, scheduler)

        row = catalog_db.get_boundaries("int-3", method="pyscenedetect")
        assert row is not None
        boundaries = json.loads(row["boundaries_json"])
        assert isinstance(boundaries, list)
        assert len(boundaries) > 0

    def test_both_methods_coexist_in_db(self, catalog_db, tmp_path) -> None:
        """Both methods' results can coexist in DB for same media_id."""
        from autopilot.analyze.scenes import _run_pyscenedetect, detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")
        catalog_db.insert_media("int-4", str(video_file))

        # First: run TransNetV2
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=300)
        mock_cv2.VideoCapture.return_value = cap
        resized = np.zeros((27, 48, 3), dtype=np.uint8)
        mock_cv2.resize.return_value = resized

        model = _make_mock_transnetv2_model()
        scheduler = _make_mock_scheduler(model)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_shots("int-4", video_file, catalog_db, scheduler)

        # Verify transnetv2 result exists
        t2v_row = catalog_db.get_boundaries("int-4", method="transnetv2")
        assert t2v_row is not None

        # Now directly run PySceneDetect (bypassing idempotency)
        mock_sd, mock_detectors = _make_mock_scenedetect()
        with patch.dict(sys.modules, {
            "scenedetect": mock_sd,
            "scenedetect.detectors": mock_detectors,
        }):
            _run_pyscenedetect("int-4", video_file, catalog_db)

        # Both should exist
        all_boundaries = catalog_db.get_boundaries("int-4")
        assert len(all_boundaries) == 2

        methods = {row["method"] for row in all_boundaries}
        assert methods == {"transnetv2", "pyscenedetect"}
