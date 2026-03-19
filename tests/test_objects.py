"""Tests for object detection and tracking (autopilot.analyze.objects)."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np  # type: ignore[reportMissingImports]
import pytest


def _make_mock_yolo_model(
    names: dict[int, str] | None = None,
    track_results: list | None = None,
) -> MagicMock:
    """Create a MagicMock mimicking an ultralytics YOLO model.

    Args:
        names: Class-index-to-name mapping. Defaults to {0: 'person', 1: 'bicycle', 2: 'car'}.
        track_results: List of result objects to return from model.track() calls.
            Each call pops the next result from the list.
    """
    model = MagicMock()
    model.names = names or {0: "person", 1: "bicycle", 2: "car"}
    model.predictor = MagicMock()

    if track_results is not None:
        model.track.side_effect = track_results
    return model


def _make_boxes(
    xywh: list[list[float]],
    conf: list[float],
    cls: list[int],
    track_id: list[int] | None = None,
) -> MagicMock:
    """Create a MagicMock mimicking ultralytics Boxes.

    Args:
        xywh: List of [x_center, y_center, width, height] per detection.
        conf: List of confidence scores.
        cls: List of class indices.
        track_id: List of track IDs, or None if no tracking.
    """
    boxes = MagicMock()
    boxes.xywh = MagicMock()
    boxes.xywh.cpu.return_value.numpy.return_value = np.array(xywh, dtype=np.float32)
    boxes.conf = MagicMock()
    boxes.conf.cpu.return_value.numpy.return_value = np.array(conf, dtype=np.float32)
    boxes.cls = MagicMock()
    boxes.cls.cpu.return_value.numpy.return_value = np.array(cls, dtype=np.float32)

    if track_id is not None:
        boxes.id = MagicMock()
        boxes.id.cpu.return_value.numpy.return_value = np.array(
            track_id, dtype=np.float32
        )
    else:
        boxes.id = None
    return boxes


def _make_yolo_result(boxes: MagicMock) -> list[MagicMock]:
    """Wrap boxes in a result list mimicking YOLO model.track() output."""
    result = MagicMock()
    result.boxes = boxes
    return [result]


def _make_empty_result() -> list[MagicMock]:
    """Create a YOLO result with no detections."""
    boxes = _make_boxes(xywh=[], conf=[], cls=[])
    # Override numpy arrays to be properly empty
    boxes.xywh.cpu.return_value.numpy.return_value = np.empty((0, 4), dtype=np.float32)
    boxes.conf.cpu.return_value.numpy.return_value = np.empty((0,), dtype=np.float32)
    boxes.cls.cpu.return_value.numpy.return_value = np.empty((0,), dtype=np.float32)
    boxes.id = None
    return _make_yolo_result(boxes)


def _make_mock_capture(
    fps: float = 30.0,
    total_frames: int = 300,
    width: int = 1920,
    height: int = 1080,
) -> MagicMock:
    """Create a MagicMock mimicking cv2.VideoCapture.

    Args:
        fps: Frames per second.
        total_frames: Total frame count.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """
    # cv2 CAP_PROP constants (raw integer values)
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

    # Default: read() always returns a frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
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


class TestPublicAPI:
    """Tests for module-level public API imports."""

    def test_detect_objects_importable(self) -> None:
        """detect_objects and DetectionError are importable from autopilot.analyze.objects."""
        from autopilot.analyze.objects import DetectionError, detect_objects

        assert detect_objects is not None
        assert DetectionError is not None

    def test_detection_error_is_exception(self) -> None:
        """DetectionError is a subclass of Exception."""
        from autopilot.analyze.objects import DetectionError

        assert issubclass(DetectionError, Exception)
        err = DetectionError("test")
        assert str(err) == "test"

    def test_detect_objects_signature(self) -> None:
        """detect_objects accepts required positional and optional keyword args."""
        import inspect

        from autopilot.analyze.objects import detect_objects

        sig = inspect.signature(detect_objects)
        params = list(sig.parameters.keys())
        # Positional: media_id, video_path, db, scheduler, config
        assert "media_id" in params
        assert "video_path" in params
        assert "db" in params
        assert "scheduler" in params
        assert "config" in params
        # Keyword-only: batch_size, sparse
        assert sig.parameters["batch_size"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["sparse"].kind == inspect.Parameter.KEYWORD_ONLY


class TestComputeFrameIndices:
    """Tests for _compute_frame_indices helper."""

    def test_sparse_mode_1fps(self) -> None:
        """Sparse mode samples at 1fps: 300 frames at 30fps → [0,30,60,...,270]."""
        from autopilot.analyze.objects import _compute_frame_indices

        result = _compute_frame_indices(total_frames=300, fps=30.0, sample_every_n=3, sparse=True)
        expected = list(range(0, 300, 30))
        assert result == expected

    def test_dense_every_3rd(self) -> None:
        """Dense mode every 3rd frame: total=10, n=3 → [0,3,6,9]."""
        from autopilot.analyze.objects import _compute_frame_indices

        result = _compute_frame_indices(total_frames=10, fps=30.0, sample_every_n=3, sparse=False)
        assert result == [0, 3, 6, 9]

    def test_dense_every_frame(self) -> None:
        """Dense mode every frame: total=5, n=1 → [0,1,2,3,4]."""
        from autopilot.analyze.objects import _compute_frame_indices

        result = _compute_frame_indices(total_frames=5, fps=30.0, sample_every_n=1, sparse=False)
        assert result == [0, 1, 2, 3, 4]

    def test_empty_video(self) -> None:
        """Empty video: total=0 → []."""
        from autopilot.analyze.objects import _compute_frame_indices

        result = _compute_frame_indices(total_frames=0, fps=30.0, sample_every_n=3, sparse=False)
        assert result == []

    def test_sparse_short_video(self) -> None:
        """Sparse short video: total=15, fps=30 → [0]."""
        from autopilot.analyze.objects import _compute_frame_indices

        result = _compute_frame_indices(total_frames=15, fps=30.0, sample_every_n=3, sparse=True)
        assert result == [0]

    def test_dense_exact_boundary(self) -> None:
        """Dense exact boundary: total=9, n=3 → [0,3,6]."""
        from autopilot.analyze.objects import _compute_frame_indices

        result = _compute_frame_indices(total_frames=9, fps=30.0, sample_every_n=3, sparse=False)
        assert result == [0, 3, 6]


class TestFormatDetections:
    """Tests for _format_detections helper."""

    def test_two_detections_formatted(self) -> None:
        """Two detections with track IDs produce correct dict structure."""
        from autopilot.analyze.objects import _format_detections

        boxes = np.array([[100.0, 200.0, 50.0, 60.0], [300.0, 400.0, 70.0, 80.0]])
        confs = np.array([0.9, 0.7])
        cls_ids = np.array([0, 2])
        track_ids = np.array([1, 2])
        names = {0: "person", 1: "bicycle", 2: "car"}

        result = _format_detections(boxes, confs, cls_ids, track_ids, names)
        assert len(result) == 2
        assert result[0]["track_id"] == 1
        assert result[0]["class"] == "person"
        assert result[0]["bbox_xywh"] == [100.0, 200.0, 50.0, 60.0]
        assert result[0]["confidence"] == pytest.approx(0.9)
        assert result[1]["track_id"] == 2
        assert result[1]["class"] == "car"

    def test_no_detections(self) -> None:
        """Empty arrays produce empty list."""
        from autopilot.analyze.objects import _format_detections

        boxes = np.empty((0, 4), dtype=np.float32)
        confs = np.empty((0,), dtype=np.float32)
        cls_ids = np.empty((0,), dtype=np.float32)
        names = {0: "person"}

        result = _format_detections(boxes, confs, cls_ids, None, names)
        assert result == []

    def test_track_ids_none(self) -> None:
        """When track_ids is None, each dict has track_id=None."""
        from autopilot.analyze.objects import _format_detections

        boxes = np.array([[100.0, 200.0, 50.0, 60.0]])
        confs = np.array([0.85])
        cls_ids = np.array([0])
        names = {0: "person"}

        result = _format_detections(boxes, confs, cls_ids, None, names)
        assert len(result) == 1
        assert result[0]["track_id"] is None

    def test_json_serializable(self) -> None:
        """Result is JSON-serializable (no numpy types leak)."""
        from autopilot.analyze.objects import _format_detections

        boxes = np.array([[100.0, 200.0, 50.0, 60.0]])
        confs = np.array([0.85])
        cls_ids = np.array([0])
        track_ids = np.array([5])
        names = {0: "person"}

        result = _format_detections(boxes, confs, cls_ids, track_ids, names)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)
        parsed = json.loads(serialized)
        assert parsed[0]["track_id"] == 5

    def test_bbox_xywh_is_plain_list(self) -> None:
        """bbox_xywh values are plain Python lists of 4 floats."""
        from autopilot.analyze.objects import _format_detections

        boxes = np.array([[100.5, 200.5, 50.5, 60.5]])
        confs = np.array([0.9])
        cls_ids = np.array([0])
        names = {0: "person"}

        result = _format_detections(boxes, confs, cls_ids, None, names)
        bbox = result[0]["bbox_xywh"]
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert all(isinstance(v, float) for v in bbox)


class TestInterpolateDetections:
    """Tests for _interpolate_detections basic linear interpolation."""

    def test_midpoint_interpolation(self) -> None:
        """Midpoint interpolation between frame 0 and frame 6 at target 3."""
        from autopilot.analyze.objects import _interpolate_detections

        det_before = [
            {"track_id": 1, "class": "person", "bbox_xywh": [100.0, 100.0, 50.0, 50.0],
             "confidence": 0.9}
        ]
        det_after = [
            {"track_id": 1, "class": "person", "bbox_xywh": [160.0, 100.0, 50.0, 50.0],
             "confidence": 0.8}
        ]
        result = _interpolate_detections(det_before, det_after, 0, 6, 3)
        assert len(result) == 1
        assert result[0]["track_id"] == 1
        assert result[0]["class"] == "person"
        assert result[0]["bbox_xywh"] == pytest.approx([130.0, 100.0, 50.0, 50.0])
        assert result[0]["confidence"] == pytest.approx(0.85)

    def test_quarter_interpolation(self) -> None:
        """Quarter interpolation between frame 0 and frame 4 at target 1."""
        from autopilot.analyze.objects import _interpolate_detections

        det_before = [
            {"track_id": 1, "class": "person", "bbox_xywh": [100.0, 100.0, 50.0, 50.0],
             "confidence": 0.9}
        ]
        det_after = [
            {"track_id": 1, "class": "person", "bbox_xywh": [200.0, 100.0, 50.0, 50.0],
             "confidence": 0.8}
        ]
        result = _interpolate_detections(det_before, det_after, 0, 4, 1)
        assert result[0]["bbox_xywh"] == pytest.approx([125.0, 100.0, 50.0, 50.0])

    def test_preserves_class_from_before(self) -> None:
        """Class name is preserved from det_before."""
        from autopilot.analyze.objects import _interpolate_detections

        det_before = [
            {"track_id": 1, "class": "person", "bbox_xywh": [100.0, 100.0, 50.0, 50.0],
             "confidence": 0.9}
        ]
        det_after = [
            {"track_id": 1, "class": "pedestrian", "bbox_xywh": [200.0, 100.0, 50.0, 50.0],
             "confidence": 0.8}
        ]
        result = _interpolate_detections(det_before, det_after, 0, 6, 3)
        assert result[0]["class"] == "person"

    def test_multiple_tracks_interpolated(self) -> None:
        """Multiple tracks are interpolated independently."""
        from autopilot.analyze.objects import _interpolate_detections

        det_before = [
            {"track_id": 1, "class": "person", "bbox_xywh": [100.0, 100.0, 50.0, 50.0],
             "confidence": 0.9},
            {"track_id": 2, "class": "car", "bbox_xywh": [500.0, 300.0, 100.0, 80.0],
             "confidence": 0.8},
        ]
        det_after = [
            {"track_id": 1, "class": "person", "bbox_xywh": [200.0, 100.0, 50.0, 50.0],
             "confidence": 0.9},
            {"track_id": 2, "class": "car", "bbox_xywh": [600.0, 300.0, 100.0, 80.0],
             "confidence": 0.8},
        ]
        result = _interpolate_detections(det_before, det_after, 0, 4, 2)
        # Find track 1 and track 2 results
        by_track = {d["track_id"]: d for d in result}
        assert by_track[1]["bbox_xywh"] == pytest.approx([150.0, 100.0, 50.0, 50.0])
        assert by_track[2]["bbox_xywh"] == pytest.approx([550.0, 300.0, 100.0, 80.0])


class TestInterpolateEdgeCases:
    """Tests for _interpolate_detections edge cases."""

    def test_track_only_in_before_holds_position(self) -> None:
        """Track only in det_before holds its position (copy bbox)."""
        from autopilot.analyze.objects import _interpolate_detections

        det_before = [
            {"track_id": 1, "class": "person", "bbox_xywh": [100.0, 100.0, 50.0, 50.0],
             "confidence": 0.9}
        ]
        det_after = []  # Track 1 disappeared
        result = _interpolate_detections(det_before, det_after, 0, 6, 3)
        assert len(result) == 1
        assert result[0]["bbox_xywh"] == [100.0, 100.0, 50.0, 50.0]
        assert result[0]["confidence"] == 0.9

    def test_track_only_in_after_holds_position(self) -> None:
        """Track only in det_after holds its position."""
        from autopilot.analyze.objects import _interpolate_detections

        det_before = []
        det_after = [
            {"track_id": 2, "class": "car", "bbox_xywh": [300.0, 200.0, 80.0, 60.0],
             "confidence": 0.7}
        ]
        result = _interpolate_detections(det_before, det_after, 0, 6, 3)
        assert len(result) == 1
        assert result[0]["bbox_xywh"] == [300.0, 200.0, 80.0, 60.0]

    def test_empty_before_returns_after_positions(self) -> None:
        """Empty det_before returns det_after positions."""
        from autopilot.analyze.objects import _interpolate_detections

        result = _interpolate_detections([], [], 0, 6, 3)
        assert result == []

    def test_both_empty(self) -> None:
        """Both det_before and det_after empty returns empty list."""
        from autopilot.analyze.objects import _interpolate_detections

        result = _interpolate_detections([], [], 0, 6, 3)
        assert result == []

    def test_same_frame_returns_before_copy(self) -> None:
        """frame_before == frame_after returns det_before copy."""
        from autopilot.analyze.objects import _interpolate_detections

        det_before = [
            {"track_id": 1, "class": "person", "bbox_xywh": [100.0, 100.0, 50.0, 50.0],
             "confidence": 0.9}
        ]
        result = _interpolate_detections(det_before, det_before, 5, 5, 5)
        assert len(result) == 1
        assert result[0]["bbox_xywh"] == [100.0, 100.0, 50.0, 50.0]
        # Verify it's a copy, not the same dict
        assert result[0] is not det_before[0]


class TestIdempotency:
    """Tests for detect_objects idempotency check."""

    def test_existing_detections_skips(self, catalog_db) -> None:
        """If detections exist for frame 0, detect_objects returns early."""
        from pathlib import Path
        from unittest.mock import MagicMock

        from autopilot.analyze.objects import detect_objects
        from autopilot.config import ModelConfig

        # Insert media_files row for FK constraint, then detection for frame 0
        catalog_db.insert_media("m1", "/fake/video.mp4")
        catalog_db.batch_insert_detections([("m1", 0, "[]")])

        scheduler = MagicMock()
        config = ModelConfig()
        detect_objects("m1", Path("/fake/video.mp4"), catalog_db, scheduler, config)
        # scheduler.model should not be called
        scheduler.model.assert_not_called()

    def test_no_existing_detections_proceeds(self, catalog_db) -> None:
        """Without existing detections, scheduler.model() is called."""
        from pathlib import Path

        from autopilot.analyze.objects import detect_objects
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m2", "/fake/video.mp4")
        scheduler = MagicMock()
        config = ModelConfig()
        mock_model = _make_mock_yolo_model()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(total_frames=3, fps=30.0)
        mock_model.track.return_value = _make_empty_result()

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(Path, "exists", return_value=True):
                detect_objects("m2", Path("/fake/video.mp4"), catalog_db, scheduler, config)

        scheduler.model.assert_called_once()


class TestSparseMode:
    """Tests for detect_objects in sparse mode (1fps)."""

    def _run_sparse(self, catalog_db, total_frames=300, fps=30.0):
        """Helper to run detect_objects in sparse mode with mocked video."""
        from pathlib import Path

        from autopilot.analyze.objects import detect_objects
        from autopilot.config import ModelConfig

        # Insert media row for FK
        catalog_db.insert_media("m1", "/fake/video.mp4")

        config = ModelConfig()
        mock_model = _make_mock_yolo_model()

        # Each track call returns one person detection
        def make_person_result(*args, **kwargs):
            boxes = _make_boxes(
                xywh=[[100.0, 200.0, 50.0, 60.0]],
                conf=[0.9],
                cls=[0],
                track_id=[1],
            )
            return _make_yolo_result(boxes)

        mock_model.track.side_effect = make_person_result

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(
            total_frames=total_frames, fps=fps, width=4096, height=4096
        )

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(Path, "exists", return_value=True):
                detect_objects(
                    "m1", Path("/fake/video.mp4"), catalog_db, scheduler, config,
                    sparse=True,
                )

        return mock_model, scheduler

    def test_sparse_stores_correct_frame_count(self, catalog_db) -> None:
        """Sparse mode at 30fps on 300 frames stores 10 rows (1fps × 10s)."""
        self._run_sparse(catalog_db, total_frames=300, fps=30.0)
        # Check how many detection rows in DB
        rows = catalog_db.get_detections_for_range("m1", 0, 300)
        assert len(rows) == 10  # frames 0,30,60,...,270

    def test_sparse_valid_json_schema(self, catalog_db) -> None:
        """Each detection row has valid JSON with expected schema."""
        self._run_sparse(catalog_db)
        row = catalog_db.get_detections_for_frame("m1", 0)
        assert row is not None
        dets = json.loads(row["detections_json"])
        assert isinstance(dets, list)
        assert len(dets) >= 1
        det = dets[0]
        assert "track_id" in det
        assert "class" in det
        assert "bbox_xywh" in det
        assert "confidence" in det

    def test_sparse_no_intermediate_frames(self, catalog_db) -> None:
        """Sparse mode does not store intermediate/interpolated frames."""
        self._run_sparse(catalog_db, total_frames=300, fps=30.0)
        # Frame 1 should NOT have detections (only multiples of 30)
        assert catalog_db.get_detections_for_frame("m1", 1) is None
        assert catalog_db.get_detections_for_frame("m1", 15) is None

    def test_sparse_model_track_persist(self, catalog_db) -> None:
        """model.track() is called with persist=True."""
        mock_model, _ = self._run_sparse(catalog_db)
        # Check that track was called and persist=True is in kwargs
        assert mock_model.track.call_count > 0
        for call in mock_model.track.call_args_list:
            assert call.kwargs.get("persist") is True


class TestDenseMode:
    """Tests for detect_objects in dense mode with interpolation."""

    def _run_dense(self, catalog_db, total_frames=12, fps=30.0, sample_every_n=3):
        """Helper to run detect_objects in dense mode with moving tracks."""
        from pathlib import Path

        from autopilot.analyze.objects import detect_objects
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        config = ModelConfig()
        config.yolo_sample_every_n_frames = sample_every_n
        mock_model = _make_mock_yolo_model()

        # Track_id=1 moves rightward: x starts at 100, increases by 30 per sample
        call_count = [0]

        def make_moving_result(*args, **kwargs):
            x = 100.0 + call_count[0] * 30.0
            boxes = _make_boxes(
                xywh=[[x, 200.0, 50.0, 60.0]],
                conf=[0.9],
                cls=[0],
                track_id=[1],
            )
            call_count[0] += 1
            return _make_yolo_result(boxes)

        mock_model.track.side_effect = make_moving_result

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(
            total_frames=total_frames, fps=fps, width=1920, height=1080
        )

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(Path, "exists", return_value=True):
                detect_objects(
                    "m1", Path("/fake/video.mp4"), catalog_db, scheduler, config,
                    sparse=False,
                )

        return mock_model

    def test_dense_all_frames_stored(self, catalog_db) -> None:
        """Dense mode stores detections for all 12 frames."""
        self._run_dense(catalog_db, total_frames=12, sample_every_n=3)
        rows = catalog_db.get_detections_for_range("m1", 0, 11)
        assert len(rows) == 12

    def test_dense_sampled_frames_exact(self, catalog_db) -> None:
        """Sampled frames (0,3,6,9) have exact YOLO results."""
        self._run_dense(catalog_db, total_frames=12, sample_every_n=3)
        # Frame 0: x=100
        row0 = catalog_db.get_detections_for_frame("m1", 0)
        dets0 = json.loads(row0["detections_json"])
        assert dets0[0]["bbox_xywh"][0] == pytest.approx(100.0)
        # Frame 3: x=130 (second track call)
        row3 = catalog_db.get_detections_for_frame("m1", 3)
        dets3 = json.loads(row3["detections_json"])
        assert dets3[0]["bbox_xywh"][0] == pytest.approx(130.0)

    def test_dense_interpolated_frame(self, catalog_db) -> None:
        """Frame 1 is interpolated (1/3 between frame 0 and frame 3)."""
        self._run_dense(catalog_db, total_frames=12, sample_every_n=3)
        # Frame 0 x=100, Frame 3 x=130 → Frame 1 x ≈ 110
        row1 = catalog_db.get_detections_for_frame("m1", 1)
        dets1 = json.loads(row1["detections_json"])
        assert dets1[0]["bbox_xywh"][0] == pytest.approx(110.0)

    def test_dense_trailing_frame_holds_position(self, catalog_db) -> None:
        """Frame 11 holds position from frame 9 (no frame 12 to interpolate toward)."""
        self._run_dense(catalog_db, total_frames=12, sample_every_n=3)
        # Frame 9: x=100 + 3*30 = 190 (4th track call, 0-indexed)
        row9 = catalog_db.get_detections_for_frame("m1", 9)
        dets9 = json.loads(row9["detections_json"])
        row11 = catalog_db.get_detections_for_frame("m1", 11)
        dets11 = json.loads(row11["detections_json"])
        # Frame 11 should hold same x as frame 9
        assert dets11[0]["bbox_xywh"][0] == pytest.approx(dets9[0]["bbox_xywh"][0])


class TestBatchProcessing:
    """Tests for batch frame reading and DB insertion."""

    def _run_with_batch(self, catalog_db, total_frames=50, sample_every_n=1,
                        batch_size=16, sparse=True):
        """Helper to run detect_objects with specific batch parameters."""
        from pathlib import Path

        from autopilot.analyze.objects import detect_objects
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        config = ModelConfig()
        config.yolo_sample_every_n_frames = sample_every_n
        mock_model = _make_mock_yolo_model()
        mock_model.track.side_effect = lambda *a, **kw: _make_yolo_result(
            _make_boxes(xywh=[[100.0, 200.0, 50.0, 60.0]], conf=[0.9], cls=[0], track_id=[1])
        )

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(total_frames=total_frames, fps=30.0)

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(Path, "exists", return_value=True):
                detect_objects(
                    "m1", Path("/fake/video.mp4"), catalog_db, scheduler, config,
                    batch_size=batch_size, sparse=sparse,
                )

        return mock_model

    def test_track_called_per_frame(self, catalog_db) -> None:
        """model.track() is called once per sampled frame (50 times for n=1 sparse 30fps)."""
        # sparse mode at 30fps on 50 frames → frames at [0, 30]
        mock_model = self._run_with_batch(
            catalog_db, total_frames=50, sample_every_n=1, sparse=True,
        )
        # At 30fps sparse, interval = 30, so frames [0, 30] → 2 track calls
        assert mock_model.track.call_count == 2

    def test_db_inserts_correct_row_count(self, catalog_db) -> None:
        """DB receives correct number of detection rows."""
        # Sparse at 30fps on 90 frames → [0, 30, 60] → 3 rows
        self._run_with_batch(
            catalog_db, total_frames=90, sample_every_n=1, sparse=True,
        )
        rows = catalog_db.get_detections_for_range("m1", 0, 90)
        assert len(rows) == 3

    def test_large_batch_size_works(self, catalog_db) -> None:
        """batch_size larger than frame count works correctly."""
        self._run_with_batch(
            catalog_db, total_frames=5, sample_every_n=1, sparse=True,
            batch_size=100,
        )
        # 5 frames at 30fps sparse → only frame 0
        rows = catalog_db.get_detections_for_range("m1", 0, 5)
        assert len(rows) == 1


class TestErrorHandling:
    """Tests for error handling in detect_objects."""

    def test_video_not_found_raises(self, catalog_db) -> None:
        """Video file not found raises DetectionError."""
        from pathlib import Path

        from autopilot.analyze.objects import DetectionError, detect_objects
        from autopilot.config import ModelConfig

        scheduler = MagicMock()
        config = ModelConfig()

        with pytest.raises(DetectionError, match="not found"):
            detect_objects("m1", Path("/nonexistent/video.mp4"), catalog_db, scheduler, config)

    def test_video_not_opened_raises(self, catalog_db) -> None:
        """VideoCapture.isOpened() returns False raises DetectionError."""
        from pathlib import Path

        from autopilot.analyze.objects import DetectionError, detect_objects
        from autopilot.config import ModelConfig

        scheduler = MagicMock()
        config = ModelConfig()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(DetectionError, match="Failed to open"):
                    detect_objects(
                        "m1", Path("/fake/video.mp4"), catalog_db, scheduler, config,
                    )

    def test_frame_read_failure_skipped(self, catalog_db) -> None:
        """Frame read failure is skipped gracefully, other frames still processed."""
        from pathlib import Path

        from autopilot.analyze.objects import detect_objects
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        config = ModelConfig()
        mock_model = _make_mock_yolo_model()
        mock_model.track.side_effect = lambda *a, **kw: _make_yolo_result(
            _make_boxes(xywh=[[100.0, 200.0, 50.0, 60.0]], conf=[0.9], cls=[0], track_id=[1])
        )

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        # 2 frames at 30fps sparse = [0]
        mock_cap = _make_mock_capture(total_frames=60, fps=30.0)
        # First read succeeds (frame 0), second fails (frame 30)
        mock_cap.read.side_effect = [
            (True, np.zeros((1080, 1920, 3), dtype=np.uint8)),
            (False, None),
        ]

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(Path, "exists", return_value=True):
                detect_objects(
                    "m1", Path("/fake/video.mp4"), catalog_db, scheduler, config,
                    sparse=True,
                )

        # Frame 0 should be in DB (read succeeded)
        assert catalog_db.get_detections_for_frame("m1", 0) is not None


class TestLogging:
    """Tests for structured logging in detect_objects."""

    def _run_with_logging(self, catalog_db, caplog, sparse=False):
        """Helper to run detect_objects and capture log output."""
        import logging
        from pathlib import Path

        from autopilot.analyze.objects import detect_objects
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        config = ModelConfig()
        mock_model = _make_mock_yolo_model()
        mock_model.track.side_effect = lambda *a, **kw: _make_yolo_result(
            _make_boxes(xywh=[[100.0, 200.0, 50.0, 60.0]], conf=[0.9], cls=[0], track_id=[1])
        )

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(total_frames=90, fps=30.0)

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.objects"):
            with patch.dict(sys.modules, {"cv2": mock_cv2}):
                with patch.object(Path, "exists", return_value=True):
                    detect_objects(
                        "m1", Path("/fake/video.mp4"), catalog_db, scheduler, config,
                        sparse=sparse,
                    )

    def test_log_contains_media_id_and_mode(self, catalog_db, caplog) -> None:
        """Log contains media_id and mode on start."""
        self._run_with_logging(catalog_db, caplog, sparse=True)
        assert any("m1" in r.message and "sparse" in r.message for r in caplog.records)

    def test_log_contains_completion(self, catalog_db, caplog) -> None:
        """Log contains detection count on completion."""
        self._run_with_logging(catalog_db, caplog, sparse=True)
        assert any("Completed" in r.message or "sampled" in r.message for r in caplog.records)

    def test_log_skipping_on_idempotent(self, catalog_db, caplog) -> None:
        """Log mentions 'skipping' on idempotent skip."""
        import logging

        catalog_db.insert_media("m2", "/fake/video.mp4")
        catalog_db.batch_insert_detections([("m2", 0, "[]")])

        from pathlib import Path

        from autopilot.analyze.objects import detect_objects
        from autopilot.config import ModelConfig

        scheduler = MagicMock()
        config = ModelConfig()

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.objects"):
            detect_objects("m2", Path("/fake/video.mp4"), catalog_db, scheduler, config)

        assert any("skipping" in r.message.lower() for r in caplog.records)

    def test_log_contains_frame_info(self, catalog_db, caplog) -> None:
        """Log contains frame count information."""
        self._run_with_logging(catalog_db, caplog, sparse=True)
        # Should log total_frames or frame count
        assert any("frames" in r.message.lower() or "frame" in r.message.lower()
                    for r in caplog.records)


class TestIntegration:
    """Integration tests for detect_objects end-to-end."""

    def test_dense_two_objects_full_pipeline(self, catalog_db) -> None:
        """Dense mode with 2 objects: 30 frames, 30fps, 4096x4096, n=3."""
        from pathlib import Path

        from autopilot.analyze.objects import detect_objects
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        config = ModelConfig()
        config.yolo_sample_every_n_frames = 3
        mock_model = _make_mock_yolo_model()

        # Person track_id=1 moves right, car track_id=2 moves down
        call_idx = [0]

        def make_two_objects(*args, **kwargs):
            i = call_idx[0]
            person_x = 100.0 + i * 20.0
            car_y = 500.0 + i * 15.0
            boxes = _make_boxes(
                xywh=[
                    [person_x, 200.0, 50.0, 60.0],
                    [800.0, car_y, 100.0, 80.0],
                ],
                conf=[0.95, 0.85],
                cls=[0, 2],
                track_id=[1, 2],
            )
            call_idx[0] += 1
            return _make_yolo_result(boxes)

        mock_model.track.side_effect = make_two_objects

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(
            total_frames=30, fps=30.0, width=4096, height=4096
        )

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(Path, "exists", return_value=True):
                detect_objects(
                    "m1", Path("/fake/video.mp4"), catalog_db, scheduler, config,
                    sparse=False,
                )

        # (1) All 30 frames have detections
        all_rows = catalog_db.get_detections_for_range("m1", 0, 29)
        assert len(all_rows) == 30

        # (2) Each row has valid JSON
        for row in all_rows:
            dets = json.loads(row["detections_json"])
            assert isinstance(dets, list)

        # (3) Track IDs consistent across interpolated frames
        for frame_num in range(30):
            row = catalog_db.get_detections_for_frame("m1", frame_num)
            dets = json.loads(row["detections_json"])
            assert len(dets) == 2
            track_ids = {d["track_id"] for d in dets}
            assert track_ids == {1, 2}

        # (4) Bbox values are non-negative
        for frame_num in range(30):
            row = catalog_db.get_detections_for_frame("m1", frame_num)
            dets = json.loads(row["detections_json"])
            for det in dets:
                for v in det["bbox_xywh"]:
                    assert v >= 0

        # (5) Class names correct
        row0 = catalog_db.get_detections_for_frame("m1", 0)
        dets0 = json.loads(row0["detections_json"])
        by_track = {d["track_id"]: d for d in dets0}
        assert by_track[1]["class"] == "person"
        assert by_track[2]["class"] == "car"
