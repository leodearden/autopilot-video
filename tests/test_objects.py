"""Tests for object detection and tracking (autopilot.analyze.objects)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, PropertyMock

import numpy as np
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
    cap = MagicMock()
    cap.isOpened.return_value = True

    def get_prop(prop_id):
        import cv2

        prop_map = {
            cv2.CAP_PROP_FPS: fps,
            cv2.CAP_PROP_FRAME_COUNT: total_frames,
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
        }
        return prop_map.get(prop_id, 0.0)

    cap.get.side_effect = get_prop

    # Default: read() always returns a frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    cap.set.return_value = True
    return cap


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
        from autopilot.analyze.objects import detect_objects
        import inspect

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
