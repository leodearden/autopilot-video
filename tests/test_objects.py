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
