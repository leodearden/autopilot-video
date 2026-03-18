"""Tests for face detection and clustering (autopilot.analyze.faces)."""

from __future__ import annotations

import inspect
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# -- Mock helpers --------------------------------------------------------------


def _make_mock_capture(
    fps: float = 30.0,
    total_frames: int = 300,
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


def _make_mock_face(
    bbox: list[float] | None = None,
    det_score: float = 0.95,
    embedding: np.ndarray | None = None,
) -> MagicMock:
    """Create a MagicMock mimicking an InsightFace Face object."""
    face = MagicMock()
    face.bbox = np.array(bbox or [10.0, 20.0, 100.0, 200.0], dtype=np.float32)
    face.det_score = det_score
    if embedding is None:
        embedding = np.random.default_rng(42).random(512).astype(np.float32)
    face.normed_embedding = embedding
    return face


# -- Public API tests ----------------------------------------------------------


class TestPublicAPI:
    """Tests for module-level public API imports."""

    def test_face_detection_error_importable(self) -> None:
        """FaceDetectionError is importable from autopilot.analyze.faces."""
        from autopilot.analyze.faces import FaceDetectionError

        assert FaceDetectionError is not None

    def test_face_detection_error_is_exception(self) -> None:
        """FaceDetectionError is a subclass of Exception."""
        from autopilot.analyze.faces import FaceDetectionError

        assert issubclass(FaceDetectionError, Exception)
        err = FaceDetectionError("test")
        assert str(err) == "test"

    def test_detect_faces_importable(self) -> None:
        """detect_faces is importable with correct positional params."""
        from autopilot.analyze.faces import detect_faces

        sig = inspect.signature(detect_faces)
        params = list(sig.parameters.keys())
        assert "media_id" in params
        assert "video_path" in params
        assert "db" in params
        assert "scheduler" in params
        assert "config" in params

    def test_cluster_faces_importable(self) -> None:
        """cluster_faces is importable with signature (db, *, eps, min_samples)."""
        from autopilot.analyze.faces import cluster_faces

        sig = inspect.signature(cluster_faces)
        params = list(sig.parameters.keys())
        assert "db" in params
        assert "eps" in params
        assert "min_samples" in params
        assert sig.parameters["eps"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["min_samples"].kind == inspect.Parameter.KEYWORD_ONLY


# -- Idempotency tests --------------------------------------------------------


class TestIdempotency:
    """Tests for detect_faces idempotency check."""

    def test_existing_faces_skips(self, catalog_db) -> None:
        """When faces already exist for a media_id, detect_faces skips."""
        from autopilot.analyze.faces import detect_faces

        # Insert media and face rows
        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        catalog_db.batch_insert_faces([
            ("vid1", 0, 0, '[10,20,100,200]', b'\x00' * 2048, None),
        ])

        scheduler = MagicMock()
        config = MagicMock()
        config.face_model = "buffalo_l"

        # Should return without touching scheduler
        detect_faces("vid1", Path("/tmp/vid1.mp4"), catalog_db, scheduler, config)
        scheduler.model.assert_not_called()

    def test_no_existing_faces_proceeds(self, catalog_db, tmp_path) -> None:
        """When no faces exist, detect_faces proceeds (calls scheduler)."""
        from autopilot.analyze.faces import detect_faces

        catalog_db.insert_media("vid2", "/tmp/vid2.mp4")

        scheduler = MagicMock()
        config = MagicMock()
        config.face_model = "buffalo_l"

        # Create a dummy video file so path exists check passes
        vid = tmp_path / "vid2.mp4"
        vid.touch()

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=30)
        mock_cv2.VideoCapture.return_value = cap

        face_model = MagicMock()
        face_model.get.return_value = []
        scheduler.model.return_value.__enter__ = MagicMock(return_value=face_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_faces("vid2", vid, catalog_db, scheduler, config)

        scheduler.model.assert_called_once()
