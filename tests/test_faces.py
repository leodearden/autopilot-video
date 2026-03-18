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


# -- Error handling tests ------------------------------------------------------


class TestErrorHandling:
    """Tests for detect_faces error handling."""

    def test_video_not_found_raises(self, catalog_db) -> None:
        """Nonexistent video path raises FaceDetectionError matching 'not found'."""
        from autopilot.analyze.faces import FaceDetectionError, detect_faces

        catalog_db.insert_media("vid1", "/nonexistent.mp4")
        scheduler = MagicMock()
        config = MagicMock()
        config.face_model = "buffalo_l"

        with pytest.raises(FaceDetectionError, match="not found"):
            detect_faces(
                "vid1", Path("/nonexistent.mp4"), catalog_db, scheduler, config
            )

    def test_video_not_opened_raises(self, catalog_db, tmp_path) -> None:
        """cap.isOpened() returns False raises FaceDetectionError."""
        from autopilot.analyze.faces import FaceDetectionError, detect_faces

        vid = tmp_path / "bad.mp4"
        vid.touch()
        catalog_db.insert_media("vid1", str(vid))

        mock_cv2 = _make_mock_cv2()
        cap = MagicMock()
        cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = cap

        scheduler = MagicMock()
        config = MagicMock()
        config.face_model = "buffalo_l"

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with pytest.raises(FaceDetectionError, match="Failed to open"):
                detect_faces("vid1", vid, catalog_db, scheduler, config)

    def test_frame_read_failure_skipped(self, catalog_db, tmp_path) -> None:
        """Frame read failure skips that frame, other frames still processed."""
        from autopilot.analyze.faces import detect_faces

        vid = tmp_path / "vid.mp4"
        vid.touch()
        catalog_db.insert_media("vid1", str(vid))

        mock_cv2 = _make_mock_cv2()
        # 90 frames at 30fps = 3 frames sampled (0, 30, 60)
        cap = _make_mock_capture(fps=30.0, total_frames=90)
        # First read fails, second and third succeed
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cap.read.side_effect = [(False, None), (True, frame), (True, frame)]
        mock_cv2.VideoCapture.return_value = cap

        face_model = MagicMock()
        face_model.get.return_value = [_make_mock_face()]

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=face_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        config = MagicMock()
        config.face_model = "buffalo_l"

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_faces("vid1", vid, catalog_db, scheduler, config)

        # 2 faces stored (frames 30 and 60 succeed, frame 0 fails)
        faces = catalog_db.get_faces_for_media("vid1")
        assert len(faces) == 2


# -- Detection loop tests -----------------------------------------------------


class TestDetectionLoop:
    """Tests for the core face detection sampling loop."""

    def _run_detect(
        self, catalog_db, tmp_path, fps, total_frames, faces_per_frame
    ):
        """Helper to run detect_faces with mocked dependencies."""
        from autopilot.analyze.faces import detect_faces

        vid = tmp_path / "vid.mp4"
        vid.touch()
        catalog_db.insert_media("vid1", str(vid))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=fps, total_frames=total_frames)
        mock_cv2.VideoCapture.return_value = cap

        # Create distinct faces per call with distinct embeddings
        call_count = [0]

        def make_faces_for_frame(frame):
            rng = np.random.default_rng(call_count[0])
            call_count[0] += 1
            return [
                _make_mock_face(embedding=rng.random(512).astype(np.float32))
                for _ in range(faces_per_frame)
            ]

        face_model = MagicMock()
        face_model.get.side_effect = make_faces_for_frame

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=face_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        config = MagicMock()
        config.face_model = "buffalo_l"

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_faces("vid1", vid, catalog_db, scheduler, config)

        return face_model, cap

    def test_samples_at_1fps(self, catalog_db, tmp_path) -> None:
        """300 frames at 30fps: model.get() called 10 times."""
        face_model, _ = self._run_detect(
            catalog_db, tmp_path, fps=30.0, total_frames=300, faces_per_frame=1
        )
        assert face_model.get.call_count == 10

    def test_stores_faces_with_embeddings(self, catalog_db, tmp_path) -> None:
        """2 faces per frame on 10 sampled frames = 20 face rows."""
        self._run_detect(
            catalog_db, tmp_path, fps=30.0, total_frames=300, faces_per_frame=2
        )
        faces = catalog_db.get_faces_for_media("vid1")
        assert len(faces) == 20

        # Check frame numbers are 0, 30, 60, ..., 270
        frame_numbers = sorted(set(f["frame_number"] for f in faces))
        assert frame_numbers == list(range(0, 300, 30))

        # Check face_index is 0 and 1 for each frame
        for fn in frame_numbers:
            frame_faces = [f for f in faces if f["frame_number"] == fn]
            assert sorted(f["face_index"] for f in frame_faces) == [0, 1]

    def test_single_face_per_frame(self, catalog_db, tmp_path) -> None:
        """1 face detected stores 1 row with face_index=0."""
        self._run_detect(
            catalog_db, tmp_path, fps=30.0, total_frames=30, faces_per_frame=1
        )
        faces = catalog_db.get_faces_for_media("vid1")
        assert len(faces) == 1
        assert faces[0]["face_index"] == 0
