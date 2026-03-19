"""Tests for face detection and clustering (autopilot.analyze.faces)."""

from __future__ import annotations

import inspect
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np  # type: ignore[reportMissingImports]
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

    def test_video_not_opened_releases_capture(self, catalog_db, tmp_path) -> None:
        """When cap.isOpened() returns False, cap.release() is still called."""
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

        # cap.release() must be called even for failed-open captures
        cap.release.assert_called_once()

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


# -- Data format tests ---------------------------------------------------------


class TestDataFormat:
    """Tests for stored face data format correctness."""

    def _run_with_face(self, catalog_db, tmp_path, face):
        """Helper to run detect_faces with a specific mock face."""
        from autopilot.analyze.faces import detect_faces

        vid = tmp_path / "vid.mp4"
        vid.touch()
        catalog_db.insert_media("vid1", str(vid))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=30)
        mock_cv2.VideoCapture.return_value = cap

        face_model = MagicMock()
        face_model.get.return_value = [face]

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=face_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        config = MagicMock()
        config.face_model = "buffalo_l"

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_faces("vid1", vid, catalog_db, scheduler, config)

        return catalog_db.get_faces_for_media("vid1")

    def test_bbox_json_valid(self, catalog_db, tmp_path) -> None:
        """Stored bbox_json is valid JSON list of 4 floats."""
        face = _make_mock_face(bbox=[10.5, 20.5, 100.5, 200.5])
        faces = self._run_with_face(catalog_db, tmp_path, face)
        assert len(faces) == 1
        bbox = json.loads(faces[0]["bbox_json"])
        assert isinstance(bbox, list)
        assert len(bbox) == 4
        assert all(isinstance(v, float) for v in bbox)
        assert bbox == [10.5, 20.5, 100.5, 200.5]

    def test_embedding_size(self, catalog_db, tmp_path) -> None:
        """Embedding BLOB is exactly 2048 bytes (512 × 4-byte float32)."""
        face = _make_mock_face()
        faces = self._run_with_face(catalog_db, tmp_path, face)
        assert len(faces) == 1
        assert len(faces[0]["embedding"]) == 2048

    def test_embedding_roundtrip(self, catalog_db, tmp_path) -> None:
        """Store then retrieve numpy array via tobytes/frombuffer matches original."""
        original_emb = np.random.default_rng(99).random(512).astype(np.float32)
        face = _make_mock_face(embedding=original_emb)
        faces = self._run_with_face(catalog_db, tmp_path, face)
        assert len(faces) == 1
        recovered = np.frombuffer(faces[0]["embedding"], dtype=np.float32)
        np.testing.assert_array_equal(recovered, original_emb)

    def test_cluster_id_starts_null(self, catalog_db, tmp_path) -> None:
        """Newly inserted faces have cluster_id=None."""
        face = _make_mock_face()
        faces = self._run_with_face(catalog_db, tmp_path, face)
        assert len(faces) == 1
        assert faces[0]["cluster_id"] is None


# -- Edge case tests -----------------------------------------------------------


class TestEdgeCases:
    """Tests for detect_faces edge cases."""

    def test_no_faces_detected_no_rows(self, catalog_db, tmp_path) -> None:
        """When model returns empty list, no face rows stored."""
        from autopilot.analyze.faces import detect_faces

        vid = tmp_path / "vid.mp4"
        vid.touch()
        catalog_db.insert_media("vid1", str(vid))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=30)
        mock_cv2.VideoCapture.return_value = cap

        face_model = MagicMock()
        face_model.get.return_value = []

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=face_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        config = MagicMock()
        config.face_model = "buffalo_l"

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_faces("vid1", vid, catalog_db, scheduler, config)

        faces = catalog_db.get_faces_for_media("vid1")
        assert len(faces) == 0

    def test_empty_video_no_error(self, catalog_db, tmp_path) -> None:
        """0 total frames returns early without error."""
        from autopilot.analyze.faces import detect_faces

        vid = tmp_path / "vid.mp4"
        vid.touch()
        catalog_db.insert_media("vid1", str(vid))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=0)
        mock_cv2.VideoCapture.return_value = cap

        scheduler = MagicMock()
        config = MagicMock()
        config.face_model = "buffalo_l"

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_faces("vid1", vid, catalog_db, scheduler, config)

        # Should not call scheduler.model for empty video
        scheduler.model.assert_not_called()


# -- Logging tests -------------------------------------------------------------


class TestLogging:
    """Tests for detect_faces logging output."""

    def test_log_contains_media_id(self, catalog_db, tmp_path, caplog) -> None:
        """INFO log with media_id on start."""
        from autopilot.analyze.faces import detect_faces

        vid = tmp_path / "vid.mp4"
        vid.touch()
        catalog_db.insert_media("vid1", str(vid))

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=30)
        mock_cv2.VideoCapture.return_value = cap

        face_model = MagicMock()
        face_model.get.return_value = [_make_mock_face()]

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=face_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        config = MagicMock()
        config.face_model = "buffalo_l"

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.faces"):
            with patch.dict(sys.modules, {"cv2": mock_cv2}):
                detect_faces("vid1", vid, catalog_db, scheduler, config)

        assert any("vid1" in rec.message for rec in caplog.records)

    def test_log_skipping_on_idempotent(self, catalog_db, caplog) -> None:
        """INFO log with 'skipping' when faces already exist."""
        from autopilot.analyze.faces import detect_faces

        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        catalog_db.batch_insert_faces([
            ("vid1", 0, 0, "[]", b"\x00" * 2048, None),
        ])

        scheduler = MagicMock()
        config = MagicMock()
        config.face_model = "buffalo_l"

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.faces"):
            detect_faces("vid1", Path("/tmp/vid1.mp4"), catalog_db, scheduler, config)

        assert any("skipping" in rec.message.lower() for rec in caplog.records)


# -- Cluster faces tests -------------------------------------------------------


class TestClusterFaces:
    """Tests for cluster_faces DBSCAN clustering."""

    def test_empty_db_no_error(self, catalog_db) -> None:
        """cluster_faces with no face rows creates no clusters, no error."""
        from autopilot.analyze.faces import cluster_faces

        cluster_faces(catalog_db)
        clusters = catalog_db.get_face_clusters()
        assert len(clusters) == 0

    def _insert_synthetic_faces(self, catalog_db, n_clusters=3, n_per_cluster=5):
        """Insert synthetic face embeddings forming distinct clusters.

        Creates n_clusters groups, each with n_per_cluster faces.
        Embeddings are unit vectors with dominant weight on different dimensions.
        """
        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        rng = np.random.default_rng(42)
        face_rows = []
        frame_num = 0

        for cluster_idx in range(n_clusters):
            for face_idx in range(n_per_cluster):
                # Create embedding with strong signal on one dimension
                emb = rng.normal(0, 0.01, 512).astype(np.float32)
                emb[cluster_idx] = 1.0  # Dominant feature
                emb = emb / np.linalg.norm(emb)  # L2 normalize
                face_rows.append((
                    "vid1",
                    frame_num,
                    0,
                    json.dumps([10.0, 20.0, 100.0, 200.0]),
                    emb.tobytes(),
                    None,
                ))
                frame_num += 1

        catalog_db.batch_insert_faces(face_rows)
        return face_rows

    def test_clusters_three_groups(self, catalog_db) -> None:
        """3 synthetic clusters of 5 faces each produce 3 face_clusters rows."""
        from autopilot.analyze.faces import cluster_faces

        self._insert_synthetic_faces(catalog_db, n_clusters=3, n_per_cluster=5)
        cluster_faces(catalog_db, eps=0.5, min_samples=3)

        clusters = catalog_db.get_face_clusters()
        assert len(clusters) == 3

        # Verify all faces have a cluster_id assigned
        faces = catalog_db.get_faces_for_media("vid1")
        assert all(f["cluster_id"] is not None for f in faces)

        # Verify faces within same synthetic group share same cluster_id
        for start in range(0, 15, 5):
            group_ids = set()
            for i in range(5):
                face = faces[start + i]
                group_ids.add(face["cluster_id"])
            assert len(group_ids) == 1  # All same cluster

    def test_centroid_is_normalized_mean(self, catalog_db) -> None:
        """representative_embedding BLOB decodes to L2-normalized mean."""
        from autopilot.analyze.faces import cluster_faces

        self._insert_synthetic_faces(catalog_db, n_clusters=2, n_per_cluster=5)
        cluster_faces(catalog_db, eps=0.5, min_samples=3)

        clusters = catalog_db.get_face_clusters()
        for cluster in clusters:
            centroid = np.frombuffer(
                cluster["representative_embedding"], dtype=np.float32
            )
            # Should be L2-normalized (norm ~= 1.0)
            assert abs(np.linalg.norm(centroid) - 1.0) < 1e-5
            assert len(centroid) == 512

    def test_sample_paths_populated(self, catalog_db) -> None:
        """sample_image_paths is JSON list of 'media_id:frame_number' strings."""
        from autopilot.analyze.faces import cluster_faces

        self._insert_synthetic_faces(catalog_db, n_clusters=2, n_per_cluster=5)
        cluster_faces(catalog_db, eps=0.5, min_samples=3)

        clusters = catalog_db.get_face_clusters()
        for cluster in clusters:
            paths = json.loads(cluster["sample_image_paths"])
            assert isinstance(paths, list)
            assert len(paths) <= 5
            assert len(paths) > 0
            for p in paths:
                assert ":" in p  # format "media_id:frame_number"

    def test_label_starts_null(self, catalog_db) -> None:
        """face_clusters.label is NULL for all clusters."""
        from autopilot.analyze.faces import cluster_faces

        self._insert_synthetic_faces(catalog_db, n_clusters=2, n_per_cluster=5)
        cluster_faces(catalog_db, eps=0.5, min_samples=3)

        clusters = catalog_db.get_face_clusters()
        for cluster in clusters:
            assert cluster["label"] is None

    def test_noise_excluded_from_clusters(self, catalog_db) -> None:
        """Outlier faces have cluster_id=NULL, no face_cluster row for noise."""
        from autopilot.analyze.faces import cluster_faces

        # Insert 2 tight clusters of 5 each + 2 outliers
        self._insert_synthetic_faces(catalog_db, n_clusters=2, n_per_cluster=5)

        # Add outliers at random locations in embedding space
        rng = np.random.default_rng(99)
        for i in range(2):
            emb = rng.normal(0, 1, 512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            catalog_db.batch_insert_faces([
                ("vid1", 100 + i, 0, "[]", emb.tobytes(), None),
            ])

        cluster_faces(catalog_db, eps=0.5, min_samples=3)

        # Outlier faces should have cluster_id=NULL
        faces = catalog_db.get_faces_for_media("vid1")
        outlier_faces = [f for f in faces if f["frame_number"] >= 100]
        assert all(f["cluster_id"] is None for f in outlier_faces)

        # Only 2 clusters, not 3 or more
        clusters = catalog_db.get_face_clusters()
        assert len(clusters) == 2

    def test_single_face_is_noise(self, catalog_db) -> None:
        """One isolated face with min_samples=3 is labeled noise."""
        from autopilot.analyze.faces import cluster_faces

        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        rng = np.random.default_rng(42)
        emb = rng.random(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        catalog_db.batch_insert_faces([
            ("vid1", 0, 0, "[]", emb.tobytes(), None),
        ])

        cluster_faces(catalog_db, min_samples=3)

        clusters = catalog_db.get_face_clusters()
        assert len(clusters) == 0

        faces = catalog_db.get_faces_for_media("vid1")
        assert faces[0]["cluster_id"] is None

    def test_atomic_rebuild_on_failure(self) -> None:
        """Failed re-clustering rolls back destructive clear, preserving old state.

        Uses a non-autocommit DB so that CatalogDB.__exit__ rollback actually
        reverses the clear_face_clusters + reset_face_cluster_ids mutations.
        """
        from autopilot.db import CatalogDB
        from autopilot.analyze.faces import cluster_faces

        # Create non-autocommit DB for proper transaction semantics
        db = CatalogDB(":memory:")
        try:
            db.insert_media("vid1", "/tmp/vid1.mp4")
            db.conn.commit()

            # Insert 2 clusters of 5 faces each
            rng = np.random.default_rng(42)
            rows = []
            for c in range(2):
                for i in range(5):
                    emb = rng.normal(0, 0.01, 512).astype(np.float32)
                    emb[c] = 1.0
                    emb = emb / np.linalg.norm(emb)
                    rows.append(("vid1", c * 5 + i, 0, "[]", emb.tobytes(), None))
            with db:
                db.batch_insert_faces(rows)

            # First clustering succeeds
            cluster_faces(db, eps=0.5, min_samples=3)

            # Verify initial state: 2 clusters, all faces have cluster_id
            clusters_before = db.get_face_clusters()
            assert len(clusters_before) == 2
            faces_before = db.get_faces_for_media("vid1")
            assert all(f["cluster_id"] is not None for f in faces_before)
            original_ids = [f["cluster_id"] for f in faces_before]

            # Monkey-patch insert_face_cluster to fail on second call
            call_count = [0]
            orig_insert = db.insert_face_cluster

            def failing_insert(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] >= 2:
                    raise RuntimeError("Simulated DB failure")
                return orig_insert(*args, **kwargs)

            db.insert_face_cluster = failing_insert  # type: ignore[assignment]

            # Re-clustering should fail
            with pytest.raises(RuntimeError, match="Simulated DB failure"):
                cluster_faces(db, eps=0.5, min_samples=3)

            # Original state must be preserved (rollback undid the clear)
            clusters_after = db.get_face_clusters()
            assert len(clusters_after) == 2, (
                f"Expected 2 clusters preserved, got {len(clusters_after)}"
            )
            faces_after = db.get_faces_for_media("vid1")
            assert all(f["cluster_id"] is not None for f in faces_after), (
                "All faces should retain their original cluster_ids"
            )
            assert [f["cluster_id"] for f in faces_after] == original_ids
        finally:
            db.close()

    def test_reclustering_replaces_old(self, catalog_db) -> None:
        """Running cluster_faces twice replaces old clusters, not doubles them."""
        from autopilot.analyze.faces import cluster_faces

        self._insert_synthetic_faces(catalog_db, n_clusters=2, n_per_cluster=5)
        cluster_faces(catalog_db, eps=0.5, min_samples=3)
        clusters1 = catalog_db.get_face_clusters()
        assert len(clusters1) == 2

        # Run again — should produce same result, not doubled
        cluster_faces(catalog_db, eps=0.5, min_samples=3)
        clusters2 = catalog_db.get_face_clusters()
        assert len(clusters2) == 2

        # All faces should still have cluster_id assigned
        faces = catalog_db.get_faces_for_media("vid1")
        assert all(f["cluster_id"] is not None for f in faces)


# -- Cluster faces logging tests -----------------------------------------------


class TestClusterFacesLogging:
    """Tests for cluster_faces logging output."""

    def test_log_cluster_count(self, catalog_db, caplog) -> None:
        """INFO log contains number of clusters."""
        from autopilot.analyze.faces import cluster_faces

        # Insert 2 clusters of 5 faces
        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        rng = np.random.default_rng(42)
        rows = []
        for c in range(2):
            for i in range(5):
                emb = rng.normal(0, 0.01, 512).astype(np.float32)
                emb[c] = 1.0
                emb = emb / np.linalg.norm(emb)
                rows.append(("vid1", c * 5 + i, 0, "[]", emb.tobytes(), None))
        catalog_db.batch_insert_faces(rows)

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.faces"):
            cluster_faces(catalog_db, eps=0.5, min_samples=3)

        assert any("2 cluster" in rec.message for rec in caplog.records)

    def test_log_noise_count(self, catalog_db, caplog) -> None:
        """INFO log contains noise face count."""
        from autopilot.analyze.faces import cluster_faces

        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        rng = np.random.default_rng(42)
        rows = []
        # One tight cluster of 5
        for i in range(5):
            emb = rng.normal(0, 0.01, 512).astype(np.float32)
            emb[0] = 1.0
            emb = emb / np.linalg.norm(emb)
            rows.append(("vid1", i, 0, "[]", emb.tobytes(), None))
        # One outlier
        emb = rng.normal(0, 1, 512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        rows.append(("vid1", 10, 0, "[]", emb.tobytes(), None))
        catalog_db.batch_insert_faces(rows)

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.faces"):
            cluster_faces(catalog_db, eps=0.5, min_samples=3)

        assert any("noise" in rec.message.lower() for rec in caplog.records)

    def test_log_empty_skip(self, catalog_db, caplog) -> None:
        """Log mentions 'No face embeddings' when DB is empty."""
        from autopilot.analyze.faces import cluster_faces

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.faces"):
            cluster_faces(catalog_db)

        assert any(
            "no face embeddings" in rec.message.lower() for rec in caplog.records
        )


# -- Label faces tests ---------------------------------------------------------


class TestLabelFaces:
    """Tests for scripts/label_faces.py helper functions."""

    def test_get_unlabeled_clusters(self, catalog_db) -> None:
        """Returns only clusters with label IS NULL."""
        from scripts.label_faces import get_unlabeled_clusters

        catalog_db.insert_face_cluster(0, label=None)
        catalog_db.insert_face_cluster(1, label="Alice")
        catalog_db.insert_face_cluster(2, label=None)

        unlabeled = get_unlabeled_clusters(catalog_db)
        assert len(unlabeled) == 2
        ids = {c["cluster_id"] for c in unlabeled}
        assert ids == {0, 2}

    def test_apply_label(self, catalog_db) -> None:
        """Updates face_clusters.label in DB."""
        from scripts.label_faces import apply_label

        catalog_db.insert_face_cluster(0, label=None)
        apply_label(catalog_db, 0, "Bob")

        cluster = catalog_db.get_face_cluster_by_id(0)
        assert cluster["label"] == "Bob"

    def test_label_roundtrip(self, catalog_db) -> None:
        """Label then retrieve verifies value."""
        from scripts.label_faces import apply_label, get_unlabeled_clusters

        catalog_db.insert_face_cluster(0, label=None)
        apply_label(catalog_db, 0, "Charlie")

        unlabeled = get_unlabeled_clusters(catalog_db)
        assert len(unlabeled) == 0

        cluster = catalog_db.get_face_cluster_by_id(0)
        assert cluster["label"] == "Charlie"

    def test_all_labeled_returns_empty(self, catalog_db) -> None:
        """After labeling all, get_unlabeled returns empty list."""
        from scripts.label_faces import apply_label, get_unlabeled_clusters

        catalog_db.insert_face_cluster(0, label=None)
        catalog_db.insert_face_cluster(1, label=None)
        apply_label(catalog_db, 0, "A")
        apply_label(catalog_db, 1, "B")

        unlabeled = get_unlabeled_clusters(catalog_db)
        assert len(unlabeled) == 0


# -- Integration test ----------------------------------------------------------


class TestIntegration:
    """End-to-end test: detect_faces then cluster_faces."""

    def test_detect_then_cluster_pipeline(self, catalog_db, tmp_path) -> None:
        """Run detect_faces on mock video, then cluster_faces."""
        from autopilot.analyze.faces import cluster_faces, detect_faces

        vid = tmp_path / "vid.mp4"
        vid.touch()
        catalog_db.insert_media("vid1", str(vid))

        mock_cv2 = _make_mock_cv2()
        # 90 frames at 30fps = 3 frames sampled (0, 30, 60)
        cap = _make_mock_capture(fps=30.0, total_frames=90)
        mock_cv2.VideoCapture.return_value = cap

        # Two distinct embedding groups for 2 faces per frame
        rng = np.random.default_rng(42)

        def make_faces(frame):
            faces = []
            for group_idx in range(2):
                emb = rng.normal(0, 0.01, 512).astype(np.float32)
                emb[group_idx] = 1.0
                emb = emb / np.linalg.norm(emb)
                faces.append(_make_mock_face(embedding=emb))
            return faces

        face_model = MagicMock()
        face_model.get.side_effect = make_faces

        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(return_value=face_model)
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        config = MagicMock()
        config.face_model = "buffalo_l"

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            detect_faces("vid1", vid, catalog_db, scheduler, config)

        # 3 frames x 2 faces = 6 face rows
        faces = catalog_db.get_faces_for_media("vid1")
        assert len(faces) == 6

        # Now cluster
        cluster_faces(catalog_db, eps=0.5, min_samples=2)

        # 2 clusters expected (one per embedding group)
        clusters = catalog_db.get_face_clusters()
        assert len(clusters) == 2

        # All faces should have cluster_id assigned
        faces = catalog_db.get_faces_for_media("vid1")
        assert all(f["cluster_id"] is not None for f in faces)

        # Clusters have valid representative_embedding
        for cluster in clusters:
            emb_blob = cluster["representative_embedding"]
            assert len(emb_blob) == 2048  # 512 x float32
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

        # sample_image_paths is valid JSON
        for cluster in clusters:
            paths = json.loads(cluster["sample_image_paths"])
            assert isinstance(paths, list)
            assert len(paths) > 0

        # Labels are NULL
        for cluster in clusters:
            assert cluster["label"] is None
