"""Tests for frame embeddings and search index (autopilot.analyze.embeddings)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_cv2():
    """Create a MagicMock cv2 module with correct CAP_PROP constants."""
    mock_cv2 = MagicMock()
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.cvtColor.side_effect = lambda frame, code: frame  # pass through
    return mock_cv2


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


def _make_mock_siglip_model(dim: int = 768):
    """Create a mock SigLIP (model, processor) tuple.

    The model returns random L2-normalized embeddings of the specified dimension.
    """
    model = MagicMock()
    processor = MagicMock()

    # model.parameters() must return an iterator with at least one tensor
    # to enable next(model.parameters()).device
    param = MagicMock()
    param.device = "cpu"
    model.parameters.side_effect = lambda: iter([param])

    def _get_image_features(**kwargs):
        # Return a tensor-like mock with the right shape
        batch_size = 1
        # Determine batch size from pixel_values if available
        pv = kwargs.get("pixel_values")
        if pv is not None and hasattr(pv, "shape"):
            batch_size = pv.shape[0]
        elif pv is not None and hasattr(pv, "__len__"):
            batch_size = len(pv)
        embeddings = np.random.randn(batch_size, dim).astype(np.float32)
        # L2-normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1.0, norms)
        result = MagicMock()
        result.cpu.return_value.numpy.return_value = embeddings
        return result

    def _get_text_features(**kwargs):
        embeddings = np.random.randn(1, dim).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.where(norms == 0, 1.0, norms)
        result = MagicMock()
        result.cpu.return_value.numpy.return_value = embeddings
        return result

    model.get_image_features.side_effect = _get_image_features
    model.get_text_features.side_effect = _get_text_features

    def _process_images(images=None, text=None, return_tensors=None, padding=None):
        result = {}
        if images is not None:
            pixel_values = MagicMock()
            pixel_values.shape = (len(images), 3, 224, 224)
            pixel_values.__len__ = lambda self: len(images)
            pixel_values.to = MagicMock(return_value=pixel_values)
            result["pixel_values"] = pixel_values
        if text is not None:
            input_ids = MagicMock()
            input_ids.to = MagicMock(return_value=input_ids)
            result["input_ids"] = input_ids
        return result

    processor.side_effect = _process_images

    return model, processor


def _make_embedding_blob(dim: int = 768) -> bytes:
    """Create a random L2-normalized embedding as a float32 BLOB."""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tobytes()


def _setup_mock_modules():
    """Return dict of mock heavy modules for sys.modules patching."""
    mock_cv2 = _make_mock_cv2()
    mock_torch = MagicMock()
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    mock_pil = MagicMock()
    mock_pil_image = MagicMock()
    mock_pil.Image = mock_pil_image
    mock_pil_image.fromarray = MagicMock(side_effect=lambda x: MagicMock())

    return {
        "cv2": mock_cv2,
        "torch": mock_torch,
        "PIL": mock_pil,
        "PIL.Image": mock_pil_image,
    }


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestDBHelpers:
    """Tests for get_all_clip_embeddings DB method."""

    def test_empty_db_returns_empty(self, catalog_db) -> None:
        """Empty database returns empty list."""
        result = catalog_db.get_all_clip_embeddings()
        assert result == []

    def test_returns_all_embeddings(self, catalog_db) -> None:
        """Returns embeddings across multiple media files."""
        catalog_db.insert_media("m1", "/fake/video1.mp4")
        catalog_db.insert_media("m2", "/fake/video2.mp4")

        blob1 = _make_embedding_blob()
        blob2 = _make_embedding_blob()
        blob3 = _make_embedding_blob()

        catalog_db.batch_insert_embeddings(
            [
                ("m1", 0, blob1),
                ("m1", 60, blob2),
                ("m2", 0, blob3),
            ]
        )

        result = catalog_db.get_all_clip_embeddings()
        assert len(result) == 3

    def test_returns_correct_columns(self, catalog_db) -> None:
        """Each row has media_id, frame_number, embedding keys."""
        catalog_db.insert_media("m1", "/fake/video.mp4")
        blob = _make_embedding_blob()
        catalog_db.batch_insert_embeddings([("m1", 0, blob)])

        result = catalog_db.get_all_clip_embeddings()
        assert len(result) == 1
        row = result[0]
        assert row["media_id"] == "m1"
        assert row["frame_number"] == 0
        assert isinstance(row["embedding"], bytes)

    def test_embedding_blob_roundtrip(self, catalog_db) -> None:
        """Embedding BLOB survives insert/read roundtrip."""
        catalog_db.insert_media("m1", "/fake/video.mp4")
        original = np.random.randn(768).astype(np.float32)
        original = original / np.linalg.norm(original)
        blob = original.tobytes()

        catalog_db.batch_insert_embeddings([("m1", 0, blob)])
        result = catalog_db.get_all_clip_embeddings()
        recovered = np.frombuffer(result[0]["embedding"], dtype=np.float32)

        np.testing.assert_array_almost_equal(recovered, original)


class TestPublicAPI:
    """Tests for module-level public API imports."""

    def test_compute_embeddings_importable(self) -> None:
        """compute_embeddings is importable from autopilot.analyze.embeddings."""
        from autopilot.analyze.embeddings import compute_embeddings

        assert compute_embeddings is not None

    def test_build_search_index_importable(self) -> None:
        """build_search_index is importable."""
        from autopilot.analyze.embeddings import build_search_index

        assert build_search_index is not None

    def test_search_by_text_importable(self) -> None:
        """search_by_text is importable."""
        from autopilot.analyze.embeddings import search_by_text

        assert search_by_text is not None

    def test_search_by_image_importable(self) -> None:
        """search_by_image is importable."""
        from autopilot.analyze.embeddings import search_by_image

        assert search_by_image is not None

    def test_embedding_error_is_exception(self) -> None:
        """EmbeddingError is a subclass of Exception."""
        from autopilot.analyze.embeddings import EmbeddingError

        assert issubclass(EmbeddingError, Exception)
        err = EmbeddingError("test")
        assert str(err) == "test"

    def test_compute_embeddings_signature(self) -> None:
        """compute_embeddings has the expected positional and keyword args."""
        import inspect

        from autopilot.analyze.embeddings import compute_embeddings

        sig = inspect.signature(compute_embeddings)
        params = list(sig.parameters.keys())
        assert "media_id" in params
        assert "video_path" in params
        assert "db" in params
        assert "scheduler" in params
        assert "config" in params
        assert sig.parameters["sample_fps"].kind == inspect.Parameter.KEYWORD_ONLY
        assert sig.parameters["batch_size"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_search_by_text_signature(self) -> None:
        """search_by_text has the expected args."""
        import inspect

        from autopilot.analyze.embeddings import search_by_text

        sig = inspect.signature(search_by_text)
        params = list(sig.parameters.keys())
        assert "query" in params
        assert "index_path" in params
        assert "model" in params
        assert sig.parameters["top_k"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_search_by_image_signature(self) -> None:
        """search_by_image has the expected args."""
        import inspect

        from autopilot.analyze.embeddings import search_by_image

        sig = inspect.signature(search_by_image)
        params = list(sig.parameters.keys())
        assert "image" in params
        assert "index_path" in params
        assert "model" in params
        assert sig.parameters["top_k"].kind == inspect.Parameter.KEYWORD_ONLY


class TestComputeSampleIndices:
    """Tests for _compute_sample_indices helper."""

    def test_half_fps_30fps_video(self) -> None:
        """0.5 FPS on 30fps video: interval=60, so [0, 60, 120, ...]."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(total_frames=300, fps=30.0, sample_fps=0.5)
        expected = list(range(0, 300, 60))
        assert result == expected

    def test_half_fps_24fps_video(self) -> None:
        """0.5 FPS on 24fps video: interval=48."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(total_frames=240, fps=24.0, sample_fps=0.5)
        expected = list(range(0, 240, 48))
        assert result == expected

    def test_1fps_sampling(self) -> None:
        """1 FPS on 30fps video: interval=30."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(total_frames=90, fps=30.0, sample_fps=1.0)
        expected = list(range(0, 90, 30))
        assert result == expected

    def test_empty_video(self) -> None:
        """Empty video returns empty list."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(total_frames=0, fps=30.0, sample_fps=0.5)
        assert result == []

    def test_zero_fps(self) -> None:
        """Zero FPS returns empty list."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(total_frames=100, fps=0.0, sample_fps=0.5)
        assert result == []

    def test_short_video(self) -> None:
        """Video shorter than one sample interval: only frame 0."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(total_frames=30, fps=30.0, sample_fps=0.5)
        assert result == [0]

    def test_sample_fps_greater_than_video_fps(self) -> None:
        """When sample_fps > video fps, interval clamps to 1."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(total_frames=5, fps=1.0, sample_fps=10.0)
        assert result == [0, 1, 2, 3, 4]


class TestIdempotency:
    """Tests for compute_embeddings idempotency check."""

    def test_existing_embeddings_skips(self, catalog_db) -> None:
        """If embeddings exist, compute_embeddings returns early."""
        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")
        blob = _make_embedding_blob()
        catalog_db.batch_insert_embeddings([("m1", 0, blob)])

        scheduler = MagicMock()
        config = ModelConfig()
        compute_embeddings("m1", Path("/fake/video.mp4"), catalog_db, scheduler, config)
        scheduler.model.assert_not_called()

    def test_no_existing_embeddings_proceeds(self, catalog_db) -> None:
        """Without existing embeddings, scheduler.model() is called."""
        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m2", "/fake/video.mp4")

        mock_model, mock_processor = _make_mock_siglip_model()
        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(
            return_value=(mock_model, mock_processor)
        )
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(total_frames=3, fps=30.0)
        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        mocks = _setup_mock_modules()
        mocks["cv2"] = mock_cv2

        with patch.dict(sys.modules, mocks):
            with patch.object(Path, "exists", return_value=True):
                compute_embeddings(
                    "m2",
                    Path("/fake/video.mp4"),
                    catalog_db,
                    scheduler,
                    config=ModelConfig(),
                )

        scheduler.model.assert_called_once()


class TestErrorHandling:
    """Tests for error handling in compute_embeddings."""

    def test_video_not_found_raises(self, catalog_db) -> None:
        """Missing video file raises EmbeddingError."""
        from autopilot.analyze.embeddings import EmbeddingError, compute_embeddings
        from autopilot.config import ModelConfig

        scheduler = MagicMock()
        config = ModelConfig()

        with pytest.raises(EmbeddingError, match="not found"):
            compute_embeddings(
                "m1",
                Path("/nonexistent/video.mp4"),
                catalog_db,
                scheduler,
                config,
            )

    def test_video_not_opened_raises(self, catalog_db) -> None:
        """VideoCapture.isOpened() False raises EmbeddingError."""
        from autopilot.analyze.embeddings import EmbeddingError, compute_embeddings
        from autopilot.config import ModelConfig

        scheduler = MagicMock()
        config = ModelConfig()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(EmbeddingError, match="Failed to open"):
                    compute_embeddings(
                        "m1",
                        Path("/fake/video.mp4"),
                        catalog_db,
                        scheduler,
                        config,
                    )

    def test_frame_read_failure_skipped(self, catalog_db) -> None:
        """Frame read failure is skipped; other frames still processed."""
        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        mock_model, mock_processor = _make_mock_siglip_model()
        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(
            return_value=(mock_model, mock_processor)
        )
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        # total_frames=120, fps=30, sample_fps=0.5 -> interval=60 -> frames [0, 60]
        mock_cap = _make_mock_capture(total_frames=120, fps=30.0)
        # First read succeeds, second fails
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, frame),
            (False, None),
        ]

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        mocks = _setup_mock_modules()
        mocks["cv2"] = mock_cv2

        with patch.dict(sys.modules, mocks):
            with patch.object(Path, "exists", return_value=True):
                compute_embeddings(
                    "m1",
                    Path("/fake/video.mp4"),
                    catalog_db,
                    scheduler,
                    config=ModelConfig(),
                    sample_fps=0.5,
                )

        # Frame 0 should have an embedding in the DB
        rows = catalog_db.get_embeddings_for_media("m1")
        assert len(rows) == 1
        assert rows[0]["frame_number"] == 0


class TestEmbeddingComputation:
    """Tests for correct embedding computation behavior."""

    def _run_compute(self, catalog_db, total_frames=120, fps=30.0, sample_fps=0.5):
        """Helper to run compute_embeddings with mocked video and model."""
        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        mock_model, mock_processor = _make_mock_siglip_model()
        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(
            return_value=(mock_model, mock_processor)
        )
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(total_frames=total_frames, fps=fps)
        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        mocks = _setup_mock_modules()
        mocks["cv2"] = mock_cv2

        with patch.dict(sys.modules, mocks):
            with patch.object(Path, "exists", return_value=True):
                compute_embeddings(
                    "m1",
                    Path("/fake/video.mp4"),
                    catalog_db,
                    scheduler,
                    config=ModelConfig(),
                    sample_fps=sample_fps,
                )

        return mock_model, mock_processor

    def test_correct_frame_count(self, catalog_db) -> None:
        """120 frames at 30fps with 0.5 FPS sampling -> 2 embeddings (frames 0, 60)."""
        self._run_compute(catalog_db, total_frames=120, fps=30.0, sample_fps=0.5)
        rows = catalog_db.get_embeddings_for_media("m1")
        assert len(rows) == 2

    def test_embedding_size(self, catalog_db) -> None:
        """Each embedding is 768 * 4 bytes (768-d float32)."""
        self._run_compute(catalog_db, total_frames=120, fps=30.0, sample_fps=0.5)
        rows = catalog_db.get_embeddings_for_media("m1")
        for row in rows:
            assert len(row["embedding"]) == 768 * 4

    def test_embedding_l2_normalized(self, catalog_db) -> None:
        """Stored embeddings are L2-normalized (norm ~= 1.0)."""
        self._run_compute(catalog_db, total_frames=120, fps=30.0, sample_fps=0.5)
        rows = catalog_db.get_embeddings_for_media("m1")
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            assert norm == pytest.approx(1.0, abs=1e-5)

    def test_embedding_roundtrip(self, catalog_db) -> None:
        """Embedding BLOB can be reconstructed to float32 array."""
        self._run_compute(catalog_db, total_frames=120, fps=30.0, sample_fps=0.5)
        rows = catalog_db.get_embeddings_for_media("m1")
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32)
            assert vec.shape == (768,)
            assert vec.dtype == np.float32

    def test_model_get_image_features_called(self, catalog_db) -> None:
        """model.get_image_features is called during compute."""
        mock_model, _ = self._run_compute(
            catalog_db,
            total_frames=120,
            fps=30.0,
            sample_fps=0.5,
        )
        assert mock_model.get_image_features.call_count > 0

    def test_scheduler_model_called_with_clip_model(self, catalog_db) -> None:
        """Scheduler.model is called with config.clip_model."""
        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        mock_model, mock_processor = _make_mock_siglip_model()
        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(
            return_value=(mock_model, mock_processor)
        )
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(total_frames=120, fps=30.0)
        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        mocks = _setup_mock_modules()
        mocks["cv2"] = mock_cv2

        config = ModelConfig()
        with patch.dict(sys.modules, mocks):
            with patch.object(Path, "exists", return_value=True):
                compute_embeddings(
                    "m1",
                    Path("/fake/video.mp4"),
                    catalog_db,
                    scheduler,
                    config=config,
                    sample_fps=0.5,
                )

        scheduler.model.assert_called_once_with(config.clip_model)


class TestLogging:
    """Tests for structured log messages."""

    def test_log_contains_media_id(self, catalog_db, caplog) -> None:
        """Log contains media_id on start."""
        import logging

        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        mock_model, mock_processor = _make_mock_siglip_model()
        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(
            return_value=(mock_model, mock_processor)
        )
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(total_frames=120, fps=30.0)
        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        mocks = _setup_mock_modules()
        mocks["cv2"] = mock_cv2

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.embeddings"):
            with patch.dict(sys.modules, mocks):
                with patch.object(Path, "exists", return_value=True):
                    compute_embeddings(
                        "m1",
                        Path("/fake/video.mp4"),
                        catalog_db,
                        scheduler,
                        config=ModelConfig(),
                        sample_fps=0.5,
                    )

        assert any("m1" in r.message for r in caplog.records)

    def test_log_contains_completion(self, catalog_db, caplog) -> None:
        """Log contains completion message."""
        import logging

        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        mock_model, mock_processor = _make_mock_siglip_model()
        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(
            return_value=(mock_model, mock_processor)
        )
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        mock_cap = _make_mock_capture(total_frames=120, fps=30.0)
        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        mocks = _setup_mock_modules()
        mocks["cv2"] = mock_cv2

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.embeddings"):
            with patch.dict(sys.modules, mocks):
                with patch.object(Path, "exists", return_value=True):
                    compute_embeddings(
                        "m1",
                        Path("/fake/video.mp4"),
                        catalog_db,
                        scheduler,
                        config=ModelConfig(),
                        sample_fps=0.5,
                    )

        assert any("Completed" in r.message or "embedded" in r.message for r in caplog.records)

    def test_log_skipping_on_idempotent(self, catalog_db, caplog) -> None:
        """Log mentions 'skipping' on idempotent skip."""
        import logging

        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")
        blob = _make_embedding_blob()
        catalog_db.batch_insert_embeddings([("m1", 0, blob)])

        scheduler = MagicMock()
        config = ModelConfig()

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.embeddings"):
            compute_embeddings("m1", Path("/fake/video.mp4"), catalog_db, scheduler, config)

        assert any("skipping" in r.message.lower() for r in caplog.records)


class TestBuildSearchIndex:
    """Tests for build_search_index."""

    def test_empty_db_skips(self, catalog_db, tmp_path) -> None:
        """Empty DB produces no index file."""
        from autopilot.analyze.embeddings import build_search_index

        output = tmp_path / "index.faiss"
        mock_faiss = MagicMock()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        assert not output.exists()

    def test_small_dataset_flat_index(self, catalog_db, tmp_path) -> None:
        """<256 vectors uses IndexFlatIP."""
        from autopilot.analyze.embeddings import build_search_index

        catalog_db.insert_media("m1", "/fake/video.mp4")
        for i in range(10):
            blob = _make_embedding_blob()
            catalog_db.batch_insert_embeddings([("m1", i * 60, blob)])

        output = tmp_path / "index.faiss"
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        mock_faiss.IndexFlatIP.assert_called_once_with(768)
        mock_index.add.assert_called_once()
        mock_faiss.write_index.assert_called_once()

    def test_id_mapping_written(self, catalog_db, tmp_path) -> None:
        """JSON sidecar with ID mapping is written."""
        from autopilot.analyze.embeddings import build_search_index

        catalog_db.insert_media("m1", "/fake/video.mp4")
        for i in range(5):
            blob = _make_embedding_blob()
            catalog_db.batch_insert_embeddings([("m1", i * 60, blob)])

        output = tmp_path / "index.faiss"
        sidecar = output.with_suffix(".faiss.ids.json")

        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = MagicMock()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        assert sidecar.exists()
        with open(sidecar) as f:
            mapping = json.load(f)
        assert len(mapping) == 5
        assert mapping[0][0] == "m1"

    def test_vector_count(self, catalog_db, tmp_path) -> None:
        """Index receives the correct number of vectors."""
        from autopilot.analyze.embeddings import build_search_index

        catalog_db.insert_media("m1", "/fake/video.mp4")
        n_vecs = 20
        for i in range(n_vecs):
            blob = _make_embedding_blob()
            catalog_db.batch_insert_embeddings([("m1", i * 60, blob)])

        output = tmp_path / "index.faiss"
        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        # Check the matrix passed to add()
        add_call = mock_index.add.call_args
        matrix = add_call[0][0]
        assert matrix.shape == (n_vecs, 768)

    def test_dimension(self, catalog_db, tmp_path) -> None:
        """Index is created with correct dimension (768)."""
        from autopilot.analyze.embeddings import build_search_index

        catalog_db.insert_media("m1", "/fake/video.mp4")
        blob = _make_embedding_blob()
        catalog_db.batch_insert_embeddings([("m1", 0, blob)])

        output = tmp_path / "index.faiss"
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = MagicMock()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        mock_faiss.IndexFlatIP.assert_called_with(768)


class TestSearchByText:
    """Tests for search_by_text."""

    def _setup_index(self, tmp_path, n_vectors=10):
        """Create a mock FAISS index and sidecar for testing."""
        index_path = tmp_path / "index.faiss"
        sidecar_path = index_path.with_suffix(".faiss.ids.json")

        id_mapping = [["m1", i * 60] for i in range(n_vectors)]
        with open(sidecar_path, "w") as f:
            json.dump(id_mapping, f)

        return index_path, id_mapping

    def test_return_type(self, tmp_path) -> None:
        """Returns a list of dicts with media_id, frame_number, score."""
        from autopilot.analyze.embeddings import search_by_text

        index_path, _ = self._setup_index(tmp_path, n_vectors=5)
        mock_model, mock_processor = _make_mock_siglip_model()

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 5
        mock_faiss.read_index.return_value = mock_index
        mock_index.search.return_value = (
            np.array([[0.95, 0.90, 0.85]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"faiss": mock_faiss, "torch": mock_torch}):
            results = search_by_text(
                "a cat",
                index_path,
                (mock_model, mock_processor),
                top_k=3,
            )

        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert "media_id" in r
            assert "frame_number" in r
            assert "score" in r

    def test_top_k_respected(self, tmp_path) -> None:
        """top_k limits the number of results."""
        from autopilot.analyze.embeddings import search_by_text

        index_path, _ = self._setup_index(tmp_path, n_vectors=10)
        mock_model, mock_processor = _make_mock_siglip_model()

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 10
        mock_faiss.read_index.return_value = mock_index
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7, 0.6, 0.5]], dtype=np.float32),
            np.array([[0, 1, 2, 3, 4]], dtype=np.int64),
        )

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"faiss": mock_faiss, "torch": mock_torch}):
            search_by_text(
                "a dog",
                index_path,
                (mock_model, mock_processor),
                top_k=5,
            )

        # search was called with min(5, 10) = 5
        mock_index.search.assert_called_once()
        call_args = mock_index.search.call_args
        assert call_args[0][1] == 5

    def test_text_encoder_called(self, tmp_path) -> None:
        """model.get_text_features is called with the query."""
        from autopilot.analyze.embeddings import search_by_text

        index_path, _ = self._setup_index(tmp_path, n_vectors=5)
        mock_model, mock_processor = _make_mock_siglip_model()

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 5
        mock_faiss.read_index.return_value = mock_index
        mock_index.search.return_value = (
            np.array([[0.9]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"faiss": mock_faiss, "torch": mock_torch}):
            search_by_text(
                "a sunset",
                index_path,
                (mock_model, mock_processor),
                top_k=1,
            )

        mock_model.get_text_features.assert_called_once()

    def test_negative_indices_skipped(self, tmp_path) -> None:
        """FAISS -1 sentinel indices are filtered out."""
        from autopilot.analyze.embeddings import search_by_text

        index_path, _ = self._setup_index(tmp_path, n_vectors=5)
        mock_model, mock_processor = _make_mock_siglip_model()

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 5
        mock_faiss.read_index.return_value = mock_index
        mock_index.search.return_value = (
            np.array([[0.9, 0.0, 0.0]], dtype=np.float32),
            np.array([[0, -1, -1]], dtype=np.int64),
        )

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"faiss": mock_faiss, "torch": mock_torch}):
            results = search_by_text(
                "test",
                index_path,
                (mock_model, mock_processor),
                top_k=3,
            )

        assert len(results) == 1


class TestSearchByImage:
    """Tests for search_by_image."""

    def _setup_index(self, tmp_path, n_vectors=10):
        """Create a mock FAISS index and sidecar for testing."""
        index_path = tmp_path / "index.faiss"
        sidecar_path = index_path.with_suffix(".faiss.ids.json")

        id_mapping = [["m1", i * 60] for i in range(n_vectors)]
        with open(sidecar_path, "w") as f:
            json.dump(id_mapping, f)

        return index_path, id_mapping

    def test_return_type(self, tmp_path) -> None:
        """Returns a list of dicts with media_id, frame_number, score."""
        from autopilot.analyze.embeddings import search_by_image

        index_path, _ = self._setup_index(tmp_path, n_vectors=5)
        mock_model, mock_processor = _make_mock_siglip_model()

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 5
        mock_faiss.read_index.return_value = mock_index
        mock_index.search.return_value = (
            np.array([[0.95, 0.90]], dtype=np.float32),
            np.array([[0, 1]], dtype=np.int64),
        )

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_pil = MagicMock()
        mock_pil_image = MagicMock()
        mock_pil.Image = mock_pil_image
        mock_pil_image.fromarray = MagicMock(return_value=MagicMock())

        with patch.dict(
            sys.modules,
            {
                "faiss": mock_faiss,
                "torch": mock_torch,
                "PIL": mock_pil,
                "PIL.Image": mock_pil_image,
            },
        ):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            results = search_by_image(
                image,
                index_path,
                (mock_model, mock_processor),
                top_k=2,
            )

        assert isinstance(results, list)
        assert len(results) == 2
        for r in results:
            assert "media_id" in r
            assert "frame_number" in r
            assert "score" in r

    def test_top_k_respected(self, tmp_path) -> None:
        """top_k limits the number of results requested from FAISS."""
        from autopilot.analyze.embeddings import search_by_image

        index_path, _ = self._setup_index(tmp_path, n_vectors=10)
        mock_model, mock_processor = _make_mock_siglip_model()

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 10
        mock_faiss.read_index.return_value = mock_index
        mock_index.search.return_value = (
            np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_pil = MagicMock()
        mock_pil_image = MagicMock()
        mock_pil.Image = mock_pil_image
        mock_pil_image.fromarray = MagicMock(return_value=MagicMock())

        with patch.dict(
            sys.modules,
            {
                "faiss": mock_faiss,
                "torch": mock_torch,
                "PIL": mock_pil,
                "PIL.Image": mock_pil_image,
            },
        ):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            search_by_image(
                image,
                index_path,
                (mock_model, mock_processor),
                top_k=3,
            )

        call_args = mock_index.search.call_args
        assert call_args[0][1] == 3

    def test_vision_encoder_called(self, tmp_path) -> None:
        """model.get_image_features is called for the query image."""
        from autopilot.analyze.embeddings import search_by_image

        index_path, _ = self._setup_index(tmp_path, n_vectors=5)
        mock_model, mock_processor = _make_mock_siglip_model()

        mock_faiss = MagicMock()
        mock_index = MagicMock()
        mock_index.ntotal = 5
        mock_faiss.read_index.return_value = mock_index
        mock_index.search.return_value = (
            np.array([[0.9]], dtype=np.float32),
            np.array([[0]], dtype=np.int64),
        )

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
        mock_pil = MagicMock()
        mock_pil_image = MagicMock()
        mock_pil.Image = mock_pil_image
        mock_pil_image.fromarray = MagicMock(return_value=MagicMock())

        with patch.dict(
            sys.modules,
            {
                "faiss": mock_faiss,
                "torch": mock_torch,
                "PIL": mock_pil,
                "PIL.Image": mock_pil_image,
            },
        ):
            image = np.zeros((224, 224, 3), dtype=np.uint8)
            search_by_image(
                image,
                index_path,
                (mock_model, mock_processor),
                top_k=1,
            )

        mock_model.get_image_features.assert_called_once()


class TestIntegration:
    """End-to-end pipeline test: compute -> build index -> search."""

    def test_compute_build_search_pipeline(self, catalog_db, tmp_path) -> None:
        """Full pipeline: compute embeddings, build index, search by text."""
        from autopilot.analyze.embeddings import (
            build_search_index,
            compute_embeddings,
            search_by_text,
        )
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")

        # -- Step 1: compute_embeddings --
        mock_model, mock_processor = _make_mock_siglip_model()
        scheduler = MagicMock()
        scheduler.model.return_value.__enter__ = MagicMock(
            return_value=(mock_model, mock_processor)
        )
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        # 300 frames at 30fps, 0.5 FPS -> interval=60 -> frames [0, 60, 120, 180, 240]
        mock_cap = _make_mock_capture(total_frames=300, fps=30.0)
        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap

        mocks = _setup_mock_modules()
        mocks["cv2"] = mock_cv2

        with patch.dict(sys.modules, mocks):
            with patch.object(Path, "exists", return_value=True):
                compute_embeddings(
                    "m1",
                    Path("/fake/video.mp4"),
                    catalog_db,
                    scheduler,
                    config=ModelConfig(),
                    sample_fps=0.5,
                )

        rows = catalog_db.get_embeddings_for_media("m1")
        assert len(rows) == 5

        # -- Step 2: build_search_index --
        index_path = tmp_path / "index.faiss"
        sidecar_path = index_path.with_suffix(".faiss.ids.json")

        # Build with a real-ish FAISS mock that tracks add() calls
        mock_faiss = MagicMock()
        mock_flat_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_flat_index

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, index_path)

        # Verify the index was built with 5 vectors of dim 768
        add_call = mock_flat_index.add.call_args
        matrix = add_call[0][0]
        assert matrix.shape == (5, 768)

        # Verify the sidecar was written
        assert sidecar_path.exists()
        with open(sidecar_path) as f:
            mapping = json.load(f)
        assert len(mapping) == 5
        assert mapping[0][0] == "m1"

        # -- Step 3: search_by_text --
        # Set up search mocks
        mock_search_index = MagicMock()
        mock_search_index.ntotal = 5
        mock_faiss.read_index.return_value = mock_search_index
        mock_search_index.search.return_value = (
            np.array([[0.95, 0.90, 0.85]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"faiss": mock_faiss, "torch": mock_torch}):
            results = search_by_text(
                "a landscape",
                index_path,
                (mock_model, mock_processor),
                top_k=3,
            )

        assert len(results) == 3
        assert results[0]["media_id"] == "m1"
        assert results[0]["score"] == pytest.approx(0.95)
