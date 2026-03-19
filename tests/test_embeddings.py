"""Tests for frame embeddings and search index (autopilot.analyze.embeddings)."""

from __future__ import annotations

import inspect
import struct
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDBHelpers:
    """Tests for CatalogDB clip embedding helper methods."""

    def test_get_all_clip_embeddings_empty(self, catalog_db) -> None:
        """get_all_clip_embeddings returns [] when clip_embeddings table is empty."""
        result = catalog_db.get_all_clip_embeddings()
        assert result == []

    def test_get_all_clip_embeddings_returns_all(self, catalog_db) -> None:
        """get_all_clip_embeddings returns all rows with non-null embeddings."""
        # Insert a media file first (FK constraint)
        catalog_db.insert_media("vid1", "/v1.mp4")
        blob1 = struct.pack("f" * 768, *([0.1] * 768))
        blob2 = struct.pack("f" * 768, *([0.2] * 768))
        catalog_db.batch_insert_embeddings([
            ("vid1", 0, blob1),
            ("vid1", 60, blob2),
        ])
        result = catalog_db.get_all_clip_embeddings()
        assert len(result) == 2
        assert result[0]["media_id"] == "vid1"
        assert result[0]["frame_number"] == 0
        assert result[0]["embedding"] == blob1
        assert result[1]["frame_number"] == 60

    def test_get_all_clip_embeddings_multiple_media(self, catalog_db) -> None:
        """get_all_clip_embeddings returns rows across multiple media files."""
        for mid in ("vid1", "vid2", "vid3"):
            catalog_db.insert_media(mid, f"/{mid}.mp4")
        blob = struct.pack("f" * 768, *([0.5] * 768))
        catalog_db.batch_insert_embeddings([
            ("vid1", 0, blob),
            ("vid2", 0, blob),
            ("vid2", 60, blob),
            ("vid3", 0, blob),
        ])
        result = catalog_db.get_all_clip_embeddings()
        assert len(result) == 4
        media_ids = {r["media_id"] for r in result}
        assert media_ids == {"vid1", "vid2", "vid3"}


class TestPublicAPI:
    """Tests for module-level public API imports."""

    def test_embedding_error_importable(self) -> None:
        """EmbeddingError is importable and is an Exception subclass with message."""
        from autopilot.analyze.embeddings import EmbeddingError

        assert issubclass(EmbeddingError, Exception)
        err = EmbeddingError("test message")
        assert str(err) == "test message"

    def test_compute_embeddings_importable(self) -> None:
        """compute_embeddings is importable from autopilot.analyze.embeddings."""
        from autopilot.analyze.embeddings import compute_embeddings

        assert callable(compute_embeddings)

    def test_build_search_index_importable(self) -> None:
        """build_search_index is importable from autopilot.analyze.embeddings."""
        from autopilot.analyze.embeddings import build_search_index

        assert callable(build_search_index)

    def test_search_by_text_importable(self) -> None:
        """search_by_text is importable from autopilot.analyze.embeddings."""
        from autopilot.analyze.embeddings import search_by_text

        assert callable(search_by_text)

    def test_search_by_image_importable(self) -> None:
        """search_by_image is importable from autopilot.analyze.embeddings."""
        from autopilot.analyze.embeddings import search_by_image

        assert callable(search_by_image)

    def test_all_exports(self) -> None:
        """__all__ contains exactly the expected public API."""
        from autopilot.analyze import embeddings

        expected = {
            "EmbeddingError",
            "compute_embeddings",
            "build_search_index",
            "search_by_text",
            "search_by_image",
        }
        assert set(embeddings.__all__) == expected


class TestSignatures:
    """Tests for function signatures and parameter contracts."""

    def test_compute_embeddings_signature(self) -> None:
        """compute_embeddings has correct positional and keyword-only params."""
        from autopilot.analyze.embeddings import compute_embeddings

        sig = inspect.signature(compute_embeddings)
        params = list(sig.parameters.keys())
        # Positional params
        assert params[:5] == ["media_id", "video_path", "db", "scheduler", "config"]
        # batch_size is keyword-only with int default
        bs = sig.parameters["batch_size"]
        assert bs.kind == inspect.Parameter.KEYWORD_ONLY
        assert bs.default == 16
        assert isinstance(bs.default, int)

    def test_build_search_index_signature(self) -> None:
        """build_search_index has params (db, output_path)."""
        from autopilot.analyze.embeddings import build_search_index

        sig = inspect.signature(build_search_index)
        params = list(sig.parameters.keys())
        assert params == ["db", "output_path"]

    def test_search_by_text_signature(self) -> None:
        """search_by_text has params (query, index_path, model) and keyword-only top_k=10."""
        from autopilot.analyze.embeddings import search_by_text

        sig = inspect.signature(search_by_text)
        params = list(sig.parameters.keys())
        assert params[:3] == ["query", "index_path", "model"]
        tk = sig.parameters["top_k"]
        assert tk.kind == inspect.Parameter.KEYWORD_ONLY
        assert tk.default == 10

    def test_search_by_image_signature(self) -> None:
        """search_by_image has params (image, index_path, model) and keyword-only top_k=10."""
        from autopilot.analyze.embeddings import search_by_image

        sig = inspect.signature(search_by_image)
        params = list(sig.parameters.keys())
        assert params[:3] == ["image", "index_path", "model"]
        tk = sig.parameters["top_k"]
        assert tk.kind == inspect.Parameter.KEYWORD_ONLY
        assert tk.default == 10


class TestComputeSampleIndices:
    """Tests for _compute_sample_indices helper."""

    def test_half_fps_30fps(self) -> None:
        """300 frames at 30fps sampled at 0.5fps -> [0, 60, 120, 180, 240]."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(300, 30.0)
        assert result == [0, 60, 120, 180, 240]

    def test_half_fps_24fps(self) -> None:
        """240 frames at 24fps sampled at 0.5fps -> [0, 48, 96, 144, 192]."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(240, 24.0)
        assert result == [0, 48, 96, 144, 192]

    def test_empty_video(self) -> None:
        """0 frames -> []."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(0, 30.0)
        assert result == []

    def test_short_video(self) -> None:
        """30 frames at 30fps -> [0] (only one sample in 1 second of video)."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(30, 30.0)
        assert result == [0]

    def test_low_fps(self) -> None:
        """10 frames at 1fps -> [0, 2, 4, 6, 8]."""
        from autopilot.analyze.embeddings import _compute_sample_indices

        result = _compute_sample_indices(10, 1.0)
        assert result == [0, 2, 4, 6, 8]


# -- Mock helpers for SigLIP and cv2 ------------------------------------------


def _make_mock_capture(
    fps: float = 30.0,
    total_frames: int = 300,
    width: int = 1920,
    height: int = 1080,
) -> MagicMock:
    """Create a MagicMock mimicking cv2.VideoCapture."""
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7

    cap = MagicMock()
    cap.isOpened.return_value = True

    def get_prop(prop_id):
        return {CAP_PROP_FPS: fps, CAP_PROP_FRAME_COUNT: total_frames}.get(prop_id, 0.0)

    cap.get.side_effect = get_prop
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    cap.set.return_value = True
    return cap


def _make_mock_cv2() -> MagicMock:
    """Create a MagicMock cv2 module with correct CAP_PROP constants."""
    mock_cv2 = MagicMock()
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    return mock_cv2


def _make_mock_siglip(dim: int = 768) -> tuple[MagicMock, MagicMock]:
    """Create mock (model, processor) mimicking SigLIP2.

    Returns:
        (model, processor) where model.get_image_features() returns a
        mock tensor-like object with .detach().cpu().numpy() chain yielding
        a deterministic L2-normalized 768-d embedding, and processor returns
        mock inputs.
    """
    model = MagicMock()
    processor = MagicMock()

    # Create a deterministic L2-normalized embedding
    rng = np.random.RandomState(42)
    raw = rng.randn(1, dim).astype(np.float32)
    norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    normed = raw / norm

    # model.get_image_features returns mock with .detach().cpu().numpy() chain
    img_features = MagicMock()
    img_features.detach.return_value.cpu.return_value.numpy.return_value = normed
    model.get_image_features.return_value = img_features

    # model.get_text_features for text search
    raw_text = rng.randn(1, dim).astype(np.float32)
    norm_text = np.linalg.norm(raw_text, axis=-1, keepdims=True)
    normed_text = raw_text / norm_text
    text_features = MagicMock()
    text_features.detach.return_value.cpu.return_value.numpy.return_value = normed_text
    model.get_text_features.return_value = text_features

    # processor returns a dict-like MagicMock with .to() support
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    processor.return_value = mock_inputs

    return model, processor


def _make_mock_torch() -> MagicMock:
    """Create a mock torch module with nn.functional.normalize."""
    mock_torch = MagicMock()

    def mock_normalize(features, dim=-1):
        # Return features as-is (already normalized in mock)
        return features

    mock_torch.nn.functional.normalize.side_effect = mock_normalize
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    return mock_torch


def _make_mock_pil() -> MagicMock:
    """Create a mock PIL module."""
    mock_pil = MagicMock()
    mock_pil.Image.fromarray.return_value = MagicMock()
    return mock_pil


def _make_scheduler_mock(model_obj: object) -> MagicMock:
    """Create a mock GPUScheduler that yields model_obj from .model() context."""
    scheduler = MagicMock()
    scheduler.model.return_value.__enter__ = MagicMock(return_value=model_obj)
    scheduler.model.return_value.__exit__ = MagicMock(return_value=False)
    return scheduler


class TestIdempotency:
    """Tests for compute_embeddings idempotency check."""

    def test_existing_embeddings_skips(self, catalog_db) -> None:
        """If embeddings exist for media_id, compute_embeddings returns early."""
        from pathlib import Path

        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")
        blob = struct.pack("f" * 768, *([0.1] * 768))
        catalog_db.batch_insert_embeddings([("m1", 0, blob)])

        scheduler = MagicMock()
        config = ModelConfig()
        compute_embeddings("m1", Path("/fake/video.mp4"), catalog_db, scheduler, config)
        scheduler.model.assert_not_called()

    def test_no_existing_embeddings_proceeds(self, catalog_db) -> None:
        """Without existing embeddings, scheduler.model() is called."""
        from pathlib import Path

        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m2", "/fake/video.mp4")
        mock_model, mock_processor = _make_mock_siglip()
        scheduler = _make_scheduler_mock((mock_model, mock_processor))
        config = ModelConfig()

        mock_cap = _make_mock_capture(total_frames=3, fps=30.0)
        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_torch = _make_mock_torch()
        mock_pil = _make_mock_pil()

        with patch.dict(sys.modules, {
            "cv2": mock_cv2, "torch": mock_torch, "torch.nn": mock_torch.nn,
            "torch.nn.functional": mock_torch.nn.functional,
            "PIL": mock_pil, "PIL.Image": mock_pil.Image,
        }):
            with patch.object(Path, "exists", return_value=True):
                compute_embeddings(
                    "m2", Path("/fake/video.mp4"), catalog_db, scheduler, config
                )

        scheduler.model.assert_called_once()


def _sys_modules_patch() -> dict[str, MagicMock]:
    """Return dict of sys.modules patches for cv2/PIL (no torch needed)."""
    mock_cv2 = _make_mock_cv2()
    mock_pil = _make_mock_pil()
    return {
        "cv2": mock_cv2,
        "PIL": mock_pil,
        "PIL.Image": mock_pil.Image,
    }


class TestErrorHandling:
    """Tests for compute_embeddings error handling."""

    def test_video_not_found_raises(self, catalog_db) -> None:
        """Nonexistent video path raises EmbeddingError matching 'not found'."""
        from pathlib import Path

        from autopilot.analyze.embeddings import EmbeddingError, compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/nonexistent/video.mp4")
        scheduler = MagicMock()
        config = ModelConfig()

        with pytest.raises(EmbeddingError, match="not found"):
            compute_embeddings(
                "m1", Path("/nonexistent/video.mp4"), catalog_db, scheduler, config
            )

    def test_video_not_opened_raises(self, catalog_db) -> None:
        """Mock cv2 with cap.isOpened()=False raises EmbeddingError."""
        from pathlib import Path

        from autopilot.analyze.embeddings import EmbeddingError, compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")
        scheduler = MagicMock()
        config = ModelConfig()

        mock_cv2 = _make_mock_cv2()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(EmbeddingError, match="Failed to open"):
                    compute_embeddings(
                        "m1", Path("/fake/video.mp4"), catalog_db, scheduler, config
                    )

    def test_frame_read_failure_skipped(self, catalog_db) -> None:
        """Frame read failure on one frame still produces embeddings for others."""
        from pathlib import Path

        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/fake/video.mp4")
        mock_model, mock_processor = _make_mock_siglip()
        scheduler = _make_scheduler_mock((mock_model, mock_processor))
        config = ModelConfig()

        # 120 frames at 30fps -> 2 sample indices: [0, 60]
        mock_cap = _make_mock_capture(total_frames=120, fps=30.0)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(False, None), (True, frame)]

        mock_cv2 = _make_mock_cv2()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_pil = _make_mock_pil()

        with patch.dict(sys.modules, {
            "cv2": mock_cv2,
            "PIL": mock_pil, "PIL.Image": mock_pil.Image,
        }):
            with patch.object(Path, "exists", return_value=True):
                compute_embeddings(
                    "m1", Path("/fake/video.mp4"), catalog_db, scheduler, config
                )

        rows = catalog_db.get_embeddings_for_media("m1")
        assert len(rows) == 1


def _run_compute(catalog_db, media_id="m1", total_frames=300, fps=30.0):
    """Helper to run compute_embeddings with full mocking."""
    from pathlib import Path

    from autopilot.analyze.embeddings import compute_embeddings
    from autopilot.config import ModelConfig

    catalog_db.insert_media(media_id, f"/{media_id}.mp4")
    mock_model, mock_processor = _make_mock_siglip()
    scheduler = _make_scheduler_mock((mock_model, mock_processor))
    config = ModelConfig()

    mock_cap = _make_mock_capture(total_frames=total_frames, fps=fps)
    mock_cv2 = _make_mock_cv2()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_pil = _make_mock_pil()

    with patch.dict(sys.modules, {
        "cv2": mock_cv2,
        "PIL": mock_pil, "PIL.Image": mock_pil.Image,
    }):
        with patch.object(Path, "exists", return_value=True):
            compute_embeddings(
                media_id, Path(f"/{media_id}.mp4"), catalog_db, scheduler, config
            )

    return scheduler, mock_model, mock_processor, config


class TestEmbeddingComputation:
    """Tests for compute_embeddings core loop."""

    def test_correct_embedding_count(self, catalog_db) -> None:
        """300 frames at 30fps stores 5 embeddings."""
        _run_compute(catalog_db, total_frames=300, fps=30.0)
        rows = catalog_db.get_embeddings_for_media("m1")
        assert len(rows) == 5

    def test_embedding_size_768d(self, catalog_db) -> None:
        """Stored BLOB is 3072 bytes (768 * 4 bytes float32)."""
        _run_compute(catalog_db)
        rows = catalog_db.get_embeddings_for_media("m1")
        assert len(rows) > 0
        assert len(rows[0]["embedding"]) == 768 * 4

    def test_embedding_roundtrip(self, catalog_db) -> None:
        """np.frombuffer(blob, float32) recovers 768-d vector."""
        _run_compute(catalog_db)
        rows = catalog_db.get_embeddings_for_media("m1")
        vec = np.frombuffer(rows[0]["embedding"], dtype=np.float32)
        assert vec.shape == (768,)

    def test_model_loaded_via_scheduler(self, catalog_db) -> None:
        """scheduler.model called with config.clip_model."""
        scheduler, _, _, config = _run_compute(catalog_db)
        scheduler.model.assert_called_once_with(config.clip_model)

    def test_correct_frame_numbers(self, catalog_db) -> None:
        """DB rows at frames [0, 60, 120, 180, 240] for 300f@30fps."""
        _run_compute(catalog_db, total_frames=300, fps=30.0)
        rows = catalog_db.get_embeddings_for_media("m1")
        frame_numbers = [r["frame_number"] for r in rows]
        assert frame_numbers == [0, 60, 120, 180, 240]


class TestLogging:
    """Tests for compute_embeddings structured logging."""

    def test_log_contains_media_id_on_start(self, catalog_db, caplog) -> None:
        """INFO log includes media_id on start."""
        import logging

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.embeddings"):
            _run_compute(catalog_db, media_id="test_vid")
        assert any("test_vid" in r.message for r in caplog.records)

    def test_log_contains_completion(self, catalog_db, caplog) -> None:
        """Log includes 'Completed' or embedding count."""
        import logging

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.embeddings"):
            _run_compute(catalog_db)
        messages = " ".join(r.message for r in caplog.records)
        assert "Completed" in messages or "embeddings stored" in messages

    def test_log_skipping_on_idempotent(self, catalog_db, caplog) -> None:
        """Log mentions 'skipping' on idempotent skip."""
        import logging

        from pathlib import Path

        from autopilot.analyze.embeddings import compute_embeddings
        from autopilot.config import ModelConfig

        catalog_db.insert_media("m1", "/v.mp4")
        blob = struct.pack("f" * 768, *([0.1] * 768))
        catalog_db.batch_insert_embeddings([("m1", 0, blob)])

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.embeddings"):
            compute_embeddings("m1", Path("/v.mp4"), catalog_db, MagicMock(), ModelConfig())
        messages = " ".join(r.message for r in caplog.records).lower()
        assert "skipping" in messages

    def test_log_contains_frame_info(self, catalog_db, caplog) -> None:
        """Log mentions frames/fps."""
        import logging

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.embeddings"):
            _run_compute(catalog_db, total_frames=300, fps=30.0)
        messages = " ".join(r.message for r in caplog.records)
        assert "300" in messages  # total frames
        assert "30.0" in messages  # fps


def _insert_mock_embeddings(catalog_db, count=10, dim=768):
    """Insert mock embeddings into catalog_db for index tests."""
    inserted_media: set[str] = set()
    rows = []
    for i in range(count):
        mid = f"vid{i // 5}"
        if mid not in inserted_media:
            catalog_db.insert_media(mid, f"/{mid}.mp4")
            inserted_media.add(mid)
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        blob = vec.tobytes()
        rows.append((mid, i * 60, blob))
    catalog_db.batch_insert_embeddings(rows)
    return rows


def _make_mock_faiss():
    """Create a mock faiss module that tracks write_index calls."""
    mock_faiss = MagicMock()

    # IndexFlatIP returns a mock with ntotal tracking
    index_obj = MagicMock()
    index_obj.d = 768
    index_obj.ntotal = 0

    def add_fn(vectors):
        index_obj.ntotal += len(vectors)

    index_obj.add.side_effect = add_fn
    mock_faiss.IndexFlatIP.return_value = index_obj

    # write_index actually writes a marker file
    def write_index_fn(index, path):
        from pathlib import Path as _Path

        _Path(path).write_text("faiss_index")

    mock_faiss.write_index.side_effect = write_index_fn

    return mock_faiss, index_obj


class TestBuildSearchIndex:
    """Tests for build_search_index."""

    def test_empty_db_no_index(self, catalog_db, tmp_path) -> None:
        """Empty DB returns without creating files."""
        from autopilot.analyze.embeddings import build_search_index

        output = tmp_path / "index.faiss"
        mock_faiss, _ = _make_mock_faiss()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        assert not output.exists()

    def test_small_dataset_creates_index(self, catalog_db, tmp_path) -> None:
        """Insert 10 mock embeddings, build index, verify output file exists."""
        from autopilot.analyze.embeddings import build_search_index

        _insert_mock_embeddings(catalog_db, count=10)
        output = tmp_path / "index.faiss"
        mock_faiss, index_obj = _make_mock_faiss()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        assert output.exists()

    def test_id_mapping_written(self, catalog_db, tmp_path) -> None:
        """Sidecar .ids.json exists with correct entries."""
        import json

        from autopilot.analyze.embeddings import build_search_index

        _insert_mock_embeddings(catalog_db, count=10)
        output = tmp_path / "index.faiss"
        mock_faiss, _ = _make_mock_faiss()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        mapping_path = tmp_path / "index.ids.json"
        assert mapping_path.exists()
        mapping = json.loads(mapping_path.read_text())
        assert len(mapping) == 10
        # Each entry is [media_id, frame_number]
        assert all(len(entry) == 2 for entry in mapping)

    def test_index_vector_count(self, catalog_db, tmp_path) -> None:
        """FAISS index.add called with correct number of vectors."""
        from autopilot.analyze.embeddings import build_search_index

        _insert_mock_embeddings(catalog_db, count=10)
        output = tmp_path / "index.faiss"
        mock_faiss, index_obj = _make_mock_faiss()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        assert index_obj.ntotal == 10

    def test_index_uses_flat_ip(self, catalog_db, tmp_path) -> None:
        """Small dataset (<256) uses IndexFlatIP(768)."""
        from autopilot.analyze.embeddings import build_search_index

        _insert_mock_embeddings(catalog_db, count=10)
        output = tmp_path / "index.faiss"
        mock_faiss, _ = _make_mock_faiss()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            build_search_index(catalog_db, output)

        mock_faiss.IndexFlatIP.assert_called_once_with(768)


def _build_test_index(tmp_path, count=10, dim=768):
    """Build a mock FAISS index and ID mapping for search tests.

    Returns (index_path, mock_faiss, id_mapping, vectors).
    """
    import json

    # Generate random normalized vectors
    rng = np.random.RandomState(99)
    vectors = rng.randn(count, dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Build ID mapping
    id_mapping = [[f"vid{i // 5}", i * 60] for i in range(count)]

    # Write mapping file
    index_path = tmp_path / "index.faiss"
    mapping_path = tmp_path / "index.ids.json"
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f)

    # Create mock faiss with search that returns top_k indices
    mock_faiss = MagicMock()
    mock_index = MagicMock()
    mock_index.ntotal = count

    def search_fn(query_vec, k):
        # Return deterministic results: first k indices
        actual_k = min(k, count)
        distances = np.ones((1, actual_k), dtype=np.float32) * 0.9
        indices = np.arange(actual_k, dtype=np.int64).reshape(1, -1)
        return distances, indices

    mock_index.search.side_effect = search_fn
    mock_faiss.read_index.return_value = mock_index

    return index_path, mock_faiss, id_mapping, vectors


class TestSearchByText:
    """Tests for search_by_text."""

    def test_returns_list_of_tuples(self, tmp_path) -> None:
        """Each result is (str, int, float)."""
        from autopilot.analyze.embeddings import search_by_text

        index_path, mock_faiss, _, _ = _build_test_index(tmp_path)
        mock_model, mock_processor = _make_mock_siglip()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            results = search_by_text(
                "a cat", index_path, (mock_model, mock_processor), top_k=3
            )

        assert isinstance(results, list)
        assert len(results) > 0
        for media_id, frame_num, score in results:
            assert isinstance(media_id, str)
            assert isinstance(frame_num, int)
            assert isinstance(score, float)

    def test_top_k_limits_results(self, tmp_path) -> None:
        """top_k=3 returns at most 3."""
        from autopilot.analyze.embeddings import search_by_text

        index_path, mock_faiss, _, _ = _build_test_index(tmp_path, count=10)
        mock_model, mock_processor = _make_mock_siglip()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            results = search_by_text(
                "a cat", index_path, (mock_model, mock_processor), top_k=3
            )

        assert len(results) <= 3

    def test_text_encoder_called(self, tmp_path) -> None:
        """Model text features method called."""
        from autopilot.analyze.embeddings import search_by_text

        index_path, mock_faiss, _, _ = _build_test_index(tmp_path)
        mock_model, mock_processor = _make_mock_siglip()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            search_by_text("a cat", index_path, (mock_model, mock_processor))

        mock_model.get_text_features.assert_called_once()

    def test_results_contain_valid_media_ids(self, tmp_path) -> None:
        """All media_ids match indexed data."""
        from autopilot.analyze.embeddings import search_by_text

        index_path, mock_faiss, id_mapping, _ = _build_test_index(tmp_path)
        valid_ids = {entry[0] for entry in id_mapping}
        mock_model, mock_processor = _make_mock_siglip()

        with patch.dict(sys.modules, {"faiss": mock_faiss}):
            results = search_by_text(
                "a cat", index_path, (mock_model, mock_processor), top_k=5
            )

        for media_id, _, _ in results:
            assert media_id in valid_ids
