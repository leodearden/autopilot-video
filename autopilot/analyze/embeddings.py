"""Frame embedding computation and semantic search using SigLIP2.

Provides compute_embeddings() for extracting SigLIP2 vision embeddings at
0.5 FPS, build_search_index() for FAISS IVF index construction,
search_by_text() and search_by_image() for semantic similarity retrieval.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.config import ModelConfig
    from autopilot.db import CatalogDB

__all__ = [
    "EmbeddingError",
    "compute_embeddings",
    "build_search_index",
    "search_by_text",
    "search_by_image",
]

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised for all frame embedding and search index failures."""


def _compute_sample_indices(
    total_frames: int, fps: float, sample_fps: float = 0.5
) -> list[int]:
    """Compute which frame indices to sample for embedding extraction.

    Args:
        total_frames: Total number of frames in the video.
        fps: Video frames per second.
        sample_fps: Target sampling rate in frames per second.

    Returns:
        Sorted list of 0-based frame indices.
    """
    if total_frames <= 0:
        return []
    interval = max(1, int(fps / sample_fps))
    return list(range(0, total_frames, interval))


def compute_embeddings(
    media_id: str,
    video_path: Path,
    db: CatalogDB,
    scheduler: GPUScheduler,
    config: ModelConfig,
    *,
    batch_size: int = 16,
) -> None:
    """Compute SigLIP2 vision embeddings for sampled video frames.

    Args:
        media_id: Unique identifier for the media file.
        video_path: Path to the video file.
        db: Catalog database for storing embeddings.
        scheduler: GPU scheduler for model loading.
        config: Model configuration (clip_model name).
        batch_size: Number of frames to process per batch.
    """
    # Idempotency: skip if embeddings already exist for this media
    existing = db.get_embeddings_for_media(media_id)
    if existing:
        logger.info("Embeddings already exist for %s, skipping", media_id)
        return

    # Validate video path
    if not video_path.exists():
        raise EmbeddingError(f"Video file not found: {video_path}")

    import cv2  # type: ignore[import-not-found]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise EmbeddingError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = _compute_sample_indices(total_frames, fps)

        logger.info(
            "Computing embeddings for %s: %d frames at %.1f fps, %d samples",
            media_id, total_frames, fps, len(sample_indices),
        )

        import numpy as _np  # local alias to avoid TYPE_CHECKING conflict

        with scheduler.model(config.clip_model) as (model_obj, processor):
            rows: list[tuple[str, int, bytes]] = []
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        "Failed to read frame %d for %s, skipping",
                        frame_idx, media_id,
                    )
                    continue

                # BGR -> RGB
                rgb = frame[:, :, ::-1]

                inputs = processor(images=rgb, return_tensors="pt")
                features = model_obj.get_image_features(**inputs)
                # L2-normalize in numpy space
                vec = features.detach().cpu().numpy().astype(_np.float32)
                norm = _np.linalg.norm(vec, axis=-1, keepdims=True)
                if norm.item() > 0:
                    vec = vec / norm
                embedding_blob = vec.tobytes()
                rows.append((media_id, frame_idx, embedding_blob))

            with db:
                db.batch_insert_embeddings(rows)

        logger.info(
            "Completed embeddings for %s: %d embeddings stored",
            media_id, len(rows),
        )
    finally:
        cap.release()


def build_search_index(db: CatalogDB, output_path: Path) -> None:
    """Build a FAISS search index from all stored clip embeddings.

    Args:
        db: Catalog database containing clip embeddings.
        output_path: Path to write the FAISS index file.
    """
    import json

    import faiss  # type: ignore[import-not-found]
    import numpy as _np  # local alias to avoid TYPE_CHECKING conflict

    rows = db.get_all_clip_embeddings()
    if not rows:
        logger.info("No embeddings found, skipping index build")
        return

    # Reconstruct float32 matrix from BLOBs
    dim = 768
    vectors = _np.stack(
        [_np.frombuffer(r["embedding"], dtype=_np.float32) for r in rows]
    )
    n = len(vectors)
    logger.info("Building search index: %d vectors, %d dimensions", n, dim)

    # Build ID mapping: list of [media_id, frame_number]
    id_mapping = [[r["media_id"], r["frame_number"]] for r in rows]

    # Two-tier index strategy
    if n < 256:
        index = faiss.IndexFlatIP(dim)
    else:
        nlist = max(1, int(_np.sqrt(n)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)

    index.add(vectors)
    faiss.write_index(index, str(output_path))

    # Write sidecar ID mapping
    mapping_path = output_path.with_suffix(".ids.json")
    with open(mapping_path, "w") as f:
        json.dump(id_mapping, f)

    logger.info("Search index written to %s (%d vectors)", output_path, n)


def _load_index_and_mapping(
    index_path: Path,
) -> tuple[object, list[list]]:
    """Load a FAISS index and its sidecar ID mapping.

    Args:
        index_path: Path to the FAISS index file.

    Returns:
        (faiss_index, mapping) where mapping is list of [media_id, frame_number].
    """
    import json

    import faiss  # type: ignore[import-not-found]

    index = faiss.read_index(str(index_path))
    mapping_path = index_path.with_suffix(".ids.json")
    with open(mapping_path) as f:
        mapping = json.load(f)
    return index, mapping


def _search_index(
    query_vec,
    index: object,
    mapping: list[list],
    top_k: int,
) -> list[tuple[str, int, float]]:
    """Search FAISS index and map results to (media_id, frame_number, score).

    Args:
        query_vec: Query vector as float32 numpy array of shape (1, dim).
        index: FAISS index object.
        mapping: List of [media_id, frame_number] entries.
        top_k: Maximum results.

    Returns:
        List of (media_id, frame_number, score) tuples.
    """
    distances, indices = index.search(query_vec, top_k)  # type: ignore[union-attr]
    results: list[tuple[str, int, float]] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        media_id, frame_number = mapping[idx]
        results.append((str(media_id), int(frame_number), float(dist)))
    return results


def search_by_text(
    query: str,
    index_path: Path,
    model: tuple,
    *,
    top_k: int = 10,
) -> list[tuple[str, int, float]]:
    """Search the FAISS index using a text query.

    Args:
        query: Text query string.
        index_path: Path to the FAISS index file.
        model: Pre-loaded (model, processor) tuple.
        top_k: Maximum number of results to return.

    Returns:
        List of (media_id, frame_number, score) tuples.
    """
    import numpy as _np  # local alias

    model_obj, processor = model
    index, mapping = _load_index_and_mapping(index_path)

    inputs = processor(text=[query], return_tensors="pt", padding=True)
    features = model_obj.get_text_features(**inputs)
    vec = features.detach().cpu().numpy().astype(_np.float32)
    norm = _np.linalg.norm(vec, axis=-1, keepdims=True)
    if norm.item() > 0:
        vec = vec / norm

    return _search_index(vec, index, mapping, top_k)


def search_by_image(
    image: np.ndarray,
    index_path: Path,
    model: tuple,
    *,
    top_k: int = 10,
) -> list[tuple[str, int, float]]:
    """Search the FAISS index using an image query.

    Args:
        image: Query image as numpy array (H, W, 3) BGR or RGB.
        index_path: Path to the FAISS index file.
        model: Pre-loaded (model, processor) tuple.
        top_k: Maximum number of results to return.

    Returns:
        List of (media_id, frame_number, score) tuples.
    """
    import numpy as _np  # local alias

    model_obj, processor = model
    index, mapping = _load_index_and_mapping(index_path)

    inputs = processor(images=image, return_tensors="pt")
    features = model_obj.get_image_features(**inputs)
    vec = features.detach().cpu().numpy().astype(_np.float32)
    norm = _np.linalg.norm(vec, axis=-1, keepdims=True)
    if norm.item() > 0:
        vec = vec / norm

    return _search_index(vec, index, mapping, top_k)
