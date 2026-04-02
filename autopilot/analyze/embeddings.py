"""Frame embeddings and search index using SigLIP 2 vision/text encoders.

Provides compute_embeddings() for extracting per-frame CLIP-style embeddings,
build_search_index() for constructing a FAISS index, and search_by_text() /
search_by_image() for querying the index.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.config import ModelConfig
    from autopilot.db import CatalogDB

__all__ = [
    "EmbeddingError",
    "build_search_index",
    "compute_embeddings",
    "search_by_image",
    "search_by_text",
]

logger = logging.getLogger(__name__)

# Embedding dimension for SigLIP 2 so400m
EMBEDDING_DIM = 768

# Default sampling rate in frames per second
SAMPLE_FPS = 0.5


class EmbeddingError(Exception):
    """Raised for all embedding computation and indexing failures."""


def _compute_sample_indices(total_frames: int, fps: float, sample_fps: float) -> list[int]:
    """Compute which frame indices to sample for embedding extraction.

    Args:
        total_frames: Total number of frames in the video.
        fps: Video frames per second.
        sample_fps: Target sampling rate in frames per second.

    Returns:
        Sorted list of 0-based frame indices.
    """
    if total_frames <= 0 or fps <= 0:
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
    sample_fps: float = SAMPLE_FPS,
    batch_size: int = 16,
) -> None:
    """Extract SigLIP 2 vision embeddings from video frames and store in DB.

    Samples frames at *sample_fps* (default 0.5 FPS), computes 768-d L2-normalized
    embeddings via the SigLIP 2 vision encoder, and stores them as raw float32
    BLOBs in the clip_embeddings table.

    Args:
        media_id: Unique identifier for the media file.
        video_path: Path to the video file.
        db: Catalog database for storing embeddings.
        scheduler: GPU scheduler for model loading.
        config: Model configuration (clip_model).
        sample_fps: Target frames per second to sample (default 0.5).
        batch_size: Number of frames to process per batch.
    """
    # Idempotency: skip if embeddings already exist for this media
    if db.has_embeddings(media_id):
        logger.info("Embeddings already exist for %s, skipping", media_id)
        return

    # Validate video path before importing cv2
    if not video_path.exists():
        raise EmbeddingError(f"Video file not found: {video_path}")

    import cv2  # type: ignore[reportMissingImports]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise EmbeddingError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Starting embedding extraction for %s (%d frames, %.1f fps, sample_fps=%.2f)",
            media_id,
            total_frames,
            fps,
            sample_fps,
        )

        frame_indices = _compute_sample_indices(total_frames, fps, sample_fps)

        if not frame_indices:
            logger.info("No frames to process for %s", media_id)
            return

        # Load SigLIP model via scheduler — entry stores (model, processor) tuple
        with scheduler.model(config.clip_model) as model_tuple:
            model, processor = model_tuple

            rows: list[tuple[str, int, bytes]] = []

            for batch_start in range(0, len(frame_indices), batch_size):
                batch_indices = frame_indices[batch_start : batch_start + batch_size]
                batch_frames = []
                batch_frame_nums = []

                for frame_idx in batch_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(
                            "Failed to read frame %d for %s, skipping",
                            frame_idx,
                            media_id,
                        )
                        continue
                    # Convert BGR to RGB for the model
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    batch_frames.append(frame_rgb)
                    batch_frame_nums.append(frame_idx)

                if not batch_frames:
                    continue

                import torch  # type: ignore[reportMissingImports]
                from PIL import Image  # type: ignore[reportMissingImports]

                pil_images = [Image.fromarray(f) for f in batch_frames]
                inputs = processor(images=pil_images, return_tensors="pt", padding=True)

                # Move inputs to same device as model
                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.get_image_features(**inputs)

                # L2-normalize embeddings
                embeddings = outputs.cpu().numpy().astype(np.float32)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                embeddings = embeddings / norms

                for i, frame_num in enumerate(batch_frame_nums):
                    embedding_blob = embeddings[i].tobytes()
                    rows.append((media_id, frame_num, embedding_blob))

                # Batch-insert after each read-batch for bounded memory
                if rows:
                    with db:
                        db.batch_insert_embeddings(rows)
                    rows = []

            # Insert any remaining rows
            if rows:
                with db:
                    db.batch_insert_embeddings(rows)

        logger.info(
            "Completed embedding extraction for %s: %d frames embedded",
            media_id,
            len(frame_indices),
        )
    finally:
        cap.release()


def build_search_index(
    db: CatalogDB,
    output_path: Path,
) -> None:
    """Build a FAISS search index from all clip embeddings in the database.

    Uses a two-tier strategy: IndexFlatIP for <256 vectors (exact search),
    IndexIVFFlat for >=256 vectors (approximate search). Embeddings are assumed
    to be L2-normalized, so inner product equals cosine similarity.

    A JSON sidecar file (.ids.json) is written alongside the index containing
    the mapping from FAISS integer IDs to (media_id, frame_number) pairs.

    Args:
        db: Catalog database containing clip embeddings.
        output_path: Path where the FAISS index file will be written.
    """
    import faiss  # type: ignore[reportMissingImports]

    all_rows = db.get_all_clip_embeddings()

    if not all_rows:
        logger.info("No embeddings found in database, skipping index build")
        return

    # Reconstruct embedding vectors from BLOBs
    vectors = []
    id_mapping = []  # list of [media_id, frame_number]

    for row in all_rows:
        embedding = np.frombuffer(cast(bytes, row["embedding"]), dtype=np.float32)
        vectors.append(embedding)
        id_mapping.append([row["media_id"], row["frame_number"]])

    matrix = np.stack(vectors).astype(np.float32)
    n_vectors, dim = matrix.shape

    logger.info(
        "Building FAISS index: %d vectors, %d dimensions",
        n_vectors,
        dim,
    )

    if n_vectors < 256:
        # Small dataset: exact search with flat index
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)  # type: ignore[arg-type]
    else:
        # Larger dataset: IVF for approximate search
        n_clusters = min(int(np.sqrt(n_vectors)), n_vectors)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(matrix)  # type: ignore[arg-type]
        index.add(matrix)  # type: ignore[arg-type]

    # Write the FAISS index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))

    # Write the ID mapping sidecar
    sidecar_path = output_path.with_suffix(output_path.suffix + ".ids.json")
    with open(sidecar_path, "w") as f:
        json.dump(id_mapping, f)

    logger.info(
        "FAISS index written to %s (%d vectors), sidecar at %s",
        output_path,
        n_vectors,
        sidecar_path,
    )


def search_by_text(
    query: str,
    index_path: Path,
    model: tuple,
    *,
    top_k: int = 10,
) -> list[dict]:
    """Search the FAISS index using a text query via SigLIP text encoder.

    Args:
        query: Text query string.
        index_path: Path to the FAISS index file.
        model: Pre-loaded (model, processor) tuple from the scheduler.
        top_k: Number of top results to return.

    Returns:
        List of dicts with keys: media_id, frame_number, score.
    """
    import faiss  # type: ignore[reportMissingImports]
    import torch  # type: ignore[reportMissingImports]

    siglip_model, processor = model

    # Encode the text query
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    device = next(siglip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_features = siglip_model.get_text_features(**inputs)

    query_vec = text_features.cpu().numpy().astype(np.float32)
    # L2-normalize
    norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    query_vec = query_vec / norm

    # Load the FAISS index and ID mapping
    index = faiss.read_index(str(index_path))
    sidecar_path = index_path.with_suffix(index_path.suffix + ".ids.json")
    with open(sidecar_path) as f:
        id_mapping = json.load(f)

    # Search
    scores, indices = index.search(query_vec, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue  # FAISS returns -1 for unfilled slots
        media_id, frame_number = id_mapping[idx]
        results.append(
            {
                "media_id": media_id,
                "frame_number": frame_number,
                "score": float(score),
            }
        )
    return results


def search_by_image(
    image: np.ndarray,
    index_path: Path,
    model: tuple,
    *,
    top_k: int = 10,
) -> list[dict]:
    """Search the FAISS index using an image query via SigLIP vision encoder.

    Args:
        image: Input image as a numpy array (RGB, HWC).
        index_path: Path to the FAISS index file.
        model: Pre-loaded (model, processor) tuple from the scheduler.
        top_k: Number of top results to return.

    Returns:
        List of dicts with keys: media_id, frame_number, score.
    """
    import faiss  # type: ignore[reportMissingImports]
    import torch  # type: ignore[reportMissingImports]
    from PIL import Image  # type: ignore[reportMissingImports]

    siglip_model, processor = model

    # Convert numpy array to PIL Image for the processor
    pil_image = Image.fromarray(image)
    inputs = processor(images=[pil_image], return_tensors="pt", padding=True)
    device = next(siglip_model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = siglip_model.get_image_features(**inputs)

    query_vec = image_features.cpu().numpy().astype(np.float32)
    # L2-normalize
    norm = np.linalg.norm(query_vec, axis=1, keepdims=True)
    norm = np.where(norm == 0, 1.0, norm)
    query_vec = query_vec / norm

    # Load the FAISS index and ID mapping
    index = faiss.read_index(str(index_path))
    sidecar_path = index_path.with_suffix(index_path.suffix + ".ids.json")
    with open(sidecar_path) as f:
        id_mapping = json.load(f)

    # Search
    scores, indices = index.search(query_vec, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        media_id, frame_number = id_mapping[idx]
        results.append(
            {
                "media_id": media_id,
                "frame_number": frame_number,
                "score": float(score),
            }
        )
    return results
