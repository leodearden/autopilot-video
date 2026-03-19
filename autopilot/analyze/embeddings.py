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
    raise NotImplementedError


def build_search_index(db: CatalogDB, output_path: Path) -> None:
    """Build a FAISS search index from all stored clip embeddings.

    Args:
        db: Catalog database containing clip embeddings.
        output_path: Path to write the FAISS index file.
    """
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError
