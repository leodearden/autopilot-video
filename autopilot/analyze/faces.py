"""Face detection and clustering using InsightFace SCRFD + ArcFace.

Provides detect_faces() for running face detection on video frames with
embedding extraction, and cluster_faces() for DBSCAN-based face grouping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.config import ModelConfig
    from autopilot.db import CatalogDB

__all__ = ["FaceDetectionError", "detect_faces", "cluster_faces"]

logger = logging.getLogger(__name__)


class FaceDetectionError(Exception):
    """Raised for all face detection and clustering failures."""


def detect_faces(
    media_id: str,
    video_path: Path,
    db: CatalogDB,
    scheduler: GPUScheduler,
    config: ModelConfig,
) -> None:
    """Run face detection on a video and store per-frame face data.

    Args:
        media_id: Unique identifier for the media file.
        video_path: Path to the video file.
        db: Catalog database for storing face detections.
        scheduler: GPU scheduler for model loading.
        config: Model configuration (face_model name).
    """
    raise NotImplementedError


def cluster_faces(
    db: CatalogDB,
    *,
    eps: float = 0.5,
    min_samples: int = 3,
) -> None:
    """Cluster all detected faces across media using DBSCAN.

    Args:
        db: Catalog database with face embeddings.
        eps: DBSCAN epsilon parameter (cosine distance threshold).
        min_samples: DBSCAN minimum samples per cluster.
    """
    raise NotImplementedError
