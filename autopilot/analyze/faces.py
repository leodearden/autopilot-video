"""Face detection and clustering using InsightFace SCRFD + ArcFace.

Provides detect_faces() for running face detection on video frames with
embedding extraction, and cluster_faces() for DBSCAN-based face grouping.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
    # Idempotency: skip if faces already exist for this media
    existing = db.get_faces_for_media(media_id)
    if existing:
        logger.info("Faces already exist for %s, skipping", media_id)
        return

    # Validate video path before importing cv2
    if not video_path.exists():
        raise FaceDetectionError(f"Video file not found: {video_path}")

    import cv2  # type: ignore[reportMissingImports]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FaceDetectionError(f"Failed to open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Starting face detection for %s (%d frames, %.1f fps)",
            media_id,
            total_frames,
            fps,
        )

        if total_frames <= 0:
            return

        interval = max(1, int(fps))
        frame_indices = list(range(0, total_frames, interval))

        with scheduler.model(config.face_model) as face_model:
            rows: list[tuple[str, int, int, str, bytes | None, int | None]] = []

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        "Failed to read frame %d for %s, skipping",
                        frame_idx,
                        media_id,
                    )
                    continue

                faces = face_model.get(frame)
                for i, face in enumerate(faces):
                    if face.normed_embedding is None:
                        continue
                    bbox_json = json.dumps(face.bbox.tolist())
                    emb_bytes = face.normed_embedding.astype(np.float32).tobytes()
                    rows.append(
                        (media_id, frame_idx, i, bbox_json, emb_bytes, None)
                    )

            if rows:
                with db:
                    db.batch_insert_faces(rows)

        logger.info(
            "Completed face detection for %s: %d faces across %d frames",
            media_id,
            len(rows),
            len(frame_indices),
        )
    finally:
        cap.release()


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
