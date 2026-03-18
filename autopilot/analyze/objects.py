"""Object detection and tracking using YOLO11x with ByteTrack.

Provides detect_objects() for running YOLO-based object detection on video
frames with persistent ByteTrack tracking, optional frame interpolation
for dense output, and transactional DB storage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.config import ModelConfig
    from autopilot.db import CatalogDB

__all__ = ["DetectionError", "detect_objects"]

logger = logging.getLogger(__name__)


class DetectionError(Exception):
    """Raised for all object detection and tracking failures."""


def _compute_frame_indices(
    total_frames: int, fps: float, sample_every_n: int, sparse: bool
) -> list[int]:
    """Compute which frame indices to run detection on.

    Args:
        total_frames: Total number of frames in the video.
        fps: Video frames per second.
        sample_every_n: Sample interval for dense mode.
        sparse: If True, sample at ~1fps; if False, use sample_every_n.

    Returns:
        Sorted list of 0-based frame indices.
    """
    if total_frames <= 0:
        return []
    if sparse:
        interval = max(1, int(fps))
    else:
        interval = sample_every_n
    return list(range(0, total_frames, interval))


def detect_objects(
    media_id: str,
    video_path: Path,
    db: CatalogDB,
    scheduler: GPUScheduler,
    config: ModelConfig,
    *,
    batch_size: int = 16,
    sparse: bool = False,
) -> None:
    """Run YOLO object detection with ByteTrack tracking on a video.

    Args:
        media_id: Unique identifier for the media file.
        video_path: Path to the video file.
        db: Catalog database for storing detections.
        scheduler: GPU scheduler for model loading.
        config: Model configuration (yolo_variant, yolo_sample_every_n_frames).
        batch_size: Number of frames to read from disk per batch.
        sparse: If True, detect at 1fps; if False, detect every N frames and interpolate.
    """
    pass
