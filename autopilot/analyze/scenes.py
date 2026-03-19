"""Shot boundary detection using TransNetV2 (GPU) and PySceneDetect (CPU fallback).

Provides detect_shots() for detecting shot/scene boundaries in video files.
TransNetV2 is tried first as the primary GPU-based method, with PySceneDetect
as a CPU-based fallback if TransNetV2 fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.db import CatalogDB

__all__ = ["ShotDetectionError", "detect_shots"]

logger = logging.getLogger(__name__)

# TransNetV2 expects frames downsampled to this resolution
TRANSNETV2_INPUT_WIDTH = 48
TRANSNETV2_INPUT_HEIGHT = 27

# PySceneDetect AdaptiveDetector threshold
PYSCENEDETECT_THRESHOLD = 27.0


class ShotDetectionError(Exception):
    """Raised for all shot boundary detection failures."""


def _read_and_downsample_frames(
    video_path: Path, cv2_module: object
) -> tuple[np.ndarray, float, int]:
    """Read all frames from a video and downsample to TransNetV2 input size.

    Args:
        video_path: Path to the video file.
        cv2_module: The cv2 module (passed to enable deferred import testing).

    Returns:
        Tuple of (frames array shape (N, 27, 48, 3), fps, total_frames).

    Raises:
        ShotDetectionError: If the video cannot be opened.
    """
    cv2 = cv2_module
    cap = cv2.VideoCapture(str(video_path))  # type: ignore[union-attr]
    if not cap.isOpened():
        cap.release()
        raise ShotDetectionError(f"Failed to open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))  # type: ignore[union-attr]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore[union-attr]

        frames = []
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            resized = cv2.resize(  # type: ignore[union-attr]
                frame,
                (TRANSNETV2_INPUT_WIDTH, TRANSNETV2_INPUT_HEIGHT),
                interpolation=cv2.INTER_AREA,  # type: ignore[union-attr]
            )
            frames.append(resized)

        if frames:
            arr = np.stack(frames, axis=0)
        else:
            arr = np.empty(
                (0, TRANSNETV2_INPUT_HEIGHT, TRANSNETV2_INPUT_WIDTH, 3),
                dtype=np.uint8,
            )
        return arr, fps, total_frames
    finally:
        cap.release()


def detect_shots(
    media_id: str,
    video_path: Path,
    db: CatalogDB,
    scheduler: GPUScheduler,
) -> None:
    """Detect shot boundaries in a video file.

    Tries TransNetV2 (GPU) first, falls back to PySceneDetect (CPU)
    on failure. Results are stored in the catalog database.

    Args:
        media_id: Unique identifier for the media file.
        video_path: Path to the video file.
        db: Catalog database for storing boundaries.
        scheduler: GPU scheduler for model loading.

    Raises:
        ShotDetectionError: If both detection methods fail.
    """
    # Idempotency: skip if boundaries already exist for this media
    existing = db.get_boundaries(media_id)
    if existing:
        logger.info("Boundaries already exist for %s, skipping", media_id)
        return

    # Validate video path before heavy imports
    if not video_path.exists():
        raise ShotDetectionError(f"Video file not found: {video_path}")

    raise NotImplementedError
