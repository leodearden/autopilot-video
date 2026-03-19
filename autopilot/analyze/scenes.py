"""Shot boundary detection using TransNetV2 (GPU) and PySceneDetect (CPU fallback).

Provides detect_shots() for detecting shot/scene boundaries in video files.
TransNetV2 is tried first as the primary GPU-based method, with PySceneDetect
as a CPU-based fallback if TransNetV2 fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

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
    raise NotImplementedError
