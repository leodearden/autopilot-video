"""Selective video captioning with Qwen2.5-VL-7B.

Provides caption_clip() for on-demand single-clip captioning and
batch_caption() for sampling/bulk captioning across media files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.config import ModelConfig
    from autopilot.db import CatalogDB

__all__ = ["CaptionError", "caption_clip", "batch_caption"]

logger = logging.getLogger(__name__)


class CaptionError(Exception):
    """Raised for all captioning-related failures."""


def _extract_clip_frames(
    video_path: Path,
    start_time: float,
    end_time: float,
    fps: float,
    num_frames: int = 8,
) -> list[Image.Image]:
    """Extract evenly-spaced frames from a video clip segment.

    Args:
        video_path: Path to the video file.
        start_time: Clip start time in seconds.
        end_time: Clip end time in seconds.
        fps: Video frame rate.
        num_frames: Number of frames to extract.

    Returns:
        List of PIL Image objects (RGB).
    """
    raise NotImplementedError


def caption_clip(
    media_id: str,
    video_path: Path,
    start_time: float,
    end_time: float,
    db: CatalogDB,
    scheduler: GPUScheduler,
    config: ModelConfig,
) -> str:
    """Generate a caption for a single video clip segment.

    Args:
        media_id: Unique identifier for the media file.
        video_path: Path to the video file.
        start_time: Clip start time in seconds.
        end_time: Clip end time in seconds.
        db: Catalog database for storing/retrieving captions.
        scheduler: GPU scheduler for model loading.
        config: Model configuration.

    Returns:
        Generated caption string.
    """
    raise NotImplementedError


def batch_caption(
    media_ids: list[str],
    db: CatalogDB,
    scheduler: GPUScheduler,
    config: ModelConfig,
    sample_rate: float = 0.1,
) -> None:
    """Caption clips across multiple media files with sampling.

    Args:
        media_ids: List of media file identifiers.
        db: Catalog database.
        scheduler: GPU scheduler for model loading.
        config: Model configuration.
        sample_rate: Fraction of clips to caption (0.0 to 1.0).
    """
    raise NotImplementedError
