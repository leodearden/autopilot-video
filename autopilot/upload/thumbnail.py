"""Thumbnail extraction and scoring for YouTube uploads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.db import CatalogDB

__all__ = [
    "ThumbnailError",
    "extract_best_thumbnail",
]

logger = logging.getLogger(__name__)


class ThumbnailError(Exception):
    """Raised for any thumbnail extraction error."""


def extract_best_thumbnail(
    narrative_id: str,
    video_path: Path,
    db: CatalogDB,
) -> Path:
    """Extract the best thumbnail frame from a video.

    Scores candidate frames using detection confidence, rule-of-thirds
    composition, and sharpness (Laplacian variance). Optionally uploads
    the thumbnail to YouTube if an upload record exists.

    Args:
        narrative_id: Identifier for the narrative.
        video_path: Path to the video file.
        db: CatalogDB instance for detection data and upload records.

    Returns:
        Path to the saved JPEG thumbnail.

    Raises:
        ThumbnailError: If thumbnail extraction fails.
    """
    raise NotImplementedError
