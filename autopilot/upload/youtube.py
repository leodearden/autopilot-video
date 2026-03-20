"""YouTube video upload via the YouTube Data API v3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.config import YouTubeConfig
    from autopilot.db import CatalogDB

__all__ = [
    "UploadError",
    "upload_video",
]

logger = logging.getLogger(__name__)


class UploadError(Exception):
    """Raised for any YouTube upload error."""


def upload_video(
    narrative_id: str,
    video_path: Path,
    db: CatalogDB,
    config: YouTubeConfig,
) -> str:
    """Upload a video to YouTube and store the result in the catalog DB.

    Args:
        narrative_id: Identifier for the narrative to upload.
        video_path: Path to the rendered video file.
        db: CatalogDB instance for metadata lookup and upload record storage.
        config: YouTubeConfig with credentials path, privacy status, etc.

    Returns:
        The YouTube URL for the uploaded video.

    Raises:
        UploadError: If the upload fails for any reason.
    """
    raise NotImplementedError
