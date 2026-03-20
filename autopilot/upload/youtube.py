"""YouTube video upload via the YouTube Data API v3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.oauth2.credentials import Credentials as _Credentials

    from autopilot.config import YouTubeConfig
    from autopilot.db import CatalogDB

__all__ = [
    "UploadError",
    "upload_video",
]

logger = logging.getLogger(__name__)


class UploadError(Exception):
    """Raised for any YouTube upload error."""


def _load_credentials(credentials_path: Path) -> _Credentials:
    """Load and optionally refresh OAuth2 credentials from file.

    Args:
        credentials_path: Path to the OAuth2 credentials JSON file.

    Returns:
        Valid Credentials instance.

    Raises:
        UploadError: If the file is missing or credentials cannot be loaded.
    """
    if not credentials_path.exists():
        msg = f"YouTube credentials file not found: {credentials_path}"
        raise UploadError(msg)

    from google.auth.transport.requests import Request  # lazy import
    from google.oauth2.credentials import Credentials  # lazy import

    try:
        creds = Credentials.from_authorized_user_file(str(credentials_path))
    except Exception as exc:
        msg = f"Failed to load YouTube credentials: {exc}"
        raise UploadError(msg) from exc

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception as exc:
            msg = f"Failed to refresh YouTube credentials: {exc}"
            raise UploadError(msg) from exc

    return creds


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
