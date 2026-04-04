"""YouTube video upload via the YouTube Data API v3."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
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


def _build_upload_metadata(
    narrative_id: str,
    db: CatalogDB,
    config: YouTubeConfig,
) -> dict:
    """Build YouTube API upload metadata from catalog data.

    Args:
        narrative_id: Identifier for the narrative.
        db: CatalogDB instance for metadata lookup.
        config: YouTubeConfig with privacy and category settings.

    Returns:
        Dict with 'snippet' and 'status' keys for YouTube API body.
    """
    narrative = db.get_narrative(narrative_id)
    title = narrative["title"] if narrative else narrative_id
    description = str(narrative.get("description", "")) if narrative else ""

    # Enrich description with script content if available
    script_row = db.get_narrative_script(narrative_id)
    if script_row and script_row.get("script_json"):
        try:
            script_data = json.loads(str(script_row["script_json"]))
            scenes = script_data.get("scenes", [])
            narrations = [s.get("narration", "") for s in scenes if s.get("narration")]
            if narrations:
                description += "\n\n" + " ".join(narrations)
        except (json.JSONDecodeError, TypeError):
            pass

    # Build tags from activity cluster labels
    tags: list[str] = []
    if narrative and narrative.get("activity_cluster_ids_json"):
        try:
            cluster_ids = json.loads(str(narrative["activity_cluster_ids_json"]))
        except (json.JSONDecodeError, TypeError):
            cluster_ids = []
        clusters = db.get_activity_clusters()
        cluster_map = {c["cluster_id"]: c for c in clusters}
        for cid in cluster_ids:
            cluster = cluster_map.get(cid)
            if cluster and cluster.get("label"):
                tags.append(str(cluster["label"]))

    # Add unique object class names from detections
    class_names: set[str] = set()
    # Get all media files to collect detections across the project
    media_files = db.list_all_media()
    for mf in media_files:
        detections = db.get_detections_for_media(str(mf["id"]))
        for det_row in detections:
            det_json = det_row.get("detections_json")
            if det_json:
                try:
                    dets = json.loads(str(det_json))
                    for d in dets:
                        if d.get("class"):
                            class_names.add(d["class"])
                except (json.JSONDecodeError, TypeError):
                    pass
    tags.extend(sorted(class_names))

    return {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": config.default_category,
        },
        "status": {
            "privacyStatus": config.privacy_status,
        },
    }


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
    import time  # noqa: E402

    from googleapiclient.discovery import build  # lazy import
    from googleapiclient.http import MediaFileUpload  # lazy import

    # Input validation
    narrative = db.get_narrative(narrative_id)
    if narrative is None:
        msg = f"Narrative not found: {narrative_id}"
        raise UploadError(msg)

    if not video_path.exists():
        msg = f"Video file not found: {video_path}"
        raise UploadError(msg)

    creds = _load_credentials(config.credentials_path)
    metadata = _build_upload_metadata(narrative_id, db, config)

    youtube = build("youtube", "v3", credentials=creds)
    media = MediaFileUpload(str(video_path), resumable=True)

    request = youtube.videos().insert(
        part="snippet,status",
        body=metadata,
        media_body=media,
    )

    max_retries = 3
    response = None
    for attempt in range(max_retries):
        try:
            while response is None:
                _, response = request.next_chunk()
            break  # Upload succeeded; exit retry loop
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning(
                    "Upload attempt %d failed: %s. Retrying in %ds...",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                msg = f"YouTube upload failed: {exc}"
                raise UploadError(msg) from exc

    if response is None:
        msg = "YouTube upload failed: no response received"
        raise UploadError(msg)
    video_id = response["id"]
    youtube_url = f"https://youtu.be/{video_id}"

    db.insert_upload(
        narrative_id,
        youtube_video_id=video_id,
        youtube_url=youtube_url,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
        privacy_status=config.privacy_status,
    )
    logger.info("Uploaded %s -> %s", narrative_id, youtube_url)

    return youtube_url
