"""Thumbnail extraction and scoring for YouTube uploads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from autopilot.db import CatalogDB

__all__ = [
    "ThumbnailError",
    "extract_best_thumbnail",
]

logger = logging.getLogger(__name__)

# Scoring weights (must sum to 1.0)
_WEIGHT_SHARPNESS = 0.3
_WEIGHT_THIRDS = 0.3
_WEIGHT_CONFIDENCE = 0.4


class ThumbnailError(Exception):
    """Raised for any thumbnail extraction error."""


def _sharpness_score(frame: NDArray) -> float:
    """Compute sharpness score using Laplacian variance.

    Higher variance means sharper image.
    """
    import cv2  # lazy import

    gray = cv2.Laplacian(frame, cv2.CV_64F)
    variance = float(gray.var())
    # Normalize: typical Laplacian variances range 0-2000+
    # Cap at 1.0 for scores above 500
    return min(variance / 500.0, 1.0)


def _rule_of_thirds_score(
    frame_shape: tuple[int, ...],
    detections: list[dict],
) -> float:
    """Score how close detection centers are to rule-of-thirds intersections.

    Args:
        frame_shape: (height, width, channels) of the frame.
        detections: List of detection dicts with 'bbox' [x1, y1, x2, y2].

    Returns:
        Score between 0.0 and 1.0 (higher = closer to thirds grid).
    """
    if not detections:
        return 0.0

    h, w = frame_shape[0], frame_shape[1]
    # Four intersection points of rule-of-thirds grid
    intersections = [
        (w / 3, h / 3),
        (2 * w / 3, h / 3),
        (w / 3, 2 * h / 3),
        (2 * w / 3, 2 * h / 3),
    ]

    max_dist = np.sqrt((w / 3) ** 2 + (h / 3) ** 2)
    best_score = 0.0

    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        # Find min distance to any intersection point
        min_dist = min(
            np.sqrt((cx - ix) ** 2 + (cy - iy) ** 2)
            for ix, iy in intersections
        )
        score = max(0.0, 1.0 - min_dist / max_dist)
        best_score = max(best_score, score)

    return best_score


def _detection_confidence_score(detections: list[dict]) -> float:
    """Return the max detection confidence from a list of detections.

    Args:
        detections: List of detection dicts with 'confidence' field.

    Returns:
        Max confidence value, or 0.0 if empty.
    """
    if not detections:
        return 0.0
    return max(d.get("confidence", 0.0) for d in detections)


def _combined_score(
    sharpness: float,
    thirds: float,
    confidence: float,
) -> float:
    """Compute weighted combination of the three scoring metrics.

    Args:
        sharpness: Sharpness score (0-1).
        thirds: Rule-of-thirds score (0-1).
        confidence: Detection confidence score (0-1).

    Returns:
        Weighted combined score.
    """
    return (
        _WEIGHT_SHARPNESS * sharpness
        + _WEIGHT_THIRDS * thirds
        + _WEIGHT_CONFIDENCE * confidence
    )


def _extract_best_frame(
    video_path: Path,
    detections: list[dict],
) -> Path | None:
    """Extract the best-scoring frame from a video and save as JPEG.

    Samples frames at ~1fps intervals, scores each using combined metrics,
    and saves the highest-scoring frame as a JPEG file.

    Args:
        video_path: Path to the video file.
        detections: List of detection dicts for scoring.

    Returns:
        Path to the saved JPEG, or None if no frames could be read.
    """
    import cv2  # lazy import

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(fps))  # sample at ~1fps

    best_score = -1.0
    best_frame = None

    for frame_idx in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        sharpness = _sharpness_score(frame)
        thirds = _rule_of_thirds_score(frame.shape, detections)
        confidence = _detection_confidence_score(detections)
        score = _combined_score(sharpness, thirds, confidence)

        if score > best_score:
            best_score = score
            best_frame = frame.copy() if hasattr(frame, "copy") else frame

    cap.release()

    if best_frame is None:
        return None

    thumb_path = video_path.parent / f"{video_path.stem}_thumbnail.jpg"
    cv2.imwrite(str(thumb_path), best_frame)
    return thumb_path


def _get_credentials_path() -> Path:
    """Return the default YouTube OAuth2 credentials path."""
    return Path("~/.config/autopilot/youtube_oauth.json").expanduser()


def _upload_thumbnail_to_youtube(
    thumb_path: Path,
    youtube_video_id: str,
    credentials_path: Path,
) -> None:
    """Upload a thumbnail image to YouTube via the thumbnails.set API.

    Args:
        thumb_path: Path to the JPEG thumbnail file.
        youtube_video_id: YouTube video ID to set the thumbnail for.
        credentials_path: Path to the OAuth2 credentials file.
    """
    from google.auth.transport.requests import Request  # lazy import
    from google.oauth2.credentials import Credentials  # lazy import
    from googleapiclient.discovery import build  # lazy import
    from googleapiclient.http import MediaFileUpload  # lazy import

    creds = Credentials.from_authorized_user_file(str(credentials_path))
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    youtube = build("youtube", "v3", credentials=creds)
    youtube.thumbnails().set(
        videoId=youtube_video_id,
        media_body=MediaFileUpload(
            str(thumb_path), mimetype="image/jpeg"
        ),
    ).execute()


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
    # Gather detections from all media files for scoring
    import json

    all_detections: list[dict] = []
    media_files = db.list_all_media()
    for mf in media_files:
        dets = db.get_detections_for_media(str(mf["id"]))
        for det_row in dets:
            det_json = det_row.get("detections_json")
            if det_json:
                try:
                    parsed = json.loads(str(det_json))
                    all_detections.extend(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass

    thumb_path = _extract_best_frame(video_path, all_detections)
    if thumb_path is None:
        msg = f"Could not extract any frames from {video_path}"
        raise ThumbnailError(msg)

    # Optionally upload thumbnail to YouTube
    upload_rec = db.get_upload(narrative_id)
    if upload_rec and upload_rec.get("youtube_video_id"):
        credentials_path = _get_credentials_path()
        try:
            _upload_thumbnail_to_youtube(
                thumb_path,
                str(upload_rec["youtube_video_id"]),
                credentials_path,
            )
            logger.info(
                "Uploaded thumbnail for %s to YouTube", narrative_id
            )
        except Exception:
            logger.warning(
                "Failed to upload thumbnail for %s to YouTube",
                narrative_id,
                exc_info=True,
            )

    return thumb_path
