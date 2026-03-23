"""B-roll footage sourcing via Pexels and Pixabay APIs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.source import BrollRequest

__all__ = [
    "BrollError",
    "source_broll",
]

logger = logging.getLogger(__name__)

_PEXELS_API_URL = "https://api.pexels.com/videos/search"
_PIXABAY_API_URL = "https://pixabay.com/api/videos/"


class BrollError(Exception):
    """Raised for any B-roll sourcing error."""


def source_broll(request: BrollRequest, output_dir: Path) -> list[Path] | None:
    """Source B-roll footage matching the request description.

    Tries Pexels first, falls back to Pixabay. Downloads top-3 results
    to let the caller select the best match.

    Args:
        request: BrollRequest with description, duration, start_time.
        output_dir: Directory where downloaded files are saved.

    Returns:
        List of Paths to downloaded video files (up to 3), or None if
        no results from either API.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try Pexels first
    results = _search_pexels(request, output_dir)
    if results:
        return results

    # Fall back to Pixabay
    results = _search_pixabay(request, output_dir)
    if results:
        return results

    logger.warning("No B-roll found for description=%r from any source", request.description)
    return None


def _search_pexels(request: BrollRequest, output_dir: Path) -> list[Path] | None:
    """Search Pexels for B-roll video footage.

    Args:
        request: BrollRequest with description to search for.
        output_dir: Directory to download results into.

    Returns:
        List of Paths to downloaded videos, or None if no results.
    """
    api_key = os.environ.get("PEXELS_API_KEY")
    if not api_key:
        logger.warning("PEXELS_API_KEY not set; skipping Pexels search")
        return None

    try:
        import requests as _requests
    except ImportError:
        logger.warning("requests library not installed; cannot search Pexels")
        return None

    try:
        logger.info("Searching Pexels for B-roll: %r", request.description)

        response = _requests.get(
            _PEXELS_API_URL,
            headers={"Authorization": api_key},
            params={
                "query": request.description,
                "per_page": 3,
                "orientation": "landscape",
            },
            timeout=(10, 120),
        )
        response.raise_for_status()
        data = response.json()

        videos = data.get("videos", [])
        if not videos:
            logger.info("No Pexels results for %r", request.description)
            return None

        # Download top-3 results
        downloaded = []
        for video in videos[:3]:
            video_files = video.get("video_files", [])
            if not video_files:
                continue

            # Pick the best quality file (prefer HD)
            best = _select_best_video_file(video_files)
            if not best or not best.get("link"):
                continue

            dl_response = _requests.get(best["link"], stream=True, timeout=(10, 300))
            dl_response.raise_for_status()

            filename = f"pexels_{video['id']}.mp4"
            output_path = output_dir / filename
            with open(output_path, "wb") as f:
                for chunk in dl_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(output_path)

            logger.info("Downloaded Pexels video %s to %s", video["id"], output_path)

        return downloaded if downloaded else None

    except Exception as e:
        logger.warning("Pexels search failed: %s", e)
        return None


def _search_pixabay(request: BrollRequest, output_dir: Path) -> list[Path] | None:
    """Search Pixabay for B-roll video footage (fallback).

    Args:
        request: BrollRequest with description to search for.
        output_dir: Directory to download results into.

    Returns:
        List of Paths to downloaded videos, or None if no results.
    """
    api_key = os.environ.get("PIXABAY_API_KEY")
    if not api_key:
        logger.warning("PIXABAY_API_KEY not set; skipping Pixabay search")
        return None

    try:
        import requests as _requests
    except ImportError:
        logger.warning("requests library not installed; cannot search Pixabay")
        return None

    try:
        logger.info("Searching Pixabay for B-roll: %r", request.description)

        response = _requests.get(
            _PIXABAY_API_URL,
            params={
                "key": api_key,
                "q": request.description,
                "video_type": "film",
                "per_page": 3,
            },
            timeout=(10, 120),
        )
        response.raise_for_status()
        data = response.json()

        hits = data.get("hits", [])
        if not hits:
            logger.info("No Pixabay results for %r", request.description)
            return None

        # Download results
        downloaded = []
        for hit in hits[:3]:
            videos = hit.get("videos", {})
            medium = videos.get("medium", {})
            url = medium.get("url")
            if not url:
                continue

            dl_response = _requests.get(url, stream=True, timeout=(10, 300))
            dl_response.raise_for_status()

            filename = f"pixabay_{hit['id']}.mp4"
            output_path = output_dir / filename
            with open(output_path, "wb") as f:
                for chunk in dl_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(output_path)

            logger.info("Downloaded Pixabay video %s to %s", hit["id"], output_path)

        return downloaded if downloaded else None

    except Exception as e:
        logger.warning("Pixabay search failed: %s", e)
        return None


def _select_best_video_file(video_files: list[dict]) -> dict | None:
    """Select the best quality video file from Pexels results.

    Prefers HD quality with width >= 1280.

    Args:
        video_files: List of video file dicts from Pexels API.

    Returns:
        Best video file dict, or None if no suitable file.
    """
    if not video_files:
        return None

    # Sort by width descending, prefer HD quality
    sorted_files = sorted(
        video_files,
        key=lambda f: (f.get("quality") == "hd", f.get("width", 0)),
        reverse=True,
    )
    return sorted_files[0]
