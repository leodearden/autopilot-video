"""Media file scanner — directory walking and metadata extraction via ffprobe/exiftool."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        # Video
        ".mp4",
        ".mov",
        # Audio
        ".wav",
        ".mp3",
        ".aac",
        # Image
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }
)


@dataclass
class MediaFile:
    """Structured metadata for a single media file.

    Maps directly to CatalogDB.insert_media() parameters.
    Only ``file_path`` is required; all other fields default to None.
    """

    file_path: Path
    codec: str | None = None
    resolution_w: int | None = None
    resolution_h: int | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    created_at: str | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None
    audio_channels: int | None = None
    sha256_prefix: str | None = None
    metadata_json: str | None = None


def _probe_file(file_path: Path) -> MediaFile:
    """Extract metadata from a single file (stub — returns minimal MediaFile)."""
    return MediaFile(file_path=file_path)


def scan_directory(
    input_dir: Path,
    *,
    max_workers: int | None = None,
) -> list[MediaFile]:
    """Walk *input_dir* recursively and return :class:`MediaFile` for every supported file.

    Parameters
    ----------
    input_dir:
        Root directory to scan.
    max_workers:
        Number of parallel workers for metadata probing (default: CPU count).
    """
    files = sorted(
        p
        for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        return []

    results: list[MediaFile] = []
    for f in files:
        try:
            results.append(_probe_file(f))
        except Exception:
            logger.warning("Failed to probe %s, skipping", f)
    return results
