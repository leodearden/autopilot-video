"""Media file scanner — directory walking and metadata extraction via ffprobe/exiftool."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


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
