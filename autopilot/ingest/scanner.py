"""Media file scanner — directory walking and metadata extraction via ffprobe/exiftool."""

from __future__ import annotations

import json
import logging
import subprocess
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


def _run_ffprobe(file_path: Path) -> dict:
    """Run ffprobe on *file_path* and return a dict of extracted metadata.

    Keys: codec, resolution_w, resolution_h, fps, duration_seconds,
    audio_channels, _raw (full parsed JSON for storage).

    Returns an empty dict on any error (subprocess failure, bad JSON, etc.).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.warning("ffprobe failed for %s", file_path)
        return {}

    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        logger.warning("ffprobe returned invalid JSON for %s", file_path)
        return {}

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    # Find the first video and audio streams
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    out: dict = {}

    # Codec — prefer video stream, fall back to audio
    if video_stream:
        out["codec"] = video_stream.get("codec_name")
        out["resolution_w"] = video_stream.get("width")
        out["resolution_h"] = video_stream.get("height")
        # Parse r_frame_rate ("30/1" -> 30.0)
        rfr = video_stream.get("r_frame_rate", "")
        if "/" in rfr:
            num, den = rfr.split("/", 1)
            try:
                out["fps"] = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                pass
    elif audio_stream:
        out["codec"] = audio_stream.get("codec_name")

    # Duration from format
    dur = fmt.get("duration")
    if dur is not None:
        try:
            out["duration_seconds"] = float(dur)
        except (ValueError, TypeError):
            pass

    # Audio channels
    if audio_stream:
        ch = audio_stream.get("channels")
        if ch is not None:
            out["audio_channels"] = int(ch)

    # Store raw data for metadata_json
    out["_raw"] = data

    return out


def _run_exiftool(file_path: Path) -> dict:
    """Run exiftool on *file_path* and return a dict of extracted metadata.

    Keys: created_at (ISO 8601 string), gps_lat, gps_lon.

    Returns an empty dict on any error.
    """
    try:
        result = subprocess.run(
            [
                "exiftool",
                "-j",
                "-n",
                "-CreateDate",
                "-DateTimeOriginal",
                "-GPSLatitude",
                "-GPSLongitude",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        logger.warning("exiftool failed for %s", file_path)
        return {}

    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        logger.warning("exiftool returned invalid JSON for %s", file_path)
        return {}

    if not data or not isinstance(data, list):
        return {}

    info = data[0]
    out: dict = {}

    # Extract creation date — prefer CreateDate, fall back to DateTimeOriginal
    raw_date = info.get("CreateDate") or info.get("DateTimeOriginal")
    if raw_date and isinstance(raw_date, str):
        # Convert 'YYYY:MM:DD HH:MM:SS' to ISO 8601 'YYYY-MM-DDTHH:MM:SS'
        out["created_at"] = raw_date.replace(":", "-", 2).replace(" ", "T", 1)

    # GPS coordinates (numeric with -n flag)
    gps_lat = info.get("GPSLatitude")
    if gps_lat is not None and isinstance(gps_lat, (int, float)):
        out["gps_lat"] = float(gps_lat)

    gps_lon = info.get("GPSLongitude")
    if gps_lon is not None and isinstance(gps_lon, (int, float)):
        out["gps_lon"] = float(gps_lon)

    return out


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
