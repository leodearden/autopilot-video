"""OTIO export: convert EDL structures to OpenTimelineIO files.

Provides export_otio() for converting JSON EDL dicts (from edl.py) to
.otio files for NLE review, and detect_otio_changes() for round-trip
change detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.db import CatalogDB

logger = logging.getLogger(__name__)

__all__ = [
    "OtioExportError",
    "export_otio",
    "detect_otio_changes",
]


class OtioExportError(Exception):
    """Raised for all OTIO export failures."""


def _tc_to_rational_time(tc: str, fps: float) -> "otio.opentime.RationalTime":
    """Convert HH:MM:SS.mmm timecode to an OTIO RationalTime.

    Args:
        tc: Timecode in HH:MM:SS.mmm format.
        fps: Frame rate for the RationalTime.

    Returns:
        RationalTime at the given fps.
    """
    import opentimelineio as otio  # type: ignore[import-untyped]

    from autopilot.plan.validator import timecode_to_seconds

    seconds = timecode_to_seconds(tc)
    return otio.opentime.RationalTime.from_seconds(seconds, fps)


_DEFAULT_FPS = 30.0


def _get_media_info(clip_id: str, db: CatalogDB) -> tuple[str, float]:
    """Look up file_path and fps for a clip from the catalog DB.

    Returns:
        (file_path, fps) tuple. Falls back to clip_id and _DEFAULT_FPS
        when the media is not in the catalog.
    """
    media = db.get_media(clip_id)
    if media is None:
        logger.warning("Media not found for clip_id=%s, using fallback", clip_id)
        return clip_id, _DEFAULT_FPS
    file_path = str(media.get("file_path") or clip_id)
    fps = float(media.get("fps") or _DEFAULT_FPS)
    return file_path, fps


def export_otio(edl: dict, output_path: Path, db: CatalogDB) -> Path:
    """Export an EDL structure to an OpenTimelineIO .otio file.

    Args:
        edl: EDL dict with clips, transitions, crop_modes, etc.
        output_path: Destination path for the .otio file.
        db: CatalogDB for media file lookups.

    Returns:
        The output_path on success.

    Raises:
        OtioExportError: If export fails.
    """
    import opentimelineio as otio  # type: ignore[import-untyped]

    clips = edl.get("clips", [])

    # Build timeline
    timeline = otio.schema.Timeline(name="autopilot_edit")

    # Group clips by track number
    tracks_map: dict[int, list[dict]] = {}
    for clip_data in clips:
        track_num = clip_data.get("track", 1)
        tracks_map.setdefault(track_num, []).append(clip_data)

    # Create video tracks
    for track_num in sorted(tracks_map):
        track = otio.schema.Track(
            name=f"V{track_num}",
            kind=otio.schema.TrackKind.Video,
        )

        # Sort clips by in_timecode within each track
        track_clips = sorted(
            tracks_map[track_num],
            key=lambda c: c.get("in_timecode", "00:00:00.000"),
        )

        for clip_data in track_clips:
            clip_id = clip_data["clip_id"]
            in_tc = clip_data["in_timecode"]
            out_tc = clip_data["out_timecode"]

            file_path, fps = _get_media_info(clip_id, db)

            start_time = _tc_to_rational_time(in_tc, fps)
            end_time = _tc_to_rational_time(out_tc, fps)
            duration = otio.opentime.RationalTime(
                end_time.value - start_time.value,
                fps,
            )

            source_range = otio.opentime.TimeRange(
                start_time=start_time,
                duration=duration,
            )

            media_ref = otio.schema.ExternalReference(target_url=file_path)

            otio_clip = otio.schema.Clip(
                name=clip_id,
                source_range=source_range,
                media_reference=media_ref,
            )

            track.append(otio_clip)

        timeline.tracks.append(track)

    # Write to file
    try:
        otio.adapters.write_to_file(timeline, str(output_path))
    except Exception as e:
        raise OtioExportError(f"Failed to write OTIO file: {e}") from e

    return output_path


def detect_otio_changes(otio_path: Path, original_edl: dict) -> dict:
    """Detect whether an .otio file has been modified from the original EDL.

    Args:
        otio_path: Path to the .otio file to check.
        original_edl: The original EDL dict used to generate the .otio file.

    Returns:
        Dict with 'modified' bool and 'changes' list of descriptions.

    Raises:
        OtioExportError: If the .otio file cannot be read.
    """
    raise NotImplementedError
