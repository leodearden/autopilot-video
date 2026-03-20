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

# EDL transition type -> OTIO transition type mapping
_TRANSITION_TYPE_MAP = {
    "crossfade": "SMPTE_Dissolve",
    "dissolve": "SMPTE_Dissolve",
    "wipe": "SMPTE_Dissolve",
    # 'cut' is implicit (no Transition object)
}


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


def _insert_transitions(
    track: "otio.schema.Track",
    edl_transitions: list[dict],
    track_num: int,
) -> None:
    """Insert OTIO Transition objects between clips on a track.

    Args:
        track: The OTIO Track to insert transitions into.
        edl_transitions: List of EDL transition dicts.
        track_num: The track number (1-indexed) for filtering transitions.
    """
    import opentimelineio as otio  # type: ignore[import-untyped]

    # Build a list of transitions for this track, indexed by position
    # position is 0-indexed: transition at position N goes between clip N and N+1
    transitions_by_pos: dict[int, dict] = {}
    for trans_data in edl_transitions:
        trans_type = trans_data.get("type", "cut")
        if trans_type == "cut":
            continue  # cuts are implicit
        pos = trans_data.get("position", 0)
        transitions_by_pos[pos] = trans_data

    if not transitions_by_pos:
        return

    # Count existing clips to determine fps for duration conversion
    clips_in_track = [
        item for item in track if isinstance(item, otio.schema.Clip)
    ]
    if not clips_in_track:
        return

    # Use the fps from the first clip's source_range
    fps = clips_in_track[0].source_range.start_time.rate if clips_in_track else _DEFAULT_FPS

    # Insert transitions in reverse order to avoid index shifting
    for pos in sorted(transitions_by_pos.keys(), reverse=True):
        trans_data = transitions_by_pos[pos]
        trans_type = trans_data.get("type", "cut")
        duration_secs = float(trans_data.get("duration", 0.5))

        otio_type = _TRANSITION_TYPE_MAP.get(trans_type, "SMPTE_Dissolve")
        half_dur = otio.opentime.RationalTime.from_seconds(duration_secs / 2.0, fps)

        transition = otio.schema.Transition(
            name=trans_type,
            transition_type=otio_type,
            in_offset=half_dur,
            out_offset=half_dur,
        )

        # Insert after clip at 'position' (which is index position + 1 in the track
        # because we're inserting between clip[pos] and clip[pos+1])
        insert_idx = pos + 1
        if insert_idx <= len(track):
            track.insert(insert_idx, transition)

    return


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

    # Build per-clip metadata lookup dicts
    crop_by_clip: dict[str, str] = {}
    for cm in edl.get("crop_modes", []):
        crop_by_clip[cm["clip_id"]] = cm.get("mode", "")

    audio_by_clip: dict[str, float] = {}
    for au in edl.get("audio_settings", []):
        audio_by_clip[au["clip_id"]] = au.get("level_db", 0.0)

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

            # Attach per-clip metadata
            clip_meta: dict = {}
            if clip_id in crop_by_clip:
                clip_meta["crop_mode"] = crop_by_clip[clip_id]
            if clip_id in audio_by_clip:
                clip_meta["level_db"] = audio_by_clip[clip_id]
            if clip_meta:
                otio_clip.metadata["autopilot"] = clip_meta

            track.append(otio_clip)

        # Insert transitions for this track (applied after all clips are added)
        _insert_transitions(track, edl.get("transitions", []), track_num)

        timeline.tracks.append(track)

    # Attach timeline-level metadata
    tl_meta: dict = {}
    for key in ("titles", "music", "voiceovers", "broll_requests"):
        val = edl.get(key)
        if val:
            tl_meta[key] = val
    if "target_duration_seconds" in edl:
        tl_meta["target_duration_seconds"] = edl["target_duration_seconds"]
    if tl_meta:
        timeline.metadata["autopilot"] = tl_meta

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
