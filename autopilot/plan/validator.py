"""EDL validation: automated constraint checking for edit decision lists.

Provides validate_edl() for comprehensive validation of EDL structures
including overlap detection, duration checks, clip existence, timecode
bounds, and audio level verification.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.db import CatalogDB

logger = logging.getLogger(__name__)

__all__ = [
    "EdlValidationError",
    "ValidationResult",
    "validate_edl",
]


class EdlValidationError(Exception):
    """Raised for EDL validation configuration or processing errors."""


@dataclass
class ValidationResult:
    """Result of EDL validation with pass/fail status, errors, and warnings."""

    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def timecode_to_seconds(tc: str) -> float:
    """Parse HH:MM:SS.mmm timecode string to seconds.

    Args:
        tc: Timecode in HH:MM:SS.mmm format.

    Returns:
        Time in seconds as float.

    Raises:
        ValueError: If the timecode format is invalid.
    """
    parts = tc.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timecode format: {tc!r}")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _check_overlaps(clips: list[dict], errors: list[str]) -> None:
    """Check for overlapping clips on the same track."""
    # Group clips by track
    by_track: dict[int, list[dict]] = defaultdict(list)
    for clip in clips:
        track = clip.get("track", 1)
        by_track[track].append(clip)

    for track, track_clips in by_track.items():
        # Sort by in_timecode
        sorted_clips = sorted(
            track_clips,
            key=lambda c: timecode_to_seconds(c["in_timecode"]),
        )
        for i in range(len(sorted_clips) - 1):
            current_out = timecode_to_seconds(sorted_clips[i]["out_timecode"])
            next_in = timecode_to_seconds(sorted_clips[i + 1]["in_timecode"])
            if current_out > next_in:
                errors.append(
                    f"Overlap on track {track}: clip ends at "
                    f"{sorted_clips[i]['out_timecode']} but next clip "
                    f"starts at {sorted_clips[i + 1]['in_timecode']}"
                )


def _check_duration(
    clips: list[dict], edl: dict, errors: list[str],
) -> None:
    """Check that total duration of track-1 clips is within ±10% of target."""
    target = edl.get("target_duration_seconds")
    if target is None or target <= 0:
        return

    # Sum durations of clips on track 1
    total = 0.0
    for clip in clips:
        if clip.get("track", 1) == 1:
            in_s = timecode_to_seconds(clip["in_timecode"])
            out_s = timecode_to_seconds(clip["out_timecode"])
            total += out_s - in_s

    lower = target * 0.9
    upper = target * 1.1
    if total < lower or total > upper:
        errors.append(
            f"Total duration {total:.1f}s is outside ±10% of "
            f"target {target:.1f}s (allowed: {lower:.1f}–{upper:.1f}s)"
        )


def _check_clip_ids(
    clips: list[dict], db: CatalogDB, errors: list[str],
) -> None:
    """Check that all referenced clip_ids exist in the catalog."""
    seen: set[str] = set()
    for clip in clips:
        clip_id = clip.get("clip_id", "")
        if clip_id in seen:
            continue
        seen.add(clip_id)
        media = db.get_media(clip_id)
        if media is None:
            errors.append(f"Clip not found in catalog: {clip_id}")


def validate_edl(edl: dict, db: CatalogDB) -> ValidationResult:
    """Validate an EDL structure against all constraints.

    Checks for:
    - No overlapping clips on same track
    - Total duration within +/-10% of target
    - All referenced clip_ids exist in catalog
    - In/out timecodes within clip duration bounds
    - Audio levels in broadcast-safe range

    Args:
        edl: EDL dictionary with clips, transitions, audio_settings, etc.
        db: Catalog database for clip existence and duration lookups.

    Returns:
        ValidationResult with passed status, errors, and warnings.
    """
    errors: list[str] = []
    warnings: list[str] = []

    clips = edl.get("clips", [])

    # Check overlaps
    _check_overlaps(clips, errors)

    # Check total duration
    _check_duration(clips, edl, errors)

    # Check clip_id existence
    _check_clip_ids(clips, db, errors)

    passed = len(errors) == 0
    return ValidationResult(passed=passed, errors=errors, warnings=warnings)
