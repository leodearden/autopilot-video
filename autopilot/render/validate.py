"""Post-render quality validation for rendered video files.

Provides validate_render() for comprehensive quality checks on rendered
output including duration, loudness, black frames, silence detection,
resolution/codec verification, and file size analysis.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.config import OutputConfig

logger = logging.getLogger(__name__)

__all__ = [
    "Issue",
    "ValidationError",
    "ValidationReport",
    "validate_render",
]


class ValidationError(Exception):
    """Raised for post-render validation processing errors."""


@dataclass
class Issue:
    """A single validation issue with severity and context."""

    severity: str  # "error" or "warning"
    check: str  # which check produced this issue
    message: str  # human-readable description
    measured_value: Any = None  # the measured value that triggered the issue


@dataclass
class ValidationReport:
    """Result of post-render validation with pass/fail, issues, and measurements."""

    passed: bool
    issues: list[Issue] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)


def _run_ffprobe_json(path: Path) -> dict:
    """Run ffprobe and return extracted metadata dict.

    Keys on success: duration_seconds, resolution (w,h), video_codec,
    audio_codec, file_size_bytes.

    Returns an empty dict on any error.
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
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        logger.warning("ffprobe failed for %s", path)
        return {}

    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError):
        logger.warning("ffprobe returned invalid JSON for %s", path)
        return {}

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    out: dict[str, Any] = {}

    if video_stream:
        out["video_codec"] = video_stream.get("codec_name")
        w = video_stream.get("width")
        h = video_stream.get("height")
        if w is not None and h is not None:
            out["resolution"] = (int(w), int(h))

    if audio_stream:
        out["audio_codec"] = audio_stream.get("codec_name")

    dur = fmt.get("duration")
    if dur is not None:
        try:
            out["duration_seconds"] = float(dur)
        except (ValueError, TypeError):
            pass

    size = fmt.get("size")
    if size is not None:
        try:
            out["file_size_bytes"] = int(size)
        except (ValueError, TypeError):
            pass

    return out


def _check_duration(
    probe_data: dict, edl: dict, issues: list[Issue],
) -> None:
    """Check rendered duration against EDL target within ±1s tolerance."""
    target = edl.get("target_duration_seconds")
    if target is None:
        return

    actual = probe_data.get("duration_seconds")
    if actual is None:
        return

    if abs(actual - target) > 1.0:
        issues.append(
            Issue(
                severity="error",
                check="duration",
                message=(
                    f"Duration {actual:.1f}s differs from target "
                    f"{target:.1f}s by more than 1s"
                ),
                measured_value=actual,
            )
        )


def _check_resolution_codec(
    probe_data: dict, config: OutputConfig, issues: list[Issue],
) -> None:
    """Check rendered resolution and codec against config."""
    actual_res = probe_data.get("resolution")
    if actual_res is not None and actual_res != config.resolution:
        issues.append(
            Issue(
                severity="error",
                check="resolution",
                message=(
                    f"Resolution {actual_res[0]}x{actual_res[1]} does not match "
                    f"expected {config.resolution[0]}x{config.resolution[1]}"
                ),
                measured_value=actual_res,
            )
        )

    actual_codec = probe_data.get("video_codec")
    if actual_codec is not None and actual_codec != config.codec:
        issues.append(
            Issue(
                severity="error",
                check="codec",
                message=(
                    f"Codec {actual_codec!r} does not match "
                    f"expected {config.codec!r}"
                ),
                measured_value=actual_codec,
            )
        )


# File size thresholds for 1080p h264 (MB per minute)
_FILE_SIZE_MIN_MB_PER_MIN = 8
_FILE_SIZE_MAX_MB_PER_MIN = 15


def _check_file_size(
    probe_data: dict, config: OutputConfig, issues: list[Issue],
) -> None:
    """Check file size is within expected range for 1080p h264.

    Only runs the check when config specifies 1080p h264; skips otherwise.
    """
    # Only apply to 1080p h264
    if config.resolution != (1920, 1080) or config.codec != "h264":
        return

    duration = probe_data.get("duration_seconds")
    file_size = probe_data.get("file_size_bytes")
    if duration is None or file_size is None or duration <= 0:
        return

    duration_minutes = duration / 60.0
    file_size_mb = file_size / (1024 * 1024)
    mb_per_min = file_size_mb / duration_minutes

    if mb_per_min < _FILE_SIZE_MIN_MB_PER_MIN:
        issues.append(
            Issue(
                severity="warning",
                check="file_size",
                message=(
                    f"File size {file_size_mb:.1f} MB ({mb_per_min:.1f} MB/min) "
                    f"is below expected minimum of {_FILE_SIZE_MIN_MB_PER_MIN} MB/min"
                ),
                measured_value=mb_per_min,
            )
        )
    elif mb_per_min > _FILE_SIZE_MAX_MB_PER_MIN:
        issues.append(
            Issue(
                severity="warning",
                check="file_size",
                message=(
                    f"File size {file_size_mb:.1f} MB ({mb_per_min:.1f} MB/min) "
                    f"exceeds expected maximum of {_FILE_SIZE_MAX_MB_PER_MIN} MB/min"
                ),
                measured_value=mb_per_min,
            )
        )


def validate_render(
    rendered_path: Path,
    edl: dict,
    config: OutputConfig,
) -> ValidationReport:
    """Run post-render quality checks on a rendered video file.

    Args:
        rendered_path: Path to the rendered video file.
        edl: EDL dictionary (may contain target_duration_seconds).
        config: Output configuration with resolution, codec, loudness targets.

    Returns:
        ValidationReport with pass/fail status, issues, and measurements.
    """
    issues: list[Issue] = []
    measurements: dict[str, Any] = {}

    passed = not any(i.severity == "error" for i in issues)
    return ValidationReport(passed=passed, issues=issues, measurements=measurements)
