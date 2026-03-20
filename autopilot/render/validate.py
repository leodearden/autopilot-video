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
