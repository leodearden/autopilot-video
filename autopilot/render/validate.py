"""Post-render quality validation for rendered video files.

Provides validate_render() for comprehensive quality checks on rendered
output including duration, loudness, black frames, silence detection,
resolution/codec verification, and file size analysis.
"""

from __future__ import annotations

import json
import logging
import re
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
            "measured_value": self.measured_value,
        }


@dataclass
class ValidationReport:
    """Result of post-render validation with pass/fail, issues, and measurements."""

    passed: bool
    issues: list[Issue] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON output."""
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "measurements": self.measurements,
        }


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


def _check_loudness(
    rendered_path: Path, config: OutputConfig, issues: list[Issue],
) -> float | None:
    """Check integrated loudness via ffmpeg loudnorm analysis pass.

    Returns the measured LUFS value on success, or None on failure.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(rendered_path),
                "-af",
                "loudnorm=print_format=json",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        issues.append(
            Issue(
                severity="error",
                check="loudness",
                message=f"ffmpeg loudness analysis failed: {exc}",
            )
        )
        return None

    # Extract JSON block from stderr
    stderr = result.stderr or ""
    match = re.search(r"\{[^}]+\}", stderr, re.DOTALL)
    if match is None:
        issues.append(
            Issue(
                severity="error",
                check="loudness",
                message="Could not parse loudnorm output from ffmpeg stderr",
            )
        )
        return None

    try:
        loudnorm_data = json.loads(match.group())
        measured_lufs = float(loudnorm_data["input_i"])
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
        issues.append(
            Issue(
                severity="error",
                check="loudness",
                message=f"Could not parse LUFS from loudnorm output: {exc}",
            )
        )
        return None

    target = config.target_loudness_lufs
    if abs(measured_lufs - target) > 1.0:
        issues.append(
            Issue(
                severity="error",
                check="loudness",
                message=(
                    f"Integrated loudness {measured_lufs:.1f} LUFS differs from "
                    f"target {target} LUFS by more than 1 LUFS"
                ),
                measured_value=measured_lufs,
            )
        )

    return measured_lufs


def _check_black_frames(rendered_path: Path, issues: list[Issue]) -> None:
    """Detect black frames via ffmpeg blackdetect filter."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(rendered_path),
                "-vf",
                "blackdetect=d=0.1:pix_th=5",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        issues.append(
            Issue(
                severity="warning",
                check="black_frames",
                message=f"ffmpeg blackdetect failed: {exc}",
            )
        )
        return

    stderr = result.stderr or ""
    for match in re.finditer(
        r"black_start:(\S+)\s+black_end:(\S+)\s+black_duration:(\S+)",
        stderr,
    ):
        start_s, end_s, dur_s = match.group(1), match.group(2), match.group(3)
        try:
            start_f = float(start_s)
            end_f = float(end_s)
            dur_f = float(dur_s)
        except ValueError:
            issues.append(
                Issue(
                    severity="warning",
                    check="black_frames",
                    message=(
                        f"Could not parse blackdetect values: {match.group()}"
                    ),
                )
            )
            continue
        issues.append(
            Issue(
                severity="warning",
                check="black_frames",
                message=(
                    f"Black frame detected: {start_s}s–{end_s}s "
                    f"(duration: {dur_s}s)"
                ),
                measured_value={
                    "start": start_f,
                    "end": end_f,
                    "duration": dur_f,
                },
            )
        )


def _check_silence(
    rendered_path: Path, edl: dict, issues: list[Issue],
) -> None:
    """Detect silence gaps >2s via ffmpeg silencedetect filter.

    Excludes gaps that overlap with EDL-marked intentional silences.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i",
                str(rendered_path),
                "-af",
                "silencedetect=n=-50dB:d=2",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        issues.append(
            Issue(
                severity="warning",
                check="silence",
                message=f"ffmpeg silencedetect failed: {exc}",
            )
        )
        return

    stderr = result.stderr or ""
    intentional = edl.get("intentional_silences", [])

    for match in re.finditer(
        r"silence_start:\s*(\S+).*?silence_end:\s*(\S+)\s*\|\s*silence_duration:\s*(\S+)",
        stderr,
        re.DOTALL,
    ):
        try:
            start = float(match.group(1))
            end = float(match.group(2))
            duration = float(match.group(3))
        except ValueError:
            issues.append(
                Issue(
                    severity="warning",
                    check="silence",
                    message=(
                        f"Could not parse silencedetect values: {match.group()}"
                    ),
                )
            )
            continue

        # Skip if covered by an intentional silence range
        is_intentional = any(
            s.get("start", 0) <= start and s.get("end", 0) >= end
            for s in intentional
        )
        if is_intentional:
            continue

        issues.append(
            Issue(
                severity="warning",
                check="silence",
                message=(
                    f"Silence detected: {start:.1f}s–{end:.1f}s "
                    f"(duration: {duration:.1f}s)"
                ),
                measured_value={
                    "start": start,
                    "end": end,
                    "duration": duration,
                },
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

    # Probe the rendered file once for metadata
    probe_data = _run_ffprobe_json(rendered_path)

    # Populate measurements from probe data
    if "duration_seconds" in probe_data:
        measurements["duration"] = probe_data["duration_seconds"]
    if "resolution" in probe_data:
        measurements["resolution"] = probe_data["resolution"]
    if "video_codec" in probe_data:
        measurements["codec"] = probe_data["video_codec"]
    if "file_size_bytes" in probe_data:
        measurements["file_size_mb"] = round(
            probe_data["file_size_bytes"] / (1024 * 1024), 2
        )

    # Run individual checks
    _check_duration(probe_data, edl, issues)
    _check_resolution_codec(probe_data, config, issues)
    _check_file_size(probe_data, config, issues)

    # Loudness check (separate ffmpeg call)
    lufs = _check_loudness(rendered_path, config, issues)
    if lufs is not None:
        measurements["loudness_lufs"] = lufs

    # Black frame detection (separate ffmpeg call)
    _check_black_frames(rendered_path, issues)

    # Silence detection (separate ffmpeg call)
    _check_silence(rendered_path, edl, issues)

    passed = not any(i.severity == "error" for i in issues)
    report = ValidationReport(passed=passed, issues=issues, measurements=measurements)

    # Write validation_report.json alongside the rendered file
    report_path = rendered_path.parent / "validation_report.json"
    try:
        rendered_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    except OSError as exc:
        logger.warning(
            "Could not write validation report to %s: %s",
            report_path,
            exc,
        )

    return report
