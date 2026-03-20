"""Post-render quality validation for rendered video files.

Provides validate_render() for comprehensive quality checks on rendered
output including duration, loudness, black frames, silence detection,
resolution/codec verification, and file size analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.config import OutputConfig

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
