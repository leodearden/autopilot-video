"""Tests for autopilot.render.validate — post-render quality validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from autopilot.config import OutputConfig


# ---------------------------------------------------------------------------
# TestPublicAPI — imports, dataclasses, function signatures, __all__
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify public API surface: exports, dataclasses, signatures."""

    def test_module_exports_validate_render(self) -> None:
        from autopilot.render.validate import __all__

        assert "validate_render" in __all__

    def test_module_exports_validation_report(self) -> None:
        from autopilot.render.validate import __all__

        assert "ValidationReport" in __all__

    def test_module_exports_validation_error(self) -> None:
        from autopilot.render.validate import __all__

        assert "ValidationError" in __all__

    def test_validation_report_has_passed_field(self) -> None:
        from autopilot.render.validate import ValidationReport

        report = ValidationReport(passed=True, issues=[], measurements={})
        assert report.passed is True

    def test_validation_report_has_issues_field(self) -> None:
        from autopilot.render.validate import ValidationReport

        report = ValidationReport(passed=True, issues=[], measurements={})
        assert isinstance(report.issues, list)

    def test_validation_report_has_measurements_field(self) -> None:
        from autopilot.render.validate import ValidationReport

        report = ValidationReport(passed=True, issues=[], measurements={})
        assert isinstance(report.measurements, dict)

    def test_issue_has_severity_check_message_measured_value(self) -> None:
        from autopilot.render.validate import Issue

        issue = Issue(
            severity="error",
            check="duration",
            message="Duration mismatch",
            measured_value=120.5,
        )
        assert issue.severity == "error"
        assert issue.check == "duration"
        assert issue.message == "Duration mismatch"
        assert issue.measured_value == 120.5

    def test_validation_error_is_exception(self) -> None:
        from autopilot.render.validate import ValidationError

        assert issubclass(ValidationError, Exception)

    def test_validate_render_returns_validation_report(self) -> None:
        """validate_render(rendered_path, edl, config) -> ValidationReport."""
        from unittest.mock import patch

        from autopilot.render.validate import ValidationReport, validate_render

        config = OutputConfig()
        edl: dict = {"target_duration_seconds": 60}

        # Mock all subprocess calls to prevent actual ffprobe/ffmpeg invocations
        with patch("subprocess.run"):
            result = validate_render(Path("/fake/video.mp4"), edl, config)

        assert isinstance(result, ValidationReport)

    def test_render_init_exports_validation_error(self) -> None:
        from autopilot.render import ValidationError  # noqa: F401

    def test_render_init_exports_validate_render(self) -> None:
        from autopilot.render import validate_render  # noqa: F401
