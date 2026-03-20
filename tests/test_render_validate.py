"""Tests for autopilot.render.validate — post-render quality validation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# Helpers — sample ffprobe JSON
# ---------------------------------------------------------------------------

SAMPLE_FFPROBE_JSON = {
    "streams": [
        {
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
        },
        {
            "codec_type": "audio",
            "codec_name": "aac",
        },
    ],
    "format": {
        "duration": "120.5",
        "size": "15000000",
    },
}


def _mock_ffprobe_result(data: dict) -> MagicMock:
    """Create a mock subprocess.CompletedProcess for ffprobe."""
    mock = MagicMock()
    mock.stdout = json.dumps(data)
    mock.returncode = 0
    return mock


# ---------------------------------------------------------------------------
# TestRunFfprobeJson — _run_ffprobe_json helper
# ---------------------------------------------------------------------------


class TestRunFfprobeJson:
    """Tests for _run_ffprobe_json internal helper."""

    def test_extracts_duration(self) -> None:
        from autopilot.render.validate import _run_ffprobe_json

        with patch("subprocess.run", return_value=_mock_ffprobe_result(SAMPLE_FFPROBE_JSON)):
            result = _run_ffprobe_json(Path("/fake/video.mp4"))

        assert result["duration_seconds"] == pytest.approx(120.5)

    def test_extracts_resolution(self) -> None:
        from autopilot.render.validate import _run_ffprobe_json

        with patch("subprocess.run", return_value=_mock_ffprobe_result(SAMPLE_FFPROBE_JSON)):
            result = _run_ffprobe_json(Path("/fake/video.mp4"))

        assert result["resolution"] == (1920, 1080)

    def test_extracts_video_codec(self) -> None:
        from autopilot.render.validate import _run_ffprobe_json

        with patch("subprocess.run", return_value=_mock_ffprobe_result(SAMPLE_FFPROBE_JSON)):
            result = _run_ffprobe_json(Path("/fake/video.mp4"))

        assert result["video_codec"] == "h264"

    def test_extracts_audio_codec(self) -> None:
        from autopilot.render.validate import _run_ffprobe_json

        with patch("subprocess.run", return_value=_mock_ffprobe_result(SAMPLE_FFPROBE_JSON)):
            result = _run_ffprobe_json(Path("/fake/video.mp4"))

        assert result["audio_codec"] == "aac"

    def test_extracts_file_size(self) -> None:
        from autopilot.render.validate import _run_ffprobe_json

        with patch("subprocess.run", return_value=_mock_ffprobe_result(SAMPLE_FFPROBE_JSON)):
            result = _run_ffprobe_json(Path("/fake/video.mp4"))

        assert result["file_size_bytes"] == 15000000

    def test_returns_empty_dict_on_subprocess_error(self) -> None:
        from autopilot.render.validate import _run_ffprobe_json

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffprobe")):
            result = _run_ffprobe_json(Path("/fake/video.mp4"))

        assert result == {}

    def test_returns_empty_dict_on_bad_json(self) -> None:
        from autopilot.render.validate import _run_ffprobe_json

        mock = MagicMock()
        mock.stdout = "not json"
        mock.returncode = 0

        with patch("subprocess.run", return_value=mock):
            result = _run_ffprobe_json(Path("/fake/video.mp4"))

        assert result == {}


# ---------------------------------------------------------------------------
# TestDurationCheck — _check_duration
# ---------------------------------------------------------------------------


class TestDurationCheck:
    """Tests for _check_duration internal helper."""

    def test_pass_when_within_tolerance(self) -> None:
        from autopilot.render.validate import Issue, _check_duration

        probe = {"duration_seconds": 60.5}
        edl = {"target_duration_seconds": 60}
        issues: list[Issue] = []

        _check_duration(probe, edl, issues)

        assert len(issues) == 0

    def test_error_when_duration_off_by_more_than_1s(self) -> None:
        from autopilot.render.validate import Issue, _check_duration

        probe = {"duration_seconds": 65.0}
        edl = {"target_duration_seconds": 60}
        issues: list[Issue] = []

        _check_duration(probe, edl, issues)

        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].check == "duration"
        assert issues[0].measured_value == 65.0

    def test_skip_when_no_target_duration(self) -> None:
        from autopilot.render.validate import Issue, _check_duration

        probe = {"duration_seconds": 120.0}
        edl: dict = {}
        issues: list[Issue] = []

        _check_duration(probe, edl, issues)

        assert len(issues) == 0

    def test_error_when_duration_too_short(self) -> None:
        from autopilot.render.validate import Issue, _check_duration

        probe = {"duration_seconds": 55.0}
        edl = {"target_duration_seconds": 60}
        issues: list[Issue] = []

        _check_duration(probe, edl, issues)

        assert len(issues) == 1
        assert issues[0].severity == "error"


# ---------------------------------------------------------------------------
# TestResolutionCodecCheck — _check_resolution_codec
# ---------------------------------------------------------------------------


class TestResolutionCodecCheck:
    """Tests for _check_resolution_codec internal helper."""

    def test_pass_when_matches(self) -> None:
        from autopilot.render.validate import Issue, _check_resolution_codec

        probe = {"resolution": (1920, 1080), "video_codec": "h264"}
        config = OutputConfig(resolution=(1920, 1080), codec="h264")
        issues: list[Issue] = []

        _check_resolution_codec(probe, config, issues)

        assert len(issues) == 0

    def test_error_on_resolution_mismatch(self) -> None:
        from autopilot.render.validate import Issue, _check_resolution_codec

        probe = {"resolution": (1280, 720), "video_codec": "h264"}
        config = OutputConfig(resolution=(1920, 1080), codec="h264")
        issues: list[Issue] = []

        _check_resolution_codec(probe, config, issues)

        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].check == "resolution"

    def test_error_on_codec_mismatch(self) -> None:
        from autopilot.render.validate import Issue, _check_resolution_codec

        probe = {"resolution": (1920, 1080), "video_codec": "vp9"}
        config = OutputConfig(resolution=(1920, 1080), codec="h264")
        issues: list[Issue] = []

        _check_resolution_codec(probe, config, issues)

        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].check == "codec"

    def test_both_mismatch_produces_two_issues(self) -> None:
        from autopilot.render.validate import Issue, _check_resolution_codec

        probe = {"resolution": (1280, 720), "video_codec": "vp9"}
        config = OutputConfig(resolution=(1920, 1080), codec="h264")
        issues: list[Issue] = []

        _check_resolution_codec(probe, config, issues)

        assert len(issues) == 2


# ---------------------------------------------------------------------------
# TestFileSizeCheck — _check_file_size
# ---------------------------------------------------------------------------


class TestFileSizeCheck:
    """Tests for _check_file_size internal helper."""

    def test_pass_when_within_range(self) -> None:
        """10 MB/min for 2 minutes = 20 MB total -> in 8-15 MB/min range."""
        from autopilot.render.validate import Issue, _check_file_size

        # 2 minutes, 20 MB = 10 MB/min -> in range
        probe = {
            "duration_seconds": 120.0,
            "file_size_bytes": 20 * 1024 * 1024,
            "resolution": (1920, 1080),
            "video_codec": "h264",
        }
        config = OutputConfig(resolution=(1920, 1080), codec="h264")
        issues: list[Issue] = []

        _check_file_size(probe, config, issues)

        assert len(issues) == 0

    def test_warning_when_below_minimum(self) -> None:
        """5 MB/min for 2 minutes = 10 MB total -> below 8 MB/min."""
        from autopilot.render.validate import Issue, _check_file_size

        # 2 minutes, 10 MB = 5 MB/min -> below 8 MB/min
        probe = {
            "duration_seconds": 120.0,
            "file_size_bytes": 10 * 1024 * 1024,
            "resolution": (1920, 1080),
            "video_codec": "h264",
        }
        config = OutputConfig(resolution=(1920, 1080), codec="h264")
        issues: list[Issue] = []

        _check_file_size(probe, config, issues)

        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].check == "file_size"

    def test_warning_when_above_maximum(self) -> None:
        """20 MB/min for 2 minutes = 40 MB total -> above 15 MB/min."""
        from autopilot.render.validate import Issue, _check_file_size

        # 2 minutes, 40 MB = 20 MB/min -> above 15 MB/min
        probe = {
            "duration_seconds": 120.0,
            "file_size_bytes": 40 * 1024 * 1024,
            "resolution": (1920, 1080),
            "video_codec": "h264",
        }
        config = OutputConfig(resolution=(1920, 1080), codec="h264")
        issues: list[Issue] = []

        _check_file_size(probe, config, issues)

        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].check == "file_size"

    def test_skip_when_not_1080p_h264(self) -> None:
        """Non-1080p or non-h264 should skip the check."""
        from autopilot.render.validate import Issue, _check_file_size

        probe = {
            "duration_seconds": 120.0,
            "file_size_bytes": 1 * 1024 * 1024,  # very small, would warn otherwise
            "resolution": (3840, 2160),
            "video_codec": "h265",
        }
        config = OutputConfig(resolution=(3840, 2160), codec="h265")
        issues: list[Issue] = []

        _check_file_size(probe, config, issues)

        assert len(issues) == 0


# ---------------------------------------------------------------------------
# TestLoudnessCheck — _check_loudness
# ---------------------------------------------------------------------------

# Sample ffmpeg loudnorm stderr output
LOUDNORM_STDERR_OK = """
[Parsed_loudnorm_0 @ 0x5555]
{
	"input_i" : "-16.2",
	"input_tp" : "-1.5",
	"input_lra" : "7.2",
	"input_thresh" : "-26.5",
	"output_i" : "-16.0",
	"output_tp" : "-1.0",
	"output_lra" : "5.0",
	"output_thresh" : "-26.0",
	"normalization_type" : "dynamic",
	"target_offset" : "0.0"
}
"""

LOUDNORM_STDERR_TOO_LOUD = """
[Parsed_loudnorm_0 @ 0x5555]
{
	"input_i" : "-10.0",
	"input_tp" : "-1.5",
	"input_lra" : "7.2",
	"input_thresh" : "-26.5",
	"output_i" : "-10.0",
	"output_tp" : "-1.0",
	"output_lra" : "5.0",
	"output_thresh" : "-26.0",
	"normalization_type" : "dynamic",
	"target_offset" : "0.0"
}
"""


class TestLoudnessCheck:
    """Tests for _check_loudness internal helper."""

    def test_pass_when_within_target(self) -> None:
        from autopilot.render.validate import Issue, _check_loudness

        mock = MagicMock()
        mock.stderr = LOUDNORM_STDERR_OK
        mock.returncode = 0

        config = OutputConfig(target_loudness_lufs=-16)
        issues: list[Issue] = []

        with patch("subprocess.run", return_value=mock):
            _check_loudness(Path("/fake/video.mp4"), config, issues)

        assert len(issues) == 0

    def test_error_when_outside_target(self) -> None:
        from autopilot.render.validate import Issue, _check_loudness

        mock = MagicMock()
        mock.stderr = LOUDNORM_STDERR_TOO_LOUD
        mock.returncode = 0

        config = OutputConfig(target_loudness_lufs=-16)
        issues: list[Issue] = []

        with patch("subprocess.run", return_value=mock):
            _check_loudness(Path("/fake/video.mp4"), config, issues)

        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].check == "loudness"
        assert issues[0].measured_value == pytest.approx(-10.0)

    def test_error_when_ffmpeg_fails(self) -> None:
        from autopilot.render.validate import Issue, _check_loudness

        config = OutputConfig(target_loudness_lufs=-16)
        issues: list[Issue] = []

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            _check_loudness(Path("/fake/video.mp4"), config, issues)

        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].check == "loudness"
