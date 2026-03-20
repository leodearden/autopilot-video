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

    def test_validate_render_returns_validation_report(self, tmp_path: Path) -> None:
        """validate_render(rendered_path, edl, config) -> ValidationReport."""
        from autopilot.render.validate import ValidationReport, validate_render

        config = OutputConfig()
        edl: dict = {"target_duration_seconds": 60}
        rendered = tmp_path / "video.mp4"
        rendered.touch()

        # Mock all subprocess calls with proper return values
        def _side_effect(*args, **kwargs):
            mock = MagicMock()
            mock.stdout = json.dumps(SAMPLE_FFPROBE_JSON)
            mock.stderr = LOUDNORM_STDERR_OK
            mock.returncode = 0
            return mock

        with patch("subprocess.run", side_effect=_side_effect):
            result = validate_render(rendered, edl, config)

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


# ---------------------------------------------------------------------------
# TestBlackFrameCheck — _check_black_frames
# ---------------------------------------------------------------------------

BLACKDETECT_STDERR_FOUND = """\
[blackdetect @ 0x5555] black_start:0 black_end:0.5 black_duration:0.5
[blackdetect @ 0x5555] black_start:58.2 black_end:58.8 black_duration:0.6
"""

BLACKDETECT_STDERR_NONE = """\
frame=  100 fps=0.0 q=0.0 size=N/A time=00:00:04.00 bitrate=N/A
"""


class TestBlackFrameCheck:
    """Tests for _check_black_frames internal helper."""

    def test_pass_when_no_black_frames(self) -> None:
        from autopilot.render.validate import Issue, _check_black_frames

        mock = MagicMock()
        mock.stderr = BLACKDETECT_STDERR_NONE
        mock.returncode = 0

        issues: list[Issue] = []

        with patch("subprocess.run", return_value=mock):
            _check_black_frames(Path("/fake/video.mp4"), issues)

        assert len(issues) == 0

    def test_warning_when_black_frames_found(self) -> None:
        from autopilot.render.validate import Issue, _check_black_frames

        mock = MagicMock()
        mock.stderr = BLACKDETECT_STDERR_FOUND
        mock.returncode = 0

        issues: list[Issue] = []

        with patch("subprocess.run", return_value=mock):
            _check_black_frames(Path("/fake/video.mp4"), issues)

        assert len(issues) == 2
        assert all(i.severity == "warning" for i in issues)
        assert all(i.check == "black_frames" for i in issues)

    def test_graceful_handling_on_ffmpeg_failure(self) -> None:
        from autopilot.render.validate import Issue, _check_black_frames

        issues: list[Issue] = []

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            _check_black_frames(Path("/fake/video.mp4"), issues)

        # Should add a warning, not crash
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_non_numeric_blackdetect_values_do_not_raise(self) -> None:
        """Non-numeric values like 'N/A' in blackdetect output should not crash."""
        from autopilot.render.validate import Issue, _check_black_frames

        stderr = (
            "[blackdetect @ 0x5555] black_start:N/A black_end:2.5 black_duration:N/A\n"
        )
        mock = MagicMock()
        mock.stderr = stderr
        mock.returncode = 0

        issues: list[Issue] = []

        with patch("subprocess.run", return_value=mock):
            _check_black_frames(Path("/fake/video.mp4"), issues)

        # Should NOT raise ValueError; instead append a warning about parse failure
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].check == "black_frames"
        assert "parse" in issues[0].message.lower() or "could not" in issues[0].message.lower()

    def test_mixed_valid_and_invalid_blackdetect_values(self) -> None:
        """Valid entries should still be parsed even if some entries have bad values."""
        from autopilot.render.validate import Issue, _check_black_frames

        stderr = (
            "[blackdetect @ 0x5555] black_start:N/A black_end:2.5 black_duration:N/A\n"
            "[blackdetect @ 0x5555] black_start:10.0 black_end:10.5 black_duration:0.5\n"
        )
        mock = MagicMock()
        mock.stderr = stderr
        mock.returncode = 0

        issues: list[Issue] = []

        with patch("subprocess.run", return_value=mock):
            _check_black_frames(Path("/fake/video.mp4"), issues)

        # Should have 2 issues: 1 parse warning + 1 valid detection
        assert len(issues) == 2
        # The valid detection should have measured_value with numeric values
        valid_issues = [i for i in issues if i.measured_value is not None and isinstance(i.measured_value, dict)]
        assert len(valid_issues) == 1
        assert valid_issues[0].measured_value["start"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TestSilenceCheck — _check_silence
# ---------------------------------------------------------------------------

SILENCE_STDERR_FOUND = """\
[silencedetect @ 0x5555] silence_start: 10.5
[silencedetect @ 0x5555] silence_end: 15.2 | silence_duration: 4.7
[silencedetect @ 0x5555] silence_start: 45.0
[silencedetect @ 0x5555] silence_end: 48.5 | silence_duration: 3.5
"""

SILENCE_STDERR_NONE = """\
frame=  200 fps=0.0 q=0.0 size=N/A time=00:00:08.00 bitrate=N/A
"""


class TestSilenceCheck:
    """Tests for _check_silence internal helper."""

    def test_pass_when_no_silence_detected(self) -> None:
        from autopilot.render.validate import Issue, _check_silence

        mock = MagicMock()
        mock.stderr = SILENCE_STDERR_NONE
        mock.returncode = 0

        issues: list[Issue] = []

        with patch("subprocess.run", return_value=mock):
            _check_silence(Path("/fake/video.mp4"), {}, issues)

        assert len(issues) == 0

    def test_warning_when_silence_found(self) -> None:
        from autopilot.render.validate import Issue, _check_silence

        mock = MagicMock()
        mock.stderr = SILENCE_STDERR_FOUND
        mock.returncode = 0

        issues: list[Issue] = []

        with patch("subprocess.run", return_value=mock):
            _check_silence(Path("/fake/video.mp4"), {}, issues)

        assert len(issues) == 2
        assert all(i.severity == "warning" for i in issues)
        assert all(i.check == "silence" for i in issues)

    def test_intentional_silence_excluded(self) -> None:
        from autopilot.render.validate import Issue, _check_silence

        mock = MagicMock()
        mock.stderr = SILENCE_STDERR_FOUND
        mock.returncode = 0

        # Mark the first silence as intentional
        edl = {
            "intentional_silences": [
                {"start": 10.0, "end": 16.0},
            ],
        }
        issues: list[Issue] = []

        with patch("subprocess.run", return_value=mock):
            _check_silence(Path("/fake/video.mp4"), edl, issues)

        # Only the second silence (45-48.5) should be reported
        assert len(issues) == 1
        assert issues[0].measured_value["start"] == pytest.approx(45.0)

    def test_graceful_handling_on_ffmpeg_failure(self) -> None:
        from autopilot.render.validate import Issue, _check_silence

        issues: list[Issue] = []

        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg")):
            _check_silence(Path("/fake/video.mp4"), {}, issues)

        assert len(issues) == 1
        assert issues[0].severity == "warning"


# ---------------------------------------------------------------------------
# TestValidateRenderE2E — end-to-end orchestration
# ---------------------------------------------------------------------------


def _make_subprocess_side_effect(
    ffprobe_data: dict,
    loudnorm_stderr: str,
    blackdetect_stderr: str,
    silence_stderr: str,
):
    """Create a side_effect function that routes calls to the right mock."""

    def side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        mock_result = MagicMock()
        mock_result.returncode = 0

        if cmd[0] == "ffprobe":
            mock_result.stdout = json.dumps(ffprobe_data)
            mock_result.stderr = ""
        elif "loudnorm" in str(cmd):
            mock_result.stdout = ""
            mock_result.stderr = loudnorm_stderr
        elif "blackdetect" in str(cmd):
            mock_result.stdout = ""
            mock_result.stderr = blackdetect_stderr
        elif "silencedetect" in str(cmd):
            mock_result.stdout = ""
            mock_result.stderr = silence_stderr
        else:
            mock_result.stdout = ""
            mock_result.stderr = ""

        return mock_result

    return side_effect


class TestValidateRenderE2E:
    """End-to-end tests for validate_render orchestration."""

    def test_full_pass_scenario(self, tmp_path: Path) -> None:
        from autopilot.render.validate import validate_render

        config = OutputConfig(
            resolution=(1920, 1080),
            codec="h264",
            target_loudness_lufs=-16,
        )
        edl = {"target_duration_seconds": 120}
        rendered = tmp_path / "video.mp4"
        rendered.touch()

        side_effect = _make_subprocess_side_effect(
            ffprobe_data=SAMPLE_FFPROBE_JSON,
            loudnorm_stderr=LOUDNORM_STDERR_OK,
            blackdetect_stderr=BLACKDETECT_STDERR_NONE,
            silence_stderr=SILENCE_STDERR_NONE,
        )

        with patch("subprocess.run", side_effect=side_effect):
            report = validate_render(rendered, edl, config)

        assert report.passed is True
        assert len([i for i in report.issues if i.severity == "error"]) == 0
        # Should have measurements populated
        assert "duration" in report.measurements
        assert "resolution" in report.measurements
        assert "codec" in report.measurements

    def test_mixed_issues_scenario(self, tmp_path: Path) -> None:
        from autopilot.render.validate import validate_render

        config = OutputConfig(
            resolution=(1920, 1080),
            codec="h264",
            target_loudness_lufs=-16,
        )
        # Duration doesn't match (probe says 120.5, target 60)
        edl = {"target_duration_seconds": 60}
        rendered = tmp_path / "video.mp4"
        rendered.touch()

        side_effect = _make_subprocess_side_effect(
            ffprobe_data=SAMPLE_FFPROBE_JSON,
            loudnorm_stderr=LOUDNORM_STDERR_TOO_LOUD,
            blackdetect_stderr=BLACKDETECT_STDERR_FOUND,
            silence_stderr=SILENCE_STDERR_NONE,
        )

        with patch("subprocess.run", side_effect=side_effect):
            report = validate_render(rendered, edl, config)

        assert report.passed is False
        errors = [i for i in report.issues if i.severity == "error"]
        warnings = [i for i in report.issues if i.severity == "warning"]
        assert len(errors) >= 2  # duration + loudness
        assert len(warnings) >= 2  # black frames

    def test_report_to_dict(self, tmp_path: Path) -> None:
        from autopilot.render.validate import validate_render

        config = OutputConfig()
        edl: dict = {}
        rendered = tmp_path / "video.mp4"
        rendered.touch()

        side_effect = _make_subprocess_side_effect(
            ffprobe_data=SAMPLE_FFPROBE_JSON,
            loudnorm_stderr=LOUDNORM_STDERR_OK,
            blackdetect_stderr=BLACKDETECT_STDERR_NONE,
            silence_stderr=SILENCE_STDERR_NONE,
        )

        with patch("subprocess.run", side_effect=side_effect):
            report = validate_render(rendered, edl, config)

        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert "passed" in report_dict
        assert "issues" in report_dict
        assert "measurements" in report_dict


# ---------------------------------------------------------------------------
# TestReportWriting — validation_report.json file output
# ---------------------------------------------------------------------------


class TestReportWriting:
    """Tests for validation_report.json file writing."""

    def test_writes_report_file(self, tmp_path: Path) -> None:
        from autopilot.render.validate import validate_render

        config = OutputConfig()
        edl: dict = {}
        rendered = tmp_path / "output" / "my_narrative" / "final.mp4"
        rendered.parent.mkdir(parents=True)
        rendered.touch()

        side_effect = _make_subprocess_side_effect(
            ffprobe_data=SAMPLE_FFPROBE_JSON,
            loudnorm_stderr=LOUDNORM_STDERR_OK,
            blackdetect_stderr=BLACKDETECT_STDERR_NONE,
            silence_stderr=SILENCE_STDERR_NONE,
        )

        with patch("subprocess.run", side_effect=side_effect):
            report = validate_render(rendered, edl, config)

        report_file = rendered.parent / "validation_report.json"
        assert report_file.exists()
        data = json.loads(report_file.read_text())
        assert data["passed"] == report.passed
        assert isinstance(data["issues"], list)
        assert isinstance(data["measurements"], dict)

    def test_creates_missing_directory(self, tmp_path: Path) -> None:
        from autopilot.render.validate import validate_render

        config = OutputConfig()
        edl: dict = {}
        rendered = tmp_path / "new_dir" / "final.mp4"
        # Do NOT create the directory — validate_render should create it

        side_effect = _make_subprocess_side_effect(
            ffprobe_data=SAMPLE_FFPROBE_JSON,
            loudnorm_stderr=LOUDNORM_STDERR_OK,
            blackdetect_stderr=BLACKDETECT_STDERR_NONE,
            silence_stderr=SILENCE_STDERR_NONE,
        )

        with patch("subprocess.run", side_effect=side_effect):
            validate_render(rendered, edl, config)

        report_file = rendered.parent / "validation_report.json"
        assert report_file.exists()
