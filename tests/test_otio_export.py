"""Tests for OTIO export (autopilot.plan.otio_export)."""

from __future__ import annotations

import inspect

import pytest

# -- Step 1: Public API surface tests -----------------------------------------


class TestOtioExportPublicAPI:
    """Verify OtioExportError, export_otio, detect_otio_changes surface."""

    def test_otio_export_error_importable(self):
        """OtioExportError is importable from otio_export module."""
        from autopilot.plan.otio_export import OtioExportError

        assert OtioExportError is not None

    def test_otio_export_error_is_exception(self):
        """OtioExportError is a subclass of Exception with message."""
        from autopilot.plan.otio_export import OtioExportError

        assert issubclass(OtioExportError, Exception)
        err = OtioExportError("test message")
        assert str(err) == "test message"

    def test_export_otio_signature(self):
        """export_otio has edl, output_path, db params and returns Path."""
        from autopilot.plan.otio_export import export_otio

        sig = inspect.signature(export_otio)
        params = list(sig.parameters.keys())
        assert "edl" in params
        assert "output_path" in params
        assert "db" in params

    def test_detect_otio_changes_signature(self):
        """detect_otio_changes has otio_path, original_edl params, returns dict."""
        from autopilot.plan.otio_export import detect_otio_changes

        sig = inspect.signature(detect_otio_changes)
        params = list(sig.parameters.keys())
        assert "otio_path" in params
        assert "original_edl" in params
        assert sig.return_annotation in (dict, "dict")

    def test_all_exports(self):
        """__all__ includes OtioExportError, export_otio, detect_otio_changes."""
        from autopilot.plan import otio_export

        assert "OtioExportError" in otio_export.__all__
        assert "export_otio" in otio_export.__all__
        assert "detect_otio_changes" in otio_export.__all__


# -- Step 3: Timecode-to-RationalTime conversion tests ------------------------

import opentimelineio as otio


class TestTcToRationalTime:
    """Verify _tc_to_rational_time converts HH:MM:SS.mmm to RationalTime."""

    def test_zero_timecode_30fps(self):
        """00:00:00.000 at 30fps -> RationalTime(0, 30)."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("00:00:00.000", 30.0)
        assert rt == otio.opentime.RationalTime(0, 30)

    def test_one_minute_thirty_seconds_30fps(self):
        """00:01:30.500 at 30fps -> 90.5 seconds."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("00:01:30.500", 30.0)
        expected_seconds = 90.5
        assert abs(otio.opentime.to_seconds(rt) - expected_seconds) < 0.001

    def test_one_hour_24fps(self):
        """01:00:00.000 at 24fps -> 3600 seconds."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("01:00:00.000", 24.0)
        assert rt.rate == 24.0
        assert abs(otio.opentime.to_seconds(rt) - 3600.0) < 0.001

    def test_millisecond_precision_60fps(self):
        """00:00:05.123 at 60fps -> 5.123 seconds."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("00:00:05.123", 60.0)
        assert rt.rate == 60.0
        assert abs(otio.opentime.to_seconds(rt) - 5.123) < 0.001

    def test_ten_seconds_30fps(self):
        """00:00:10.000 at 30fps -> 10.0 seconds."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("00:00:10.000", 30.0)
        assert abs(otio.opentime.to_seconds(rt) - 10.0) < 0.001
