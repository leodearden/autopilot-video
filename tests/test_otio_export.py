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
