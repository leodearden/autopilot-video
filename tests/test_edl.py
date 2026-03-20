"""Tests for EDL generation (autopilot.plan.edl)."""

from __future__ import annotations

import inspect
import json
import sys
from unittest.mock import MagicMock, patch

import pytest


# -- Step 13: Public API surface tests ----------------------------------------


class TestEdlPublicAPI:
    """Verify EdlError, generate_edl, and TOOL_DEFINITIONS surface."""

    def test_edl_error_importable(self):
        """EdlError is importable from edl module."""
        from autopilot.plan.edl import EdlError

        assert EdlError is not None

    def test_edl_error_is_exception(self):
        """EdlError is a subclass of Exception with message."""
        from autopilot.plan.edl import EdlError

        assert issubclass(EdlError, Exception)
        err = EdlError("test message")
        assert str(err) == "test message"

    def test_generate_edl_signature(self):
        """generate_edl has narrative_id, db, config params, returns dict."""
        from autopilot.plan.edl import generate_edl

        sig = inspect.signature(generate_edl)
        params = list(sig.parameters.keys())
        assert "narrative_id" in params
        assert "db" in params
        assert "config" in params
        assert sig.return_annotation in (dict, "dict")

    def test_all_exports(self):
        """__all__ includes EdlError, generate_edl, TOOL_DEFINITIONS."""
        from autopilot.plan import edl

        assert "EdlError" in edl.__all__
        assert "generate_edl" in edl.__all__
        assert "TOOL_DEFINITIONS" in edl.__all__
