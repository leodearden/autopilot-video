"""Tests for thumbnail extraction (autopilot.upload.thumbnail)."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------


class TestThumbnailPublicAPI:
    """Verify ThumbnailError, extract_best_thumbnail surface."""

    def test_thumbnail_error_importable(self):
        """ThumbnailError is importable from thumbnail module."""
        from autopilot.upload.thumbnail import ThumbnailError

        assert ThumbnailError is not None

    def test_thumbnail_error_is_exception(self):
        """ThumbnailError is a subclass of Exception."""
        from autopilot.upload.thumbnail import ThumbnailError

        assert issubclass(ThumbnailError, Exception)
        err = ThumbnailError("test error")
        assert str(err) == "test error"

    def test_extract_best_thumbnail_signature(self):
        """extract_best_thumbnail has narrative_id, video_path, db params."""
        from autopilot.upload.thumbnail import extract_best_thumbnail

        sig = inspect.signature(extract_best_thumbnail)
        params = list(sig.parameters.keys())
        assert "narrative_id" in params
        assert "video_path" in params
        assert "db" in params

    def test_all_exports(self):
        """__all__ includes ThumbnailError and extract_best_thumbnail."""
        from autopilot.upload import thumbnail

        assert "ThumbnailError" in thumbnail.__all__
        assert "extract_best_thumbnail" in thumbnail.__all__
