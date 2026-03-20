"""Tests for YouTube upload (autopilot.upload.youtube)."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------


class TestYouTubePublicAPI:
    """Verify UploadError, upload_video surface."""

    def test_upload_error_importable(self):
        """UploadError is importable from youtube module."""
        from autopilot.upload.youtube import UploadError

        assert UploadError is not None

    def test_upload_error_is_exception(self):
        """UploadError is a subclass of Exception."""
        from autopilot.upload.youtube import UploadError

        assert issubclass(UploadError, Exception)
        err = UploadError("test error")
        assert str(err) == "test error"

    def test_upload_video_signature(self):
        """upload_video has narrative_id, video_path, db, config params."""
        from autopilot.upload.youtube import upload_video

        sig = inspect.signature(upload_video)
        params = list(sig.parameters.keys())
        assert "narrative_id" in params
        assert "video_path" in params
        assert "db" in params
        assert "config" in params

    def test_all_exports(self):
        """__all__ includes UploadError and upload_video."""
        from autopilot.upload import youtube

        assert "UploadError" in youtube.__all__
        assert "upload_video" in youtube.__all__
