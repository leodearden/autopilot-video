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


# ---------------------------------------------------------------------------
# OAuth credential loading tests
# ---------------------------------------------------------------------------


class TestLoadCredentials:
    """Verify _load_credentials helper."""

    def test_loads_credentials_from_config_path(self, tmp_path):
        """Loads OAuth2 credentials from the given file path."""
        creds_file = tmp_path / "oauth.json"
        creds_file.write_text("{}")

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.expired = False

        with patch(
            "autopilot.upload.youtube.Credentials"
        ) as mock_creds_cls:
            mock_creds_cls.from_authorized_user_file.return_value = mock_creds
            from autopilot.upload.youtube import _load_credentials

            result = _load_credentials(creds_file)

        mock_creds_cls.from_authorized_user_file.assert_called_once_with(
            str(creds_file),
        )
        assert result is mock_creds

    def test_raises_upload_error_when_credentials_file_missing(self, tmp_path):
        """Raises UploadError when credentials file does not exist."""
        from autopilot.upload.youtube import UploadError, _load_credentials

        missing = tmp_path / "nonexistent.json"
        with pytest.raises(UploadError, match="credentials"):
            _load_credentials(missing)

    def test_refreshes_expired_credentials(self, tmp_path):
        """Refreshes credentials when expired but refresh_token available."""
        creds_file = tmp_path / "oauth.json"
        creds_file.write_text("{}")

        mock_creds = MagicMock()
        mock_creds.valid = False
        mock_creds.expired = True
        mock_creds.refresh_token = "some-refresh-token"

        with (
            patch(
                "autopilot.upload.youtube.Credentials"
            ) as mock_creds_cls,
            patch(
                "autopilot.upload.youtube.Request"
            ) as mock_request_cls,
        ):
            mock_creds_cls.from_authorized_user_file.return_value = mock_creds
            from autopilot.upload.youtube import _load_credentials

            result = _load_credentials(creds_file)

        mock_creds.refresh.assert_called_once_with(mock_request_cls())
        assert result is mock_creds
