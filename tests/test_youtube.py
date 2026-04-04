"""Tests for YouTube upload (autopilot.upload.youtube)."""

from __future__ import annotations

import inspect
import json
import sys
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


def _setup_google_mocks():
    """Install mock google.* modules in sys.modules for testing."""
    mock_google = MagicMock()
    mock_google_auth = MagicMock()
    mock_google_auth_transport = MagicMock()
    mock_google_auth_transport_requests = MagicMock()
    mock_google_oauth2 = MagicMock()
    mock_google_oauth2_credentials = MagicMock()

    mods = {
        "google": mock_google,
        "google.auth": mock_google_auth,
        "google.auth.transport": mock_google_auth_transport,
        "google.auth.transport.requests": mock_google_auth_transport_requests,
        "google.oauth2": mock_google_oauth2,
        "google.oauth2.credentials": mock_google_oauth2_credentials,
        "googleapiclient": MagicMock(),
        "googleapiclient.discovery": MagicMock(),
        "googleapiclient.errors": MagicMock(),
        "googleapiclient.http": MagicMock(),
    }
    return mods, mock_google_oauth2_credentials, mock_google_auth_transport_requests


class TestLoadCredentials:
    """Verify _load_credentials helper."""

    def test_loads_credentials_from_config_path(self, tmp_path):
        """Loads OAuth2 credentials from the given file path."""
        creds_file = tmp_path / "oauth.json"
        creds_file.write_text("{}")

        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_creds.expired = False

        mods, mock_oauth2_creds, _ = _setup_google_mocks()
        mock_creds_cls = MagicMock()
        mock_creds_cls.from_authorized_user_file.return_value = mock_creds
        mods["google.oauth2.credentials"].Credentials = mock_creds_cls

        with patch.dict(sys.modules, mods):
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

        mods, _, mock_transport_req = _setup_google_mocks()
        mock_creds_cls = MagicMock()
        mock_creds_cls.from_authorized_user_file.return_value = mock_creds
        mods["google.oauth2.credentials"].Credentials = mock_creds_cls
        mock_request_instance = MagicMock()
        mods["google.auth.transport.requests"].Request.return_value = mock_request_instance

        with patch.dict(sys.modules, mods):
            from autopilot.upload.youtube import _load_credentials

            result = _load_credentials(creds_file)

        mock_creds.refresh.assert_called_once_with(mock_request_instance)
        assert result is mock_creds


# ---------------------------------------------------------------------------
# Metadata assembly tests
# ---------------------------------------------------------------------------


class TestBuildUploadMetadata:
    """Verify _build_upload_metadata helper."""

    def test_builds_title_from_narrative(self, catalog_db):
        """Title comes from narrative title in DB."""
        from autopilot.upload.youtube import _build_upload_metadata

        catalog_db.insert_narrative("n1", title="My Great Video", description="desc")
        config = MagicMock()
        config.privacy_status = "unlisted"
        config.default_category = "22"

        meta = _build_upload_metadata("n1", catalog_db, config)
        assert meta["snippet"]["title"] == "My Great Video"

    def test_builds_description_from_script(self, catalog_db):
        """Description includes narrative description and script content."""
        from autopilot.upload.youtube import _build_upload_metadata

        catalog_db.insert_narrative("n1", title="Title", description="Narrative desc")
        catalog_db.upsert_narrative_script(
            "n1",
            json.dumps({"scenes": [{"narration": "Scene one narration."}]}),
        )
        config = MagicMock()
        config.privacy_status = "unlisted"
        config.default_category = "22"

        meta = _build_upload_metadata("n1", catalog_db, config)
        assert "Narrative desc" in meta["snippet"]["description"]

    def test_tags_from_activity_labels_and_detections(self, catalog_db):
        """Tags combine activity cluster labels and detected object classes."""
        from autopilot.upload.youtube import _build_upload_metadata

        catalog_db.insert_narrative(
            "n1",
            title="Title",
            description="desc",
            activity_cluster_ids_json=json.dumps(["c1", "c2"]),
        )
        catalog_db.insert_activity_cluster("c1", label="hiking")
        catalog_db.insert_activity_cluster("c2", label="camping")
        # Insert a media file + detections with class names
        catalog_db.insert_media("m1", file_path="/tmp/m1.mp4")
        catalog_db.batch_insert_detections(
            [
                (
                    "m1",
                    0,
                    json.dumps(
                        [
                            {"class": "person", "confidence": 0.9},
                            {"class": "backpack", "confidence": 0.8},
                        ]
                    ),
                ),
                (
                    "m1",
                    1,
                    json.dumps(
                        [
                            {"class": "person", "confidence": 0.85},
                            {"class": "tent", "confidence": 0.7},
                        ]
                    ),
                ),
            ]
        )
        config = MagicMock()
        config.privacy_status = "unlisted"
        config.default_category = "22"

        meta = _build_upload_metadata("n1", catalog_db, config)
        tags = meta["snippet"]["tags"]
        # Activity labels present
        assert "hiking" in tags
        assert "camping" in tags
        # Detection class names present (deduplicated)
        assert "person" in tags
        assert "backpack" in tags
        assert "tent" in tags

    def test_uses_config_privacy_status_and_category(self, catalog_db):
        """Privacy and category come from YouTubeConfig."""
        from autopilot.upload.youtube import _build_upload_metadata

        catalog_db.insert_narrative("n1", title="Title", description="desc")
        config = MagicMock()
        config.privacy_status = "private"
        config.default_category = "19"

        meta = _build_upload_metadata("n1", catalog_db, config)
        assert meta["snippet"]["categoryId"] == "19"
        assert meta["status"]["privacyStatus"] == "private"


# ---------------------------------------------------------------------------
# Full upload_video flow tests
# ---------------------------------------------------------------------------


class TestUploadVideoFlow:
    """Verify full upload_video function."""

    def _make_config(self, tmp_path):
        """Create a mock YouTubeConfig with a credentials file."""
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")
        config = MagicMock()
        config.credentials_path = creds_file
        config.privacy_status = "unlisted"
        config.default_category = "22"
        return config

    def test_upload_calls_youtube_api_insert(self, catalog_db, tmp_path):
        """Calls YouTube videos().insert() with MediaFileUpload."""
        from autopilot.upload.youtube import upload_video

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        config = self._make_config(tmp_path)

        mods, mock_oauth2_creds, _ = _setup_google_mocks()
        mock_creds = MagicMock()
        mock_creds.expired = False
        mods[
            "google.oauth2.credentials"
        ].Credentials.from_authorized_user_file.return_value = mock_creds

        mock_youtube = MagicMock()
        mock_insert_req = MagicMock()
        mock_insert_req.next_chunk.return_value = (
            None,
            {"id": "abc123"},
        )
        mock_youtube.videos.return_value.insert.return_value = mock_insert_req
        mods["googleapiclient.discovery"].build.return_value = mock_youtube

        with patch.dict(sys.modules, mods):
            result = upload_video("n1", video_file, catalog_db, config)

        mock_youtube.videos.return_value.insert.assert_called_once()
        assert result == "https://youtu.be/abc123"

    def test_upload_stores_result_in_db(self, catalog_db, tmp_path):
        """Stores upload record in catalog DB after upload."""
        from autopilot.upload.youtube import upload_video

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        config = self._make_config(tmp_path)

        mods, _, _ = _setup_google_mocks()
        mock_creds = MagicMock()
        mock_creds.expired = False
        mods[
            "google.oauth2.credentials"
        ].Credentials.from_authorized_user_file.return_value = mock_creds

        mock_youtube = MagicMock()
        mock_insert_req = MagicMock()
        mock_insert_req.next_chunk.return_value = (
            None,
            {"id": "xyz789"},
        )
        mock_youtube.videos.return_value.insert.return_value = mock_insert_req
        mods["googleapiclient.discovery"].build.return_value = mock_youtube

        with patch.dict(sys.modules, mods):
            upload_video("n1", video_file, catalog_db, config)

        upload_rec = catalog_db.get_upload("n1")
        assert upload_rec is not None
        assert upload_rec["youtube_video_id"] == "xyz789"
        assert upload_rec["youtube_url"] == "https://youtu.be/xyz789"

    def test_upload_returns_youtube_url(self, catalog_db, tmp_path):
        """Returns the YouTube URL for the uploaded video."""
        from autopilot.upload.youtube import upload_video

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        config = self._make_config(tmp_path)

        mods, _, _ = _setup_google_mocks()
        mock_creds = MagicMock()
        mock_creds.expired = False
        mods[
            "google.oauth2.credentials"
        ].Credentials.from_authorized_user_file.return_value = mock_creds

        mock_youtube = MagicMock()
        mock_insert_req = MagicMock()
        mock_insert_req.next_chunk.return_value = (
            None,
            {"id": "vid_001"},
        )
        mock_youtube.videos.return_value.insert.return_value = mock_insert_req
        mods["googleapiclient.discovery"].build.return_value = mock_youtube

        with patch.dict(sys.modules, mods):
            url = upload_video("n1", video_file, catalog_db, config)

        assert url == "https://youtu.be/vid_001"

    def test_upload_raises_on_api_error(self, catalog_db, tmp_path):
        """Wraps API HttpError as UploadError."""
        from autopilot.upload.youtube import UploadError, upload_video

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        config = self._make_config(tmp_path)

        mods, _, _ = _setup_google_mocks()
        mock_creds = MagicMock()
        mock_creds.expired = False
        mods[
            "google.oauth2.credentials"
        ].Credentials.from_authorized_user_file.return_value = mock_creds

        mock_youtube = MagicMock()
        mock_insert_req = MagicMock()
        mock_insert_req.next_chunk.side_effect = Exception("API quota exceeded")
        mock_youtube.videos.return_value.insert.return_value = mock_insert_req
        mods["googleapiclient.discovery"].build.return_value = mock_youtube

        with patch.dict(sys.modules, mods):
            with pytest.raises(UploadError, match="quota exceeded"):
                upload_video("n1", video_file, catalog_db, config)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestUploadVideoEdgeCases:
    """Verify upload_video edge cases."""

    def _make_config(self, tmp_path):
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")
        config = MagicMock()
        config.credentials_path = creds_file
        config.privacy_status = "unlisted"
        config.default_category = "22"
        return config

    def test_narrative_not_found_raises_upload_error(self, catalog_db, tmp_path):
        """Raises UploadError when narrative_id not in DB."""
        from autopilot.upload.youtube import UploadError, upload_video

        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        config = self._make_config(tmp_path)

        mods, _, _ = _setup_google_mocks()
        mock_creds = MagicMock()
        mock_creds.expired = False
        mods[
            "google.oauth2.credentials"
        ].Credentials.from_authorized_user_file.return_value = mock_creds

        with patch.dict(sys.modules, mods):
            with pytest.raises(UploadError, match="[Nn]arrative.*not found"):
                upload_video("nonexistent", video_file, catalog_db, config)

    def test_video_file_not_found_raises_upload_error(self, catalog_db, tmp_path):
        """Raises UploadError when video_path doesn't exist."""
        from autopilot.upload.youtube import UploadError, upload_video

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        missing_video = tmp_path / "nonexistent.mp4"
        config = self._make_config(tmp_path)

        mods, _, _ = _setup_google_mocks()
        mock_creds = MagicMock()
        mock_creds.expired = False
        mods[
            "google.oauth2.credentials"
        ].Credentials.from_authorized_user_file.return_value = mock_creds

        with patch.dict(sys.modules, mods):
            with pytest.raises(UploadError, match="[Vv]ideo.*not found"):
                upload_video("n1", missing_video, catalog_db, config)

    def test_upload_breaks_after_first_success(self, catalog_db, tmp_path):
        """Outer retry loop breaks after first successful upload."""
        from autopilot.upload.youtube import upload_video

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        config = self._make_config(tmp_path)

        mods, _, _ = _setup_google_mocks()
        mock_creds = MagicMock()
        mock_creds.expired = False
        mods[
            "google.oauth2.credentials"
        ].Credentials.from_authorized_user_file.return_value = mock_creds

        mock_youtube = MagicMock()
        mock_insert_req = MagicMock()
        # First call succeeds immediately
        mock_insert_req.next_chunk.return_value = (
            None,
            {"id": "ok123"},
        )
        mock_youtube.videos.return_value.insert.return_value = mock_insert_req
        mods["googleapiclient.discovery"].build.return_value = mock_youtube

        with patch.dict(sys.modules, mods):
            upload_video("n1", video_file, catalog_db, config)

        # If break is missing, next_chunk would be called 3 times
        # (once per retry iteration) because subsequent calls return
        # the same non-None response immediately, skipping the while.
        # With break, it should be called exactly once.
        assert mock_insert_req.next_chunk.call_count == 1

    def test_resumable_upload_retry(self, catalog_db, tmp_path):
        """Retries on transient error then succeeds."""
        from autopilot.upload.youtube import upload_video

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"\x00" * 100)
        config = self._make_config(tmp_path)

        mods, _, _ = _setup_google_mocks()
        mock_creds = MagicMock()
        mock_creds.expired = False
        mods[
            "google.oauth2.credentials"
        ].Credentials.from_authorized_user_file.return_value = mock_creds

        mock_youtube = MagicMock()
        mock_insert_req = MagicMock()
        # First call raises, second succeeds
        mock_insert_req.next_chunk.side_effect = [
            Exception("transient error"),
            (None, {"id": "retry_ok"}),
        ]
        mock_youtube.videos.return_value.insert.return_value = mock_insert_req
        mods["googleapiclient.discovery"].build.return_value = mock_youtube

        with patch.dict(sys.modules, mods):
            url = upload_video("n1", video_file, catalog_db, config)

        assert url == "https://youtu.be/retry_ok"


# ---------------------------------------------------------------------------
# OAuth setup script tests
# ---------------------------------------------------------------------------


class TestSetupYoutubeOAuth:
    """Verify scripts/setup_youtube_oauth.py."""

    def test_oauth_flow_saves_credentials(self, tmp_path):
        """OAuth flow serializes and writes credentials to output path."""
        google_mods, _, _ = _setup_google_mocks()
        google_mods["google_auth_oauthlib"] = MagicMock()
        google_mods["google_auth_oauthlib.flow"] = MagicMock()

        mock_flow = MagicMock()
        mock_creds = MagicMock()
        mock_creds.to_json.return_value = '{"token": "test"}'
        mock_flow.run_local_server.return_value = mock_creds
        google_mods[
            "google_auth_oauthlib.flow"
        ].InstalledAppFlow.from_client_secrets_file.return_value = mock_flow

        output_path = tmp_path / "oauth.json"
        client_secrets = tmp_path / "client_secrets.json"
        client_secrets.write_text("{}")

        with patch.dict(sys.modules, google_mods):
            from scripts.setup_youtube_oauth import run_oauth_flow

            run_oauth_flow(
                client_secrets_path=client_secrets,
                output_path=output_path,
            )

        assert output_path.exists()
        assert output_path.read_text() == '{"token": "test"}'

    def test_main_requires_client_secrets_arg(self):
        """argparse requires --client-secrets argument."""
        google_mods, _, _ = _setup_google_mocks()
        google_mods["google_auth_oauthlib"] = MagicMock()
        google_mods["google_auth_oauthlib.flow"] = MagicMock()

        with patch.dict(sys.modules, google_mods):
            from scripts.setup_youtube_oauth import build_parser

            parser = build_parser()
            with pytest.raises(SystemExit):
                parser.parse_args([])
