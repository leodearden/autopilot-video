"""Tests for thumbnail extraction (autopilot.upload.thumbnail)."""

from __future__ import annotations

import inspect
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


# ---------------------------------------------------------------------------
# Mock helpers for cv2
# ---------------------------------------------------------------------------


def _setup_cv2_mock():
    """Create a mock cv2 module for testing."""
    mock_cv2 = MagicMock()
    mock_cv2.CV_64F = 6
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    return mock_cv2


# ---------------------------------------------------------------------------
# Frame scoring tests
# ---------------------------------------------------------------------------


class TestScoringFunctions:
    """Verify individual frame scoring functions."""

    def test_sharpness_score_uses_laplacian_variance(self):
        """Sharpness is computed from Laplacian variance."""
        mock_cv2 = _setup_cv2_mock()
        # Simulate a Laplacian result with known variance
        laplacian_result = np.array([[10.0, 20.0], [30.0, 40.0]])
        mock_cv2.Laplacian.return_value = laplacian_result

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.upload.thumbnail import _sharpness_score

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            score = _sharpness_score(frame)

        mock_cv2.Laplacian.assert_called_once()
        assert isinstance(score, float)
        assert score > 0

    def test_rule_of_thirds_score_high_at_intersections(self):
        """Subject at rule-of-thirds intersection scores higher than center."""
        from autopilot.upload.thumbnail import _rule_of_thirds_score

        frame_shape = (1080, 1920, 3)
        # Detection bbox centered at a thirds intersection point (~640, 360)
        thirds_dets = [
            {"bbox": [590, 310, 690, 410], "confidence": 0.9}
        ]
        # Detection bbox centered at dead center (960, 540)
        center_dets = [
            {"bbox": [910, 490, 1010, 590], "confidence": 0.9}
        ]

        thirds_score = _rule_of_thirds_score(frame_shape, thirds_dets)
        center_score = _rule_of_thirds_score(frame_shape, center_dets)

        assert thirds_score > center_score

    def test_detection_confidence_score_aggregates(self):
        """Uses max confidence from list of detections."""
        from autopilot.upload.thumbnail import _detection_confidence_score

        detections = [
            {"class_name": "person", "confidence": 0.7},
            {"class_name": "car", "confidence": 0.95},
            {"class_name": "dog", "confidence": 0.8},
        ]
        score = _detection_confidence_score(detections)
        assert score == pytest.approx(0.95)

    def test_detection_confidence_empty(self):
        """Returns 0 for empty detections list."""
        from autopilot.upload.thumbnail import _detection_confidence_score

        assert _detection_confidence_score([]) == 0.0

    def test_combined_score_weighted(self):
        """Combined score uses weights summing to 1.0."""
        from autopilot.upload.thumbnail import _combined_score

        score = _combined_score(
            sharpness=1.0, thirds=1.0, confidence=1.0
        )
        assert score == pytest.approx(1.0)

        # Weights: 0.3 sharpness, 0.3 thirds, 0.4 confidence
        score2 = _combined_score(
            sharpness=0.5, thirds=0.5, confidence=0.5
        )
        assert score2 == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Frame extraction tests
# ---------------------------------------------------------------------------


class TestFrameExtraction:
    """Verify frame extraction and best-frame selection."""

    def _make_mock_cap(self, num_frames=30, fps=30.0):
        """Create a mock cv2.VideoCapture with N frames."""
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            7: float(num_frames),  # CAP_PROP_FRAME_COUNT
            5: fps,  # CAP_PROP_FPS
        }.get(prop, 0.0)
        mock_cap.isOpened.return_value = True
        # Return frames: each read returns (True, numpy array)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        return mock_cap

    def test_extracts_best_frame_as_jpeg(self, tmp_path):
        """Highest-scoring frame is saved as JPEG."""
        mock_cv2 = _setup_cv2_mock()
        mock_cap = self._make_mock_cap(num_frames=90, fps=30.0)
        mock_cv2.VideoCapture.return_value = mock_cap

        laplacian_result = np.random.randn(100, 100)
        mock_cv2.Laplacian.return_value = laplacian_result
        mock_cv2.imwrite.return_value = True

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"\x00" * 100)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.upload.thumbnail import _extract_best_frame

            result = _extract_best_frame(video_path, [])

        assert result is not None
        mock_cv2.imwrite.assert_called_once()
        # Verify saved as JPEG
        save_path = mock_cv2.imwrite.call_args[0][0]
        assert save_path.endswith(".jpg")

    def test_samples_frames_at_intervals(self, tmp_path):
        """Frames are sampled at ~1fps intervals, not every frame."""
        mock_cv2 = _setup_cv2_mock()
        mock_cap = self._make_mock_cap(num_frames=300, fps=30.0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.Laplacian.return_value = np.zeros((10, 10))
        mock_cv2.imwrite.return_value = True

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"\x00" * 100)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.upload.thumbnail import _extract_best_frame

            _extract_best_frame(video_path, [])

        # Should sample ~10 frames (300 frames / 30 fps = 10 seconds)
        # not all 300 frames
        set_calls = [
            c
            for c in mock_cap.set.call_args_list
            if c[0][0] == 1  # CAP_PROP_POS_FRAMES
        ]
        assert 5 <= len(set_calls) <= 15

    def test_returns_thumbnail_path(self, tmp_path):
        """Returns Path to the saved JPEG in video_path.parent."""
        mock_cv2 = _setup_cv2_mock()
        mock_cap = self._make_mock_cap(num_frames=30, fps=30.0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.Laplacian.return_value = np.zeros((10, 10))
        mock_cv2.imwrite.return_value = True

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"\x00" * 100)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.upload.thumbnail import _extract_best_frame

            result = _extract_best_frame(video_path, [])

        assert result is not None
        assert isinstance(result, Path)
        assert result.parent == tmp_path

    def test_video_capture_released_on_scoring_exception(self, tmp_path):
        """cap.release() called even when scoring raises an exception."""
        mock_cv2 = _setup_cv2_mock()
        mock_cap = self._make_mock_cap(num_frames=90, fps=30.0)
        mock_cv2.VideoCapture.return_value = mock_cap
        # Laplacian raises on first call, simulating a malformed frame
        mock_cv2.Laplacian.side_effect = RuntimeError("malformed frame")

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"\x00" * 100)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.upload.thumbnail import _extract_best_frame

            with pytest.raises(RuntimeError, match="malformed frame"):
                _extract_best_frame(video_path, [])

        # Verify cap.release() was called despite the exception
        mock_cap.release.assert_called_once()

    def test_imwrite_failure_returns_none(self, tmp_path):
        """Returns None when cv2.imwrite returns False (disk full, etc)."""
        mock_cv2 = _setup_cv2_mock()
        mock_cap = self._make_mock_cap(num_frames=30, fps=30.0)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.Laplacian.return_value = np.zeros((10, 10))
        # Simulate imwrite failure (disk full, permission denied)
        mock_cv2.imwrite.return_value = False

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"\x00" * 100)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.upload.thumbnail import _extract_best_frame

            result = _extract_best_frame(video_path, [])

        # Should return None, not a path to a non-existent file
        assert result is None


# ---------------------------------------------------------------------------
# Google API mock helpers
# ---------------------------------------------------------------------------


def _setup_google_mocks():
    """Install mock google.* modules in sys.modules for testing."""
    mods = {
        "google": MagicMock(),
        "google.auth": MagicMock(),
        "google.auth.transport": MagicMock(),
        "google.auth.transport.requests": MagicMock(),
        "google.oauth2": MagicMock(),
        "google.oauth2.credentials": MagicMock(),
        "googleapiclient": MagicMock(),
        "googleapiclient.discovery": MagicMock(),
        "googleapiclient.errors": MagicMock(),
        "googleapiclient.http": MagicMock(),
    }
    return mods


# ---------------------------------------------------------------------------
# Thumbnail upload to YouTube tests
# ---------------------------------------------------------------------------


class TestThumbnailUpload:
    """Verify YouTube thumbnail upload via thumbnails().set() API."""

    def _make_extract_env(self, tmp_path):
        """Set up cv2 mock + video file for frame extraction."""
        mock_cv2 = _setup_cv2_mock()
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            7: 30.0,  # FRAME_COUNT
            5: 30.0,  # FPS
        }.get(prop, 0.0)
        mock_cap.isOpened.return_value = True
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.Laplacian.return_value = np.zeros((10, 10))
        mock_cv2.imwrite.return_value = True

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"\x00" * 100)

        return mock_cv2, video_path

    def test_uploads_thumbnail_via_api(self, catalog_db, tmp_path):
        """Calls thumbnails().set() with youtube_video_id from DB."""
        from autopilot.upload.thumbnail import extract_best_thumbnail

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        catalog_db.insert_upload(
            "n1",
            youtube_video_id="abc123",
            youtube_url="https://youtu.be/abc123",
        )

        mock_cv2, video_path = self._make_extract_env(tmp_path)
        google_mods = _setup_google_mocks()

        mock_creds = MagicMock()
        mock_creds.expired = False
        google_mods["google.oauth2.credentials"].Credentials \
            .from_authorized_user_file.return_value = mock_creds

        mock_youtube = MagicMock()
        google_mods["googleapiclient.discovery"].build.return_value = (
            mock_youtube
        )

        # Create a fake credentials file
        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")

        all_mods = {"cv2": mock_cv2, **google_mods}
        with patch.dict(sys.modules, all_mods):
            with patch(
                "autopilot.upload.thumbnail._get_credentials_path",
                return_value=creds_file,
            ):
                result = extract_best_thumbnail(
                    "n1", video_path, catalog_db
                )

        mock_youtube.thumbnails.return_value.set.assert_called_once()
        call_kwargs = (
            mock_youtube.thumbnails.return_value.set.call_args
        )
        assert call_kwargs[1]["videoId"] == "abc123"
        assert result is not None

    def test_thumbnail_upload_error_logged_not_raised(
        self, catalog_db, tmp_path
    ):
        """API error is logged as warning, does not raise."""
        from autopilot.upload.thumbnail import extract_best_thumbnail

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        catalog_db.insert_upload(
            "n1",
            youtube_video_id="abc123",
            youtube_url="https://youtu.be/abc123",
        )

        mock_cv2, video_path = self._make_extract_env(tmp_path)
        google_mods = _setup_google_mocks()

        mock_creds = MagicMock()
        mock_creds.expired = False
        google_mods["google.oauth2.credentials"].Credentials \
            .from_authorized_user_file.return_value = mock_creds

        mock_youtube = MagicMock()
        mock_youtube.thumbnails.return_value.set.return_value \
            .execute.side_effect = Exception("Thumbnail API error")
        google_mods["googleapiclient.discovery"].build.return_value = (
            mock_youtube
        )

        creds_file = tmp_path / "creds.json"
        creds_file.write_text("{}")

        all_mods = {"cv2": mock_cv2, **google_mods}
        with patch.dict(sys.modules, all_mods):
            with patch(
                "autopilot.upload.thumbnail._get_credentials_path",
                return_value=creds_file,
            ):
                # Should not raise - thumbnail upload errors are non-fatal
                result = extract_best_thumbnail(
                    "n1", video_path, catalog_db
                )

        assert result is not None

    def test_no_youtube_upload_skips_thumbnail_upload(
        self, catalog_db, tmp_path
    ):
        """No upload record in DB -> still returns thumbnail path."""
        from autopilot.upload.thumbnail import extract_best_thumbnail

        catalog_db.insert_narrative("n1", title="Test", description="desc")
        # No upload record inserted

        mock_cv2, video_path = self._make_extract_env(tmp_path)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            result = extract_best_thumbnail("n1", video_path, catalog_db)

        assert result is not None
        assert isinstance(result, Path)
