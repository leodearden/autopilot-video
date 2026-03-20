"""Tests for B-roll sourcing (autopilot.source.broll)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autopilot.source import BrollRequest


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_broll_request(
    description: str = "aerial view of mountains",
    duration: float = 5.0,
    start_time: str = "00:02:00.000",
) -> BrollRequest:
    """Create a BrollRequest for testing."""
    return BrollRequest(description=description, duration=duration, start_time=start_time)


def _make_pexels_response(videos: list[dict] | None = None) -> MagicMock:
    """Create a mock Pexels API response."""
    response = MagicMock()
    response.status_code = 200
    response.raise_for_status = MagicMock()
    if videos is None:
        videos = [
            {
                "id": 1001,
                "video_files": [
                    {"link": "https://example.com/vid1.mp4", "quality": "hd", "width": 1920},
                ],
            },
            {
                "id": 1002,
                "video_files": [
                    {"link": "https://example.com/vid2.mp4", "quality": "hd", "width": 1920},
                ],
            },
            {
                "id": 1003,
                "video_files": [
                    {"link": "https://example.com/vid3.mp4", "quality": "hd", "width": 1920},
                ],
            },
        ]
    response.json.return_value = {"videos": videos}
    return response


def _make_download_response(content: bytes = b"fake_video_data") -> MagicMock:
    """Create a mock download response."""
    response = MagicMock()
    response.status_code = 200
    response.content = content
    response.raise_for_status = MagicMock()
    return response


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------

class TestBrollPublicAPI:
    """Verify BrollError, source_broll surface."""

    def test_broll_error_importable(self):
        """BrollError is importable from broll module."""
        from autopilot.source.broll import BrollError

        assert BrollError is not None

    def test_broll_error_is_exception(self):
        """BrollError is a subclass of Exception."""
        from autopilot.source.broll import BrollError

        assert issubclass(BrollError, Exception)
        err = BrollError("test error")
        assert str(err) == "test error"

    def test_source_broll_signature(self):
        """source_broll has request, output_dir params."""
        from autopilot.source.broll import source_broll

        sig = inspect.signature(source_broll)
        params = list(sig.parameters.keys())
        assert "request" in params
        assert "output_dir" in params

    def test_all_exports(self):
        """__all__ includes BrollError and source_broll."""
        from autopilot.source import broll

        assert "BrollError" in broll.__all__
        assert "source_broll" in broll.__all__


# ---------------------------------------------------------------------------
# Pexels API tests
# ---------------------------------------------------------------------------

class TestPexelsSearch:
    """Tests for the Pexels video API search."""

    def test_pexels_returns_downloaded_files(self, tmp_path):
        """Pexels search downloads top-3 results and returns list of Paths."""
        mock_requests = MagicMock()
        search_resp = _make_pexels_response()
        dl_resp = _make_download_response()
        # search + 3 downloads
        mock_requests.get.side_effect = [search_resp, dl_resp, dl_resp, dl_resp]

        request = _make_broll_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.broll" in sys.modules:
                del sys.modules["autopilot.source.broll"]
            from autopilot.source.broll import source_broll

            with patch.dict("os.environ", {"PEXELS_API_KEY": "test-pexels-key"}):
                result = source_broll(request, tmp_path)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) <= 3
        for p in result:
            assert isinstance(p, Path)

    def test_pexels_uses_api_key(self, tmp_path):
        """Pexels search uses PEXELS_API_KEY from environment."""
        mock_requests = MagicMock()
        search_resp = _make_pexels_response(videos=[])
        mock_requests.get.return_value = search_resp

        request = _make_broll_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.broll" in sys.modules:
                del sys.modules["autopilot.source.broll"]
            from autopilot.source.broll import _search_pexels

            with patch.dict("os.environ", {"PEXELS_API_KEY": "my-pexels-key"}):
                _search_pexels(request, tmp_path)

        # Verify API key was used in headers
        call_args = mock_requests.get.call_args
        assert "my-pexels-key" in str(call_args)


# ---------------------------------------------------------------------------
# Pixabay fallback tests
# ---------------------------------------------------------------------------

class TestPixabayFallback:
    """Tests for the Pixabay fallback when Pexels returns no results."""

    def test_pixabay_fallback_on_empty_pexels(self, tmp_path):
        """When Pexels returns no results, Pixabay is tried."""
        mock_requests = MagicMock()
        # Pexels returns empty
        pexels_resp = _make_pexels_response(videos=[])
        # Pixabay returns results
        pixabay_resp = MagicMock()
        pixabay_resp.status_code = 200
        pixabay_resp.raise_for_status = MagicMock()
        pixabay_resp.json.return_value = {
            "hits": [
                {
                    "id": 2001,
                    "videos": {"medium": {"url": "https://example.com/pixabay1.mp4"}},
                },
            ]
        }
        dl_resp = _make_download_response()
        mock_requests.get.side_effect = [pexels_resp, pixabay_resp, dl_resp]

        request = _make_broll_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.broll" in sys.modules:
                del sys.modules["autopilot.source.broll"]
            from autopilot.source.broll import source_broll

            with patch.dict("os.environ", {
                "PEXELS_API_KEY": "test-key",
                "PIXABAY_API_KEY": "test-pixabay-key",
            }):
                result = source_broll(request, tmp_path)

        assert result is not None


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestBrollErrors:
    """Error handling in B-roll sourcing."""

    def test_returns_none_when_both_apis_fail(self, tmp_path):
        """Returns None when both Pexels and Pixabay return no results."""
        mock_requests = MagicMock()
        pexels_resp = _make_pexels_response(videos=[])
        pixabay_resp = MagicMock()
        pixabay_resp.status_code = 200
        pixabay_resp.raise_for_status = MagicMock()
        pixabay_resp.json.return_value = {"hits": []}
        mock_requests.get.side_effect = [pexels_resp, pixabay_resp]

        request = _make_broll_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.broll" in sys.modules:
                del sys.modules["autopilot.source.broll"]
            from autopilot.source.broll import source_broll

            with patch.dict("os.environ", {
                "PEXELS_API_KEY": "test-key",
                "PIXABAY_API_KEY": "test-key",
            }):
                result = source_broll(request, tmp_path)

        assert result is None

    def test_returns_none_when_no_api_keys(self, tmp_path):
        """Returns None when no API keys are set."""
        request = _make_broll_request()

        if "autopilot.source.broll" in sys.modules:
            del sys.modules["autopilot.source.broll"]
        from autopilot.source.broll import source_broll

        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("PEXELS_API_KEY", None)
            os.environ.pop("PIXABAY_API_KEY", None)
            result = source_broll(request, tmp_path)

        assert result is None
