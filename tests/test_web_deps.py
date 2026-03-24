"""Tests for autopilot.web.deps shared route helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from autopilot.web.deps import get_db, is_htmx


def _make_request(*, db_path: str | None = None, headers: dict[str, str] | None = None) -> MagicMock:
    """Build a mock Request with optional app.state.db_path and headers."""
    request = MagicMock()
    if db_path is not None:
        request.app.state.db_path = db_path
    if headers is not None:
        request.headers.get = lambda key, default=None: headers.get(key, default)
    else:
        request.headers.get = lambda key, default=None: default
    return request


# --- get_db tests ---


class TestGetDb:
    def test_get_db_returns_catalog_db(self, tmp_path):
        """get_db() should return a CatalogDB instance."""
        from autopilot.db import CatalogDB

        db_file = str(tmp_path / "test.db")
        request = _make_request(db_path=db_file)
        result = get_db(request)
        assert isinstance(result, CatalogDB)

    def test_get_db_uses_app_db_path(self, tmp_path):
        """The returned CatalogDB should be connected to app.state.db_path."""
        db_file = str(tmp_path / "test.db")
        request = _make_request(db_path=db_file)
        result = get_db(request)
        # Verify connection is functional by executing a trivial query
        cursor = result.conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1
        result.close()


# --- is_htmx tests ---


class TestIsHtmx:
    def test_is_htmx_true_when_header_present(self):
        """is_htmx() should return True when hx-request header is 'true'."""
        request = _make_request(headers={"hx-request": "true"})
        assert is_htmx(request) is True

    def test_is_htmx_false_when_header_absent(self):
        """is_htmx() should return False when hx-request header is absent."""
        request = _make_request(headers={})
        assert is_htmx(request) is False

    def test_is_htmx_false_for_non_true_value(self):
        """is_htmx() should return False for non-'true' header values."""
        request = _make_request(headers={"hx-request": "false"})
        assert is_htmx(request) is False
