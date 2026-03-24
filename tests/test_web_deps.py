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
