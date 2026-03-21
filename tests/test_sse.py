"""Tests for the SSE event endpoint at GET /api/events."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app


@pytest.fixture
def sse_db(tmp_path: Path) -> CatalogDB:
    """Create a CatalogDB backed by a real file so the app can connect to it."""
    db_path = str(tmp_path / "catalog.db")
    db = CatalogDB(db_path)
    db.conn.isolation_level = None  # autocommit for test convenience
    db._test_path = db_path  # stash for fixture use
    return db


@pytest.fixture
def sse_app(sse_db: CatalogDB) -> FastAPI:
    """Create a FastAPI app pointing at the same DB as sse_db."""
    return create_app(sse_db._test_path)


@pytest.fixture
def sse_client(sse_app: FastAPI):
    """Create a TestClient for SSE tests."""
    return TestClient(sse_app)


class TestSSEEndpointBasic:
    """Tests for the basic SSE endpoint behavior."""

    def test_sse_endpoint_returns_200_event_stream(self, sse_app) -> None:
        """GET /api/events returns 200 with content-type text/event-stream.

        We patch the generator to yield one event then stop, avoiding the
        infinite stream that would hang tests.
        """
        from autopilot.web.routes import sse as sse_module

        async def _finite_gen(request):
            yield {"data": "hello"}

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            response = client.get("/api/events")
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
