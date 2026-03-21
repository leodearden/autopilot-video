"""Tests for the SSE event endpoint at GET /api/events."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI

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
    from starlette.testclient import TestClient

    return TestClient(sse_app)


class TestSSEEndpointBasic:
    """Tests for the basic SSE endpoint behavior."""

    def test_sse_endpoint_returns_200_event_stream(self, sse_client) -> None:
        """GET /api/events returns 200 with content-type text/event-stream."""
        with sse_client.stream("GET", "/api/events") as response:
            assert response.status_code == 200
            content_type = response.headers["content-type"]
            assert "text/event-stream" in content_type
