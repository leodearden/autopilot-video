"""Tests for the FastAPI web application skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from autopilot.web.app import create_app


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with a temporary db path."""
    return create_app(str(tmp_path / "catalog.db"))


@pytest.fixture
def client(app: FastAPI):
    """Create a TestClient for the app."""
    from starlette.testclient import TestClient

    return TestClient(app)


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_create_app_returns_fastapi_instance(self, tmp_path: Path) -> None:
        """create_app(db_path) returns a FastAPI instance."""
        db_path = str(tmp_path / "catalog.db")
        app = create_app(db_path)
        assert isinstance(app, FastAPI)


class TestHealthCheck:
    """Tests for the /api/health endpoint."""

    def test_health_returns_200_ok(self, client) -> None:
        """GET /api/health returns 200 with {'status': 'ok'}."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
