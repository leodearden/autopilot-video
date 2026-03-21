"""Tests for the FastAPI web application skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

from autopilot.web.app import create_app


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with a temporary db path."""
    app = create_app(str(tmp_path / "catalog.db"))

    # Add a test-only route that renders base.html for template testing
    @app.get("/test-base-template")
    def _test_base_template(request: Request):
        return app.state.templates.TemplateResponse(
            request, "base.html", {"page_title": "Test"}
        )

    return app


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


class TestStaticFiles:
    """Tests for static file serving."""

    def test_app_css_served(self, client) -> None:
        """GET /static/app.css returns 200."""
        response = client.get("/static/app.css")
        assert response.status_code == 200

    def test_app_js_served(self, client) -> None:
        """GET /static/app.js returns 200."""
        response = client.get("/static/app.js")
        assert response.status_code == 200


class TestBaseTemplate:
    """Tests for the base.html Jinja2 template."""

    def test_base_template_contains_tailwind_cdn(self, client) -> None:
        """base.html includes the Tailwind CSS CDN play script."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "tailwindcss" in response.text.lower() or "cdn.tailwindcss.com" in response.text

    def test_base_template_contains_htmx(self, client) -> None:
        """base.html includes the HTMX CDN script."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "htmx" in response.text.lower()

    def test_base_template_has_nav_items(self, client) -> None:
        """base.html nav contains all required menu items."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        html = response.text
        for item in ["Dashboard", "Media", "Pipeline", "Review", "Gates", "Settings"]:
            assert item in html, f"Nav item '{item}' not found in base.html"

    def test_base_template_has_notification_bell(self, client) -> None:
        """base.html contains a notification bell placeholder."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "notification" in response.text.lower()

    def test_base_template_has_toast_container(self, client) -> None:
        """base.html contains a toast container div."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "toast-container" in response.text

    def test_base_template_has_dark_theme(self, client) -> None:
        """base.html uses dark theme classes."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "dark" in response.text
