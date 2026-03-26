"""Tests for settings page."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.web.app import create_app


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with a file-backed CatalogDB via tmp_path."""
    db_path = str(tmp_path / "catalog.db")
    return create_app(db_path)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a TestClient for the app."""
    return TestClient(app)


class TestSettingsPage:
    """Tests for GET /settings HTML page."""

    def test_settings_page_returns_200(self, client: TestClient) -> None:
        """GET /settings returns 200 with text/html."""
        response = client.get("/settings")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_settings_page_has_title(self, client: TestClient) -> None:
        """HTML contains 'Settings' as the page title."""
        response = client.get("/settings")
        assert "Settings" in response.text
        assert "Settings - Autopilot Video" in response.text

    def test_settings_page_extends_base(self, client: TestClient) -> None:
        """Settings page extends base.html (contains nav and toast container)."""
        response = client.get("/settings")
        html = response.text
        # Nav bar from base.html
        assert "Autopilot Video" in html
        assert "Dashboard" in html
        # Toast container from base.html
        assert "toast-container" in html


class TestSettingsContent:
    """Tests for settings page content sections."""

    def test_notification_preferences_section(self, client: TestClient) -> None:
        """Settings page contains a 'Notification Preferences' section."""
        response = client.get("/settings")
        assert "Notification Preferences" in response.text

    def test_console_preferences_section(self, client: TestClient) -> None:
        """Settings page contains a 'Console Preferences' section."""
        response = client.get("/settings")
        assert "Console Preferences" in response.text

    def test_notification_toggle_control(self, client: TestClient) -> None:
        """Notification preferences section has a browser notifications toggle."""
        response = client.get("/settings")
        assert "Browser Notifications" in response.text

    def test_theme_placeholder(self, client: TestClient) -> None:
        """Console preferences section has a theme placeholder."""
        response = client.get("/settings")
        assert "Theme" in response.text
