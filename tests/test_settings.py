"""Tests for the settings page."""

from __future__ import annotations

import pytest


class TestSettingsPage:
    """Tests for the GET /settings endpoint."""

    def test_settings_returns_200(self, client) -> None:
        """GET /settings returns 200."""
        response = client.get("/settings")
        assert response.status_code == 200

    def test_settings_contains_page_title(self, client) -> None:
        """GET /settings HTML contains the Settings page title."""
        response = client.get("/settings")
        assert "Settings" in response.text

    def test_settings_has_console_preferences_section(self, client) -> None:
        """GET /settings contains the Console Preferences section."""
        response = client.get("/settings")
        assert "Console Preferences" in response.text

    def test_settings_has_notification_section(self, client) -> None:
        """GET /settings contains the Notification Configuration section."""
        response = client.get("/settings")
        assert "Notification Configuration" in response.text

    def test_settings_extends_base_template(self, client) -> None:
        """GET /settings uses the base template (has nav items)."""
        response = client.get("/settings")
        html = response.text
        assert "Dashboard" in html
        assert "Autopilot Video" in html
