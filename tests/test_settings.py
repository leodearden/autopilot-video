"""Tests for the settings page."""

from __future__ import annotations

import re
from pathlib import Path

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


class TestNavLinkContract:
    """Regression guard: every nav link in base.html must resolve to a non-404 route."""

    @staticmethod
    def _nav_hrefs() -> list[str]:
        """Extract all href values from the <nav> section of base.html."""
        base_html = (
            Path(__file__).resolve().parent.parent
            / "autopilot"
            / "web"
            / "templates"
            / "base.html"
        )
        html = base_html.read_text()

        # Isolate the <nav>...</nav> block
        nav_match = re.search(r"<nav\b[^>]*>(.*?)</nav>", html, re.DOTALL)
        assert nav_match is not None, "<nav> block not found in base.html"

        # Extract all href="..." values within the nav
        return re.findall(r'href="([^"]+)"', nav_match.group(1))

    def test_all_nav_links_return_non_404(self, client) -> None:
        """Every href in the base.html nav bar resolves to a non-404 route."""
        hrefs = self._nav_hrefs()
        assert hrefs, "No nav links found in base.html — test is misconfigured"

        not_found = []
        for href in hrefs:
            response = client.get(href)
            if response.status_code == 404:
                not_found.append(href)

        assert not not_found, (
            f"Nav links returning 404 (missing route handlers): {not_found}"
        )
