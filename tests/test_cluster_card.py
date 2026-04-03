"""Tests for cluster_card.html template — querySelector null-guard."""

from __future__ import annotations

import re
from pathlib import Path

import jinja2

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "autopilot" / "web" / "templates"

# Regex matching *unguarded* querySelector(...).value — the dot after ) is NOT preceded by ?
# This pattern will match: querySelector("[name=label]").value
# but NOT match:           querySelector("[name=label]")?.value
UNGUARDED_QS_RE = re.compile(r'querySelector\([^)]+\)\.value')

# Regex matching null-safe querySelector(...)?.value
GUARDED_QS_RE = re.compile(r'querySelector\([^)]+\)\?\.value')


def _render_cluster_card(*, excluded: int = 0) -> str:
    """Render the cluster_card.html partial with a sample cluster context."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("partials/cluster_card.html")
    return template.render(
        cluster={
            "cluster_id": "c-001",
            "label": "Morning walk",
            "description": "Walking through the park",
            "clip_count": 5,
            "time_start": "08:00",
            "time_end": "08:30",
            "excluded": excluded,
        }
    )


class TestRelabelHxVals:
    """Verify the Relabel button's hx-vals uses null-safe querySelector access."""

    def test_no_unguarded_queryselector(self) -> None:
        """Rendered HTML must not contain querySelector(...).value without optional chaining."""
        html = _render_cluster_card(excluded=0)
        matches = UNGUARDED_QS_RE.findall(html)
        assert matches == [], (
            f"Found unguarded querySelector().value patterns: {matches}"
        )

    def test_has_null_safe_access(self) -> None:
        """Rendered HTML must contain at least 2 null-safe querySelector(...)?.value accesses."""
        html = _render_cluster_card(excluded=0)
        matches = GUARDED_QS_RE.findall(html)
        assert len(matches) >= 2, (
            f"Expected ≥2 null-safe querySelector accesses, found {len(matches)}: {matches}"
        )

    def test_relabel_no_json_content_type_header(self) -> None:
        """Relabel button must not set hx-headers with Content-Type application/json."""
        html = _render_cluster_card(excluded=0)
        pattern = re.compile(r'hx-headers=.*Content-Type.*application/json')
        matches = pattern.findall(html)
        assert matches == [], (
            f"Found hx-headers with JSON Content-Type override: {matches}"
        )
