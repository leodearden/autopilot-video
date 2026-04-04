"""Tests for narrative_status_badge.html macro — shared status badge rendering."""

from __future__ import annotations

from pathlib import Path

import jinja2

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "autopilot" / "web" / "templates"


def _render_badge(status: str, extra_classes: str = "") -> str:
    """Render the status_badge macro with the given status and optional extra classes."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("macros/narrative_status_badge.html")
    module = template.module
    if extra_classes:
        return module.status_badge(status, extra_classes)
    return module.status_badge(status)


class TestStatusBadgeMacro:
    """Verify the macro renders correct color classes for each status."""

    def test_approved_green(self) -> None:
        html = _render_badge("approved")
        assert "bg-green-900" in html
        assert "text-green-300" in html

    def test_rejected_red(self) -> None:
        html = _render_badge("rejected")
        assert "bg-red-900" in html
        assert "text-red-300" in html

    def test_proposed_amber(self) -> None:
        html = _render_badge("proposed")
        assert "bg-amber-900" in html
        assert "text-amber-300" in html

    def test_scripted_blue(self) -> None:
        html = _render_badge("scripted")
        assert "bg-blue-900" in html
        assert "text-blue-300" in html

    def test_unknown_status_purple(self) -> None:
        html = _render_badge("draft")
        assert "bg-purple-900" in html
        assert "text-purple-300" in html

    def test_status_text_displayed(self) -> None:
        html = _render_badge("approved")
        assert "approved" in html
