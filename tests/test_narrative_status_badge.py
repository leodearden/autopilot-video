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
        return module.status_badge(status, extra_classes)  # type: ignore[reportAttributeAccessIssue]
    return module.status_badge(status)  # type: ignore[reportAttributeAccessIssue]


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

    def test_none_status(self) -> None:
        html = _render_badge(None)  # type: ignore[arg-type]
        assert "unknown" in html
        assert "bg-purple-900" in html
        assert "text-purple-300" in html

    def test_empty_string_status(self) -> None:
        html = _render_badge("")
        assert "unknown" in html
        assert "bg-purple-900" in html
        assert "text-purple-300" in html

    def test_status_text_displayed(self) -> None:
        html = _render_badge("approved")
        assert "approved" in html


class TestExtraClasses:
    """Verify the extra_classes parameter adds classes to the badge span."""

    def test_extra_classes_included(self) -> None:
        html = _render_badge("approved", "ml-2 whitespace-nowrap")
        assert "ml-2" in html
        assert "whitespace-nowrap" in html

    def test_no_extra_classes_by_default(self) -> None:
        html = _render_badge("proposed")
        assert "ml-2" not in html


class TestSourceUsesGenericMacro:
    """Verify narrative_status_badge.html delegates to the generic macro."""

    def test_source_uses_generic_macro(self) -> None:
        source_path = TEMPLATES_DIR / "macros" / "narrative_status_badge.html"
        source = source_path.read_text()
        expected = "{% from 'macros/status_badge.html' import status_badge as _generic_badge %}"
        assert expected in source


class TestNarrativeCardBadge:
    """Verify narrative_card.html uses the shared status_badge macro."""

    def test_renders_approved_badge(self) -> None:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
        )
        template = env.get_template("partials/narrative_card.html")
        html = template.render(narrative={
            "narrative_id": "n-1",
            "status": "approved",
            "title": "Test",
            "description": "desc",
        })
        assert "bg-green-900" in html

    def test_source_uses_macro_import(self) -> None:
        source_path = TEMPLATES_DIR / "partials" / "narrative_card.html"
        source = source_path.read_text()
        assert "{% from 'macros/narrative_status_badge.html' import status_badge %}" in source


class TestNarrativeEditFormBadge:
    """Verify narrative_edit_form.html uses the shared status_badge macro with extra classes."""

    def test_renders_proposed_badge_with_extra_classes(self) -> None:
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=True,
        )
        template = env.get_template("partials/narrative_edit_form.html")
        html = template.render(narrative={
            "narrative_id": "n-2",
            "status": "proposed",
            "title": "Edit Test",
            "description": "desc",
        })
        assert "bg-amber-900" in html
        assert "ml-2" in html

    def test_source_uses_macro_import(self) -> None:
        source_path = TEMPLATES_DIR / "partials" / "narrative_edit_form.html"
        source = source_path.read_text()
        assert "{% from 'macros/narrative_status_badge.html' import status_badge %}" in source


class TestReviewNarrativesBadge:
    """Verify review/narratives.html uses the shared status_badge macro."""

    def test_source_uses_macro_import(self) -> None:
        source_path = TEMPLATES_DIR / "review" / "narratives.html"
        source = source_path.read_text()
        assert "{% from 'macros/narrative_status_badge.html' import status_badge %}" in source


class TestReviewScriptsBadge:
    """Verify review/scripts.html uses the shared status_badge macro."""

    def test_source_uses_macro_import(self) -> None:
        source_path = TEMPLATES_DIR / "review" / "scripts.html"
        source = source_path.read_text()
        assert "{% from 'macros/narrative_status_badge.html' import status_badge %}" in source
