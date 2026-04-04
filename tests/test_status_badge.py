"""Tests for macros/status_badge.html — generic status badge macro."""

from __future__ import annotations

from pathlib import Path

import jinja2

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "autopilot" / "web" / "templates"


def _render_badge(
    status: str,
    color_map: dict[str, str],
    extra_classes: str = "",
    label: str | None = None,
    default: str = "bg-gray-700 text-gray-300",
) -> str:
    """Render the generic status_badge macro with the given arguments."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("macros/status_badge.html")
    module = template.module
    kwargs: dict = {"color_map": color_map, "extra_classes": extra_classes, "default": default}
    if label is not None:
        kwargs["label"] = label
    return module.status_badge(status, **kwargs)  # type: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# Unit tests for the generic macro
# ---------------------------------------------------------------------------


class TestGenericStatusBadge:
    """Verify color_map lookup and fallback behavior."""

    SAMPLE_MAP = {
        "running": "bg-blue-900 text-blue-300",
        "completed": "bg-green-900 text-green-300",
        "failed": "bg-red-900 text-red-300",
    }

    def test_known_status_returns_correct_classes(self) -> None:
        html = _render_badge("running", self.SAMPLE_MAP)
        assert "bg-blue-900" in html
        assert "text-blue-300" in html

    def test_unknown_status_falls_back_to_default_gray(self) -> None:
        html = _render_badge("cancelled", self.SAMPLE_MAP)
        assert "bg-gray-700" in html
        assert "text-gray-300" in html

    def test_none_status_shows_unknown_text_and_default_color(self) -> None:
        html = _render_badge(None, self.SAMPLE_MAP)  # type: ignore[arg-type]
        assert "unknown" in html
        assert "bg-gray-700" in html
        assert "text-gray-300" in html

    def test_empty_string_status_shows_unknown(self) -> None:
        html = _render_badge("", self.SAMPLE_MAP)
        assert "unknown" in html

    def test_status_text_displayed(self) -> None:
        html = _render_badge("completed", self.SAMPLE_MAP)
        assert "completed" in html

    def test_base_classes_present(self) -> None:
        html = _render_badge("running", self.SAMPLE_MAP)
        assert "px-2" in html
        assert "py-1" in html
        assert "rounded" in html
        assert "text-xs" in html
        assert "font-medium" in html


class TestExtraClasses:
    """Verify the extra_classes parameter adds classes to the badge span."""

    SAMPLE_MAP = {"ok": "bg-green-900 text-green-300"}

    def test_extra_classes_included(self) -> None:
        html = _render_badge("ok", self.SAMPLE_MAP, extra_classes="ml-2 whitespace-nowrap")
        assert "ml-2" in html
        assert "whitespace-nowrap" in html

    def test_no_extra_classes_by_default(self) -> None:
        html = _render_badge("ok", self.SAMPLE_MAP)
        assert "ml-2" not in html


class TestLabelParameter:
    """Verify the label parameter overrides the displayed text."""

    SAMPLE_MAP = {"done": "bg-green-900 text-green-300"}

    def test_custom_label_overrides_status_text(self) -> None:
        html = _render_badge("done", self.SAMPLE_MAP, label="done: 5")
        assert "done: 5" in html

    def test_label_none_defaults_to_status_text(self) -> None:
        html = _render_badge("done", self.SAMPLE_MAP)
        assert "done" in html


class TestDefaultColorOverride:
    """Verify passing a custom default uses that color for unknown statuses."""

    SAMPLE_MAP = {"known": "bg-green-900 text-green-300"}

    def test_purple_default_for_unknown(self) -> None:
        html = _render_badge(
            "mystery",
            self.SAMPLE_MAP,
            default="bg-purple-900 text-purple-300",
        )
        assert "bg-purple-900" in html
        assert "text-purple-300" in html

    def test_purple_default_not_gray(self) -> None:
        html = _render_badge(
            "mystery",
            self.SAMPLE_MAP,
            default="bg-purple-900 text-purple-300",
        )
        assert "bg-gray-700" not in html
