"""Tests for autopilot.web.deps shared route helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import Request
from fastapi.responses import HTMLResponse

from autopilot.web.deps import get_db, is_htmx, render_partial


def _make_request(
    *, db_path: str | None = None, headers: dict[str, str] | None = None,
) -> MagicMock:
    """Build a mock Request with optional app.state.db_path and headers."""
    request = MagicMock()
    if db_path is not None:
        request.app.state.db_path = db_path
    if headers is not None:
        request.headers.get = lambda key, default=None: headers.get(key, default)
    else:
        request.headers.get = lambda key, default=None: default
    return request


# --- get_db tests ---


class TestGetDb:
    def test_get_db_returns_catalog_db(self, tmp_path):
        """get_db() should return a CatalogDB instance."""
        from autopilot.db import CatalogDB

        db_file = str(tmp_path / "test.db")
        request = _make_request(db_path=db_file)
        result = get_db(request)
        try:
            assert isinstance(result, CatalogDB)
        finally:
            result.close()

    def test_get_db_uses_app_db_path(self, tmp_path):
        """The returned CatalogDB should be connected to app.state.db_path."""
        db_file = str(tmp_path / "test.db")
        request = _make_request(db_path=db_file)
        result = get_db(request)
        try:
            # Verify connection is functional by executing a trivial query
            cursor = result.conn.execute("SELECT 1")
            assert cursor.fetchone()[0] == 1
        finally:
            result.close()


# --- is_htmx tests ---


class TestIsHtmx:
    def test_is_htmx_true_when_header_present(self):
        """is_htmx() should return True when hx-request header is 'true'."""
        request = _make_request(headers={"hx-request": "true"})
        assert is_htmx(request) is True

    def test_is_htmx_false_when_header_absent(self):
        """is_htmx() should return False when hx-request header is absent."""
        request = _make_request(headers={})
        assert is_htmx(request) is False

    def test_is_htmx_false_for_non_true_value(self):
        """is_htmx() should return False for non-'true' header values."""
        request = _make_request(headers={"hx-request": "false"})
        assert is_htmx(request) is False


# --- render_partial tests ---


class TestRenderPartial:
    """Unit tests for render_partial helper."""

    def _make_request(self, rendered_html: str = "<div>ok</div>") -> Request:
        """Create a mock Request with templates on app.state."""
        mock_template = MagicMock()
        mock_template.render.return_value = rendered_html

        mock_templates = MagicMock()
        mock_templates.get_template.return_value = mock_template

        request = MagicMock(spec=Request)
        request.app.state.templates = mock_templates
        return request

    def test_calls_get_template_with_name(self) -> None:
        """render_partial calls get_template with the given template_name."""
        request = self._make_request()
        render_partial(request, "partials/narrative_card.html")
        request.app.state.templates.get_template.assert_called_once_with(
            "partials/narrative_card.html",
        )

    def test_passes_ctx_kwargs_to_render(self) -> None:
        """render_partial passes **ctx keyword arguments to template.render."""
        request = self._make_request()
        render_partial(request, "partials/card.html", narrative={"id": "n-1"}, extra="val")
        template = request.app.state.templates.get_template.return_value
        template.render.assert_called_once_with(narrative={"id": "n-1"}, extra="val")

    def test_returns_html_response(self) -> None:
        """render_partial returns an HTMLResponse."""
        request = self._make_request(rendered_html="<p>hello</p>")
        result = render_partial(request, "partials/card.html")
        assert isinstance(result, HTMLResponse)

    def test_response_body_matches_rendered_content(self) -> None:
        """HTMLResponse body contains the template-rendered HTML."""
        request = self._make_request(rendered_html="<div>content</div>")
        result = render_partial(request, "partials/card.html")
        assert result.body == b"<div>content</div>"

    def test_no_ctx_renders_with_empty_kwargs(self) -> None:
        """render_partial with no extra kwargs passes empty kwargs to render."""
        request = self._make_request()
        render_partial(request, "partials/empty.html")
        template = request.app.state.templates.get_template.return_value
        template.render.assert_called_once_with()
