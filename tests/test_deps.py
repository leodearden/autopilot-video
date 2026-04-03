"""Tests for shared route-handler dependencies in autopilot.web.deps."""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import Request
from fastapi.responses import HTMLResponse

from autopilot.web.deps import render_partial


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
