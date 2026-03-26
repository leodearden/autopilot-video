"""Shared route-handler dependencies for the web layer."""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import HTMLResponse

from autopilot.db import CatalogDB


def get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


def is_htmx(request: Request) -> bool:
    """Check if the request is from HTMX."""
    return request.headers.get("hx-request") == "true"


def render_partial(
    request: Request, template_name: str, **ctx: object,
) -> HTMLResponse:
    """Render a Jinja2 template partial and return it as an HTMLResponse.

    Usage::

        return render_partial(request, "partials/narrative_card.html", narrative=parsed)
    """
    templates = request.app.state.templates
    template = templates.get_template(template_name)
    html: str = template.render(**ctx)
    return HTMLResponse(content=html)
