"""Settings page."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/settings")
def settings_page(request: Request) -> HTMLResponse:
    """Render the settings page."""
    templates = request.app.state.templates
    context = {"page_title": "Settings"}
    return templates.TemplateResponse(request, "settings/config.html", context)
