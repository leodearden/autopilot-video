"""Review hub and narrative review routes."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from autopilot.db import CatalogDB

router = APIRouter()


def _get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


def _is_htmx(request: Request) -> bool:
    """Check if the request is from HTMX."""
    return request.headers.get("hx-request") == "true"


@router.get("/review")
def review_hub(request: Request) -> HTMLResponse:
    """Render the review hub page."""
    templates = request.app.state.templates
    context = {"page_title": "Review Hub"}
    return templates.TemplateResponse(request, "review/hub.html", context)


@router.get("/api/narratives")
def api_list_narratives(
    request: Request, status: str | None = None,
) -> list[dict[str, object]]:
    """Return all narratives as a JSON list, optionally filtered by status."""
    db = _get_db(request)
    try:
        return db.list_narratives(status=status)
    finally:
        db.close()
