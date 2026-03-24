"""Review hub and narrative review routes."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
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


def _parse_narrative(row: dict[str, object]) -> dict[str, object]:
    """Enrich a narrative row with parsed activity_cluster_ids."""
    result = dict(row)
    raw = result.pop("activity_cluster_ids_json", None)
    result["activity_cluster_ids"] = json.loads(str(raw)) if raw else []
    return result


@router.get("/api/narratives/{narrative_id}")
def api_get_narrative(request: Request, narrative_id: str) -> dict[str, object]:
    """Return a single narrative by ID with parsed cluster IDs."""
    db = _get_db(request)
    try:
        row = db.get_narrative(narrative_id)
    finally:
        db.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Narrative not found")
    return _parse_narrative(row)
