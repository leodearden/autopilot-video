"""Review hub and narrative review routes."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.responses import Response

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


def _narrative_status_action(
    request: Request, narrative_id: str, status: str,
) -> Response:
    """Set a narrative's status and return updated narrative or HTML partial."""
    db = _get_db(request)
    try:
        row = db.get_narrative(narrative_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Narrative not found")
        db.update_narrative_status(narrative_id, status)
        db.conn.commit()
        updated = db.get_narrative(narrative_id)
    finally:
        db.close()
    if updated is None:
        raise HTTPException(status_code=404, detail="Narrative not found")
    parsed = _parse_narrative(updated)
    if _is_htmx(request):
        return _render_narrative_partial(request, parsed)
    return JSONResponse(content=parsed)


def _render_narrative_partial(
    request: Request, narrative: dict[str, object],
) -> HTMLResponse:
    """Render a single narrative card partial as HTML."""
    templates = request.app.state.templates
    template = templates.get_template("partials/narrative_card.html")
    html = template.render(narrative=narrative)
    return HTMLResponse(content=html)


@router.post("/api/narratives/{narrative_id}/approve", response_model=None)
def api_approve_narrative(request: Request, narrative_id: str) -> Response:
    """Approve a narrative."""
    return _narrative_status_action(request, narrative_id, "approved")


@router.post("/api/narratives/{narrative_id}/reject", response_model=None)
def api_reject_narrative(request: Request, narrative_id: str) -> Response:
    """Reject a narrative."""
    return _narrative_status_action(request, narrative_id, "rejected")


class NarrativeUpdate(BaseModel):
    """Request body for updating narrative fields."""

    title: Optional[str] = None
    description: Optional[str] = None
    proposed_duration_seconds: Optional[float] = None

    model_config = ConfigDict(extra="forbid")


@router.put("/api/narratives/{narrative_id}", response_model=None)
def api_update_narrative(
    request: Request, narrative_id: str, body: NarrativeUpdate,
) -> Response:
    """Update editable fields of a narrative."""
    db = _get_db(request)
    try:
        row = db.get_narrative(narrative_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Narrative not found")
        updates = body.model_dump(exclude_unset=True)
        if updates:
            db.update_narrative(narrative_id, **updates)
            db.conn.commit()
        updated = db.get_narrative(narrative_id)
    finally:
        db.close()
    if updated is None:
        raise HTTPException(status_code=404, detail="Narrative not found")
    parsed = _parse_narrative(updated)
    if _is_htmx(request):
        return _render_narrative_partial(request, parsed)
    return JSONResponse(content=parsed)
