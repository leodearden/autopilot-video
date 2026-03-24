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


_REVIEW_LINKS: dict[str, str] = {
    "narrate": "/review/narratives",
}


@router.get("/review")
def review_hub(request: Request) -> HTMLResponse:
    """Render the review hub page with waiting gate summaries."""
    db = _get_db(request)
    try:
        gates = db.get_all_gates()
        waiting_gates = []
        for gate in gates:
            if gate.get("status") == "waiting":
                stage = str(gate["stage"])
                summary: dict[str, object] = {
                    "stage": stage,
                    "link": _REVIEW_LINKS.get(stage, f"/review/{stage}"),
                }
                # Add pending counts per stage
                if stage == "narrate":
                    proposed = db.list_narratives(status="proposed")
                    summary["pending_count"] = len(proposed)
                    summary["pending_label"] = "proposed narratives"
                waiting_gates.append(summary)
    finally:
        db.close()
    templates = request.app.state.templates
    context = {
        "page_title": "Review Hub",
        "waiting_gates": waiting_gates,
    }
    return templates.TemplateResponse(request, "review/hub.html", context)


@router.get("/review/narratives")
def narratives_page(request: Request) -> HTMLResponse:
    """Render the narrative review page with all narratives."""
    db = _get_db(request)
    try:
        narratives_raw = db.list_narratives()
        narratives = [_parse_narrative(dict(n)) for n in narratives_raw]
        proposed_count = sum(1 for n in narratives if n.get("status") == "proposed")
        narrate_gate = db.get_gate("narrate")
    finally:
        db.close()
    gate_waiting = (
        narrate_gate is not None
        and narrate_gate.get("status") == "waiting"
    )
    show_approve_gate = gate_waiting and proposed_count == 0
    templates = request.app.state.templates
    context = {
        "page_title": "Narrative Review",
        "narratives": narratives,
        "show_approve_gate": show_approve_gate,
        "proposed_count": proposed_count,
    }
    return templates.TemplateResponse(request, "review/narratives.html", context)


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


class BulkApproveRequest(BaseModel):
    """Request body for bulk-approving narratives."""

    ids: list[str]


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


# ---------------------------------------------------------------------------
# Cluster review helpers + routes
# ---------------------------------------------------------------------------


def _parse_cluster(row: dict[str, object]) -> dict[str, object]:
    """Enrich a cluster row with parsed clip_ids and clip_count."""
    result = dict(row)
    raw = result.pop("clip_ids_json", None)
    clip_ids: list[str] = json.loads(str(raw)) if raw else []
    result["clip_ids"] = clip_ids
    result["clip_count"] = len(clip_ids)
    return result


@router.get("/api/clusters")
def api_list_clusters(request: Request) -> list[dict[str, object]]:
    """Return all activity clusters as a JSON list with parsed clip_ids."""
    db = _get_db(request)
    try:
        rows = db.get_activity_clusters()
        return [_parse_cluster(dict(r)) for r in rows]
    finally:
        db.close()


@router.get("/api/clusters/{cluster_id}")
def api_get_cluster(request: Request, cluster_id: str) -> dict[str, object]:
    """Return a single cluster by ID with parsed clip_ids."""
    db = _get_db(request)
    try:
        row = db.get_activity_cluster(cluster_id)
    finally:
        db.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return _parse_cluster(dict(row))


class ClusterRelabel(BaseModel):
    """Request body for relabelling a cluster."""

    label: Optional[str] = None
    description: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


@router.post("/api/clusters/{cluster_id}/relabel", response_model=None)
def api_relabel_cluster(
    request: Request, cluster_id: str, body: ClusterRelabel,
) -> Response:
    """Update label/description of a cluster."""
    db = _get_db(request)
    try:
        row = db.get_activity_cluster(cluster_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Cluster not found")
        updates = body.model_dump(exclude_unset=True)
        if updates:
            db.update_activity_cluster(cluster_id, **updates)
            db.conn.commit()
        updated = db.get_activity_cluster(cluster_id)
    finally:
        db.close()
    if updated is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    parsed = _parse_cluster(dict(updated))
    return JSONResponse(content=parsed)


@router.post("/api/narratives/bulk-approve")
def api_bulk_approve(request: Request, body: BulkApproveRequest) -> dict[str, int]:
    """Approve multiple narratives at once, returning actual count of rows updated."""
    db = _get_db(request)
    try:
        affected = sum(db.update_narrative_status(nid, "approved") for nid in body.ids)
        db.conn.commit()
    finally:
        db.close()
    return {"approved": affected}
