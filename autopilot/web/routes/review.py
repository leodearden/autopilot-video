"""Review hub, narrative review, and cluster review routes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.responses import FileResponse, Response

from autopilot.db import CatalogDB
from autopilot.web.deps import render_partial

router = APIRouter()


def _get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


def _is_htmx(request: Request) -> bool:
    """Check if the request is from HTMX."""
    return request.headers.get("hx-request") == "true"


_REVIEW_LINKS: dict[str, str] = {
    "narrate": "/review/narratives",
    "classify": "/review/clusters",
    "render": "/review/render",
    "upload": "/review/uploads",
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
                elif stage == "classify":
                    summary["pending_count"] = db.count_non_excluded_clusters()
                    summary["pending_label"] = "activity clusters"
                elif stage == "render":
                    plans = db.list_edit_plans()
                    pending_renders = [
                        p for p in plans if not p.get("render_path")
                    ]
                    summary["pending_count"] = len(pending_renders)
                    summary["pending_label"] = "pending renders"
                elif stage == "upload":
                    plans = db.list_edit_plans()
                    rendered_ids = {
                        p["narrative_id"]
                        for p in plans
                        if p.get("render_path")
                    }
                    uploaded_ids = {
                        u["narrative_id"] for u in db.list_uploads()
                    }
                    summary["pending_count"] = len(
                        rendered_ids - uploaded_ids,
                    )
                    summary["pending_label"] = "pending uploads"
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


def _safe_json_list(raw: object) -> list[object]:
    """Parse *raw* as a JSON list, returning ``[]`` on any failure.

    Handles ``None``, empty strings, malformed JSON, and valid JSON that
    is not a list (e.g. a string, integer, or object).
    """
    if not raw:
        return []
    try:
        result = json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(result, list):
        return []
    return result


def _parse_narrative(row: dict[str, object]) -> dict[str, object]:
    """Enrich a narrative row with parsed activity_cluster_ids."""
    result = dict(row)
    raw = result.pop("activity_cluster_ids_json", None)
    result["activity_cluster_ids"] = _safe_json_list(raw)
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
        return render_partial(request, "partials/narrative_card.html", narrative=parsed)
    return JSONResponse(content=parsed)


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
        return render_partial(request, "partials/narrative_card.html", narrative=parsed)
    return JSONResponse(content=parsed)


# ---------------------------------------------------------------------------
# Cluster review helpers + routes
# ---------------------------------------------------------------------------


def _parse_ts(s: str) -> datetime:
    """Parse an ISO timestamp, raising HTTP 422 with the offending value on failure."""
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid timestamp format: {s!r}",
        )


def _parse_cluster(row: dict[str, object]) -> dict[str, Any]:
    """Enrich a cluster row with parsed clip_ids and clip_count."""
    result = dict(row)
    raw = result.pop("clip_ids_json", None)
    if raw:
        try:
            clip_ids: list[str] = json.loads(str(raw))
        except json.JSONDecodeError:
            clip_ids = []
    else:
        clip_ids = []
    result["clip_ids"] = clip_ids
    result["clip_count"] = len(clip_ids)
    return result


@router.get("/review/clusters")
def clusters_page(request: Request) -> HTMLResponse:
    """Render the cluster review page with all clusters."""
    db = _get_db(request)
    try:
        rows = db.get_activity_clusters()
        clusters = [_parse_cluster(r) for r in rows]
        classify_gate = db.get_gate("classify")
    finally:
        db.close()
    gate_waiting = (
        classify_gate is not None
        and classify_gate.get("status") == "waiting"
    )
    non_excluded = [c for c in clusters if not c.get("excluded")]
    show_approve_gate = gate_waiting and len(non_excluded) > 0
    templates = request.app.state.templates
    context = {
        "page_title": "Cluster Review",
        "clusters": clusters,
        "show_approve_gate": show_approve_gate,
    }
    return templates.TemplateResponse(request, "review/clusters.html", context)


@router.get("/api/clusters")
def api_list_clusters(request: Request) -> list[dict[str, Any]]:
    """Return all activity clusters as a JSON list with parsed clip_ids."""
    db = _get_db(request)
    try:
        rows = db.get_activity_clusters()
        return [_parse_cluster(r) for r in rows]
    finally:
        db.close()


@router.get("/api/clusters/{cluster_id}")
def api_get_cluster(request: Request, cluster_id: str) -> dict[str, Any]:
    """Return a single cluster by ID with parsed clip_ids."""
    db = _get_db(request)
    try:
        row = db.get_activity_cluster(cluster_id)
    finally:
        db.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    return _parse_cluster(row)


def _update_and_respond_cluster(
    request: Request, db: CatalogDB, cluster_id: str,
) -> Response:
    """Fetch cluster, parse, and return HTMX partial or JSON response.

    Shared post-update logic for relabel and exclude routes.
    Raises 404 if the cluster doesn't exist.
    """
    updated = db.get_activity_cluster(cluster_id)
    if updated is None:
        raise HTTPException(status_code=404, detail="Cluster not found")
    parsed = _parse_cluster(updated)
    if _is_htmx(request):
        return render_partial(request, "partials/cluster_card.html", cluster=parsed)
    return JSONResponse(content=parsed)


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
        updates = body.model_dump(exclude_unset=True)
        if updates:
            affected = db.update_activity_cluster(cluster_id, **updates)
            if affected == 0:
                raise HTTPException(status_code=404, detail="Cluster not found")
            db.conn.commit()
        return _update_and_respond_cluster(request, db, cluster_id)
    finally:
        db.close()


@router.post("/api/clusters/{cluster_id}/exclude", response_model=None)
def api_exclude_cluster(request: Request, cluster_id: str) -> Response:
    """Mark a cluster as excluded."""
    db = _get_db(request)
    try:
        affected = db.update_activity_cluster(cluster_id, excluded=1)
        if affected == 0:
            raise HTTPException(status_code=404, detail="Cluster not found")
        db.conn.commit()
        return _update_and_respond_cluster(request, db, cluster_id)
    finally:
        db.close()


class MergeRequest(BaseModel):
    """Request body for merging clusters."""

    cluster_ids: list[str]

    model_config = ConfigDict(extra="forbid")


@router.post("/api/clusters/merge")
def api_merge_clusters(
    request: Request, body: MergeRequest,
) -> dict[str, Any]:
    """Merge multiple clusters into the largest one by clip count."""
    if len(body.cluster_ids) < 2:
        raise HTTPException(
            status_code=422, detail="At least 2 cluster_ids required",
        )
    db = _get_db(request)
    try:
        # Batch-load all clusters (eliminates N+1 per-cluster queries)
        clusters_by_id = db.get_activity_clusters_by_ids(body.cluster_ids)
        missing = set(body.cluster_ids) - set(clusters_by_id)
        if missing:
            first_missing = next(m for m in body.cluster_ids if m in missing)
            raise HTTPException(
                status_code=404,
                detail=f"Cluster {first_missing} not found",
            )
        clusters: list[dict[str, object]] = [
            clusters_by_id[cid] for cid in body.cluster_ids
        ]

        parsed_clusters = [_parse_cluster(c) for c in clusters]

        # Find largest by clip count
        largest = max(parsed_clusters, key=lambda c: c["clip_count"])
        largest_id = largest["cluster_id"]

        # Combine all clip_ids (deduplicated, preserving order)
        all_clip_ids: list[str] = list(dict.fromkeys(
            cid for pc in parsed_clusters for cid in pc["clip_ids"]
        ))

        # Compute time range (chronological comparison, not lexicographic)
        time_starts = [str(c["time_start"]) for c in clusters if c.get("time_start")]
        time_ends = [str(c["time_end"]) for c in clusters if c.get("time_end")]
        min_start = min(time_starts, key=_parse_ts) if time_starts else None
        max_end = max(time_ends, key=_parse_ts) if time_ends else None

        # Update largest cluster with combined data
        update_kwargs: dict[str, object] = {
            "clip_ids_json": json.dumps(all_clip_ids),
        }
        if min_start is not None:
            update_kwargs["time_start"] = min_start
        if max_end is not None:
            update_kwargs["time_end"] = max_end

        # Wrap mutations in context manager for explicit transaction safety:
        # __exit__ commits on success, rolls back on exception.
        with db:
            db.update_activity_cluster(largest_id, **update_kwargs)

            # Batch-delete non-surviving clusters
            non_surviving = [
                pc["cluster_id"] for pc in parsed_clusters
                if pc["cluster_id"] != largest_id
            ]
            db.batch_delete_activity_clusters(non_surviving)

            merged_map = db.get_activity_clusters_by_ids([largest_id])
        merged = merged_map.get(largest_id)
        if merged is None:
            raise HTTPException(status_code=500, detail="Merged cluster missing")
    finally:
        db.close()
    return _parse_cluster(merged)


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


# ---------------------------------------------------------------------------
# Render review helpers + routes
# ---------------------------------------------------------------------------


def _parse_render(edit_plan: dict[str, object]) -> dict[str, object]:
    """Parse validation_json from an edit plan into a structured dict."""
    result = dict(edit_plan)
    raw = result.pop("validation_json", None)
    result["validation"] = json.loads(str(raw)) if raw else None
    return result


@router.get("/api/renders/{narrative_id}")
def api_get_render(
    request: Request, narrative_id: str,
) -> dict[str, object]:
    """Return render data for a narrative with parsed validation."""
    db = _get_db(request)
    try:
        edit_plan = db.get_edit_plan(narrative_id)
        if edit_plan is None:
            raise HTTPException(status_code=404, detail="Edit plan not found")
        narrative = db.get_narrative(narrative_id)
    finally:
        db.close()
    parsed = _parse_render(edit_plan)
    parsed["narrative_title"] = (
        narrative["title"] if narrative else None
    )
    return parsed


@router.get("/api/renders/{narrative_id}/video")
def api_stream_video(
    request: Request, narrative_id: str,
) -> FileResponse:
    """Stream the rendered video file for a narrative.

    Starlette's FileResponse handles Range headers natively for seeking.
    """
    db = _get_db(request)
    try:
        edit_plan = db.get_edit_plan(narrative_id)
    finally:
        db.close()
    if edit_plan is None:
        raise HTTPException(status_code=404, detail="Edit plan not found")
    render_path = edit_plan.get("render_path")
    if not render_path or not Path(str(render_path)).is_file():
        raise HTTPException(status_code=404, detail="Render file not available")
    return FileResponse(str(render_path), media_type="video/mp4")


@router.get("/api/uploads")
def api_list_uploads(request: Request) -> list[dict[str, object]]:
    """Return all uploads as a JSON list with narrative titles."""
    db = _get_db(request)
    try:
        return db.list_uploads()
    finally:
        db.close()


@router.get("/review/render")
def render_index_page(request: Request) -> HTMLResponse:
    """List narratives with edit plans and links to individual review pages."""
    db = _get_db(request)
    try:
        edit_plans = db.list_edit_plans()
    finally:
        db.close()
    templates = request.app.state.templates
    context = {
        "page_title": "Render Review",
        "edit_plans": edit_plans,
    }
    return templates.TemplateResponse(
        request, "review/render_index.html", context,
    )


@router.get("/review/render/{narrative_id}")
def render_review_page(
    request: Request, narrative_id: str,
) -> HTMLResponse:
    """Render the render review page for a single narrative."""
    db = _get_db(request)
    try:
        narrative = db.get_narrative(narrative_id)
        if narrative is None:
            raise HTTPException(status_code=404, detail="Narrative not found")
        edit_plan = db.get_edit_plan(narrative_id)
        if edit_plan is None:
            raise HTTPException(
                status_code=404, detail="Edit plan not found",
            )
        script = db.get_narrative_script(narrative_id)
    finally:
        db.close()

    parsed = _parse_render(edit_plan)
    # Parse script scenes
    scenes: list[dict[str, object]] = []
    if script and script.get("script_json"):
        script_data = json.loads(str(script["script_json"]))
        if isinstance(script_data, dict):
            scenes = script_data.get("scenes", [])
        elif isinstance(script_data, list):
            scenes = script_data

    # Check if render file exists
    render_path = parsed.get("render_path")
    has_render = bool(
        render_path and Path(str(render_path)).is_file()
    )

    templates = request.app.state.templates
    context = {
        "page_title": f"Render Review: {narrative['title']}",
        "narrative": narrative,
        "edit_plan": parsed,
        "has_render": has_render,
        "scenes": scenes,
    }
    return templates.TemplateResponse(
        request, "review/renders.html", context,
    )


@router.get("/review/uploads")
def uploads_page(request: Request) -> HTMLResponse:
    """Render the uploads status page listing all YouTube uploads."""
    db = _get_db(request)
    try:
        uploads = db.list_uploads()
    finally:
        db.close()
    templates = request.app.state.templates
    context = {
        "page_title": "Upload Status",
        "uploads": uploads,
    }
    return templates.TemplateResponse(
        request, "review/uploads.html", context,
    )
