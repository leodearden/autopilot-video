"""Media list page, detail page, and API endpoints."""

from __future__ import annotations

import json
import math

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from autopilot.db import CatalogDB

router = APIRouter()


def _get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


def _format_duration(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS, or '--:--' if None."""
    if seconds is None:
        return "--:--"
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _query_params(
    q: str | None,
    status: str | None,
    sort: str,
    order: str,
    page: int,
    per_page: int,
    date_from: str | None,
    date_to: str | None,
) -> dict:
    """Bundle query parameters into a dict for reuse."""
    return {
        "q": q,
        "status": status,
        "sort": sort,
        "order": order,
        "page": page,
        "per_page": per_page,
        "date_from": date_from,
        "date_to": date_to,
    }


@router.get("/api/media")
def api_media(
    request: Request,
    q: str | None = Query(None),
    status: str | None = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    """Return paginated, filtered media list as JSON (or HTML partial for HTMX)."""
    db = _get_db(request)
    try:
        result = db.query_media(
            q=q,
            status=status,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            order=order,
            page=page,
            per_page=per_page,
        )
    finally:
        db.close()

    # HTMX partial: return rendered table rows
    if request.headers.get("hx-request"):
        templates = request.app.state.templates
        html = templates.get_template("partials/media_row.html")
        rows = []
        for item in result["items"]:
            rendered = html.render(item=item, format_duration=_format_duration)
            rows.append(rendered)
        return HTMLResponse("".join(rows))

    return result


@router.get("/media")
def media_page(
    request: Request,
    q: str | None = Query(None),
    status: str | None = Query(None),
    sort: str = Query("created_at"),
    order: str = Query("desc"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    """Render the media list HTML page."""
    db = _get_db(request)
    try:
        result = db.query_media(
            q=q,
            status=status,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            order=order,
            page=page,
            per_page=per_page,
        )
    finally:
        db.close()

    total_pages = max(1, math.ceil(result["total"] / per_page))
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "media/list.html",
        {
            "items": result["items"],
            "total": result["total"],
            "total_pages": total_pages,
            "page": page,
            "per_page": per_page,
            "q": q,
            "status": status,
            "date_from": date_from,
            "date_to": date_to,
            "sort": sort,
            "order": order,
            "format_duration": _format_duration,
        },
    )


@router.get("/media/{media_id}")
def media_detail_page(request: Request, media_id: str):
    """Render the media detail HTML page."""
    db = _get_db(request)
    try:
        detail = db.get_media_detail(media_id)
    finally:
        db.close()
    if detail is None:
        raise HTTPException(status_code=404, detail="Media not found")

    media = detail["media"]
    extra_metadata = {}
    if media.get("metadata_json"):
        try:
            extra_metadata = json.loads(media["metadata_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "media/detail.html",
        {
            "media": media,
            "detail": detail,
            "extra_metadata": extra_metadata,
            "format_duration": _format_duration,
        },
    )


@router.get("/api/media/{media_id}")
def api_media_detail(request: Request, media_id: str):
    """Return detail for a single media file as JSON."""
    db = _get_db(request)
    try:
        detail = db.get_media_detail(media_id)
    finally:
        db.close()
    if detail is None:
        raise HTTPException(status_code=404, detail="Media not found")
    return detail


@router.get("/api/media/{media_id}/transcript")
def api_media_transcript(request: Request, media_id: str):
    """Return parsed transcript for a media file."""
    db = _get_db(request)
    try:
        media = db.get_media(media_id)
        if media is None:
            raise HTTPException(status_code=404, detail="Media not found")
        transcript = db.get_transcript(media_id)
    finally:
        db.close()
    if transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    segments = json.loads(transcript["segments_json"]) if transcript["segments_json"] else []
    return {"language": transcript["language"], "segments": segments}


@router.get("/api/media/{media_id}/detections")
def api_media_detections(request: Request, media_id: str):
    """Return aggregated detection summary for a media file."""
    db = _get_db(request)
    try:
        media = db.get_media(media_id)
        if media is None:
            raise HTTPException(status_code=404, detail="Media not found")
        det_rows = db.get_detections_for_media(media_id)
    finally:
        db.close()
    total = 0
    classes: dict[str, int] = {}
    for row in det_rows:
        dets = json.loads(row["detections_json"]) if row["detections_json"] else []
        total += len(dets)
        for d in dets:
            cls = d.get("class", "unknown")
            classes[cls] = classes.get(cls, 0) + 1
    return {
        "total_detections": total,
        "classes": classes,
        "frame_count": len(det_rows),
    }
