"""Media list page, detail page, and API endpoints."""

from __future__ import annotations

import json
import math

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from autopilot.web.deps import get_db, is_htmx

router = APIRouter()


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
    db = get_db(request)
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
    if is_htmx(request):
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
    db = get_db(request)
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
    db = get_db(request)
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


def _format_timestamp(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS for transcript display."""
    if seconds is None:
        return "--:--:--"
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


_VALID_TABS = {"metadata", "transcript", "detections", "faces", "audio_events", "embeddings"}


@router.get("/media/{media_id}/tab/{tab_name}")
def media_tab(request: Request, media_id: str, tab_name: str):
    """Render a single tab partial for HTMX swap."""
    if tab_name not in _VALID_TABS:
        raise HTTPException(status_code=404, detail="Invalid tab")
    db = get_db(request)
    try:
        detail = db.get_media_detail(media_id)
    finally:
        db.close()
    if detail is None:
        raise HTTPException(status_code=404, detail="Media not found")

    media = detail["media"]
    templates = request.app.state.templates

    if tab_name == "metadata":
        extra_metadata = {}
        if media.get("metadata_json"):
            try:
                extra_metadata = json.loads(media["metadata_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        html = templates.get_template("partials/tab_metadata.html").render(
            media=media,
            extra_metadata=extra_metadata,
            format_duration=_format_duration,
        )
    elif tab_name == "transcript":
        transcript = detail["transcript"]
        segments = []
        if transcript and transcript.get("segments_json"):
            try:
                segments = json.loads(transcript["segments_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        html = templates.get_template("partials/tab_transcript.html").render(
            transcript=transcript,
            segments=segments,
            format_timestamp=_format_timestamp,
        )
    elif tab_name == "detections":
        det_rows = detail["detections"]
        total_detections = 0
        classes: dict[str, int] = {}
        frame_details = []
        for row in det_rows:
            dets: list = []
            if row.get("detections_json"):
                try:
                    dets = json.loads(row["detections_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            total_detections += len(dets)
            frame_details.append({"number": row["frame_number"], "count": len(dets)})
            for d in dets:
                cls = d.get("class", "unknown")
                classes[cls] = classes.get(cls, 0) + 1
        class_counts = sorted(classes.items(), key=lambda x: x[1], reverse=True)
        html = templates.get_template("partials/tab_detections.html").render(
            detections=det_rows,
            total_detections=total_detections,
            frame_count=len(det_rows),
            class_counts=class_counts,
            frame_details=frame_details,
        )
    elif tab_name == "faces":
        faces = detail["faces"]
        face_clusters = detail.get("face_clusters", {})
        # Group faces by cluster_id
        cluster_groups: dict[int | None, int] = {}
        for f in faces:
            cid = f.get("cluster_id")
            cluster_groups[cid] = cluster_groups.get(cid, 0) + 1
        clusters = []
        for cid, count in sorted(cluster_groups.items(), key=lambda x: x[1], reverse=True):
            cluster_info = face_clusters.get(str(cid), {}) if cid is not None else {}
            clusters.append(
                {
                    "cluster_id": cid,
                    "label": cluster_info.get("label"),
                    "count": count,
                }
            )
        html = templates.get_template("partials/tab_faces.html").render(
            clusters=clusters,
            total_faces=len(faces),
        )
    elif tab_name == "audio_events":
        audio_rows = detail["audio_events"]
        events = []
        for row in audio_rows:
            parsed = []
            if row.get("events_json"):
                try:
                    parsed = json.loads(row["events_json"])
                except (json.JSONDecodeError, TypeError):
                    pass
            event_classes = [
                {"name": e.get("class", "unknown"), "confidence": e.get("confidence", 0)}
                for e in parsed
            ]
            events.append({"timestamp": row["timestamp_seconds"], "classes": event_classes})
        html = templates.get_template("partials/tab_audio_events.html").render(
            events=events,
            format_timestamp=_format_timestamp,
        )
    elif tab_name == "embeddings":
        embedding_count = detail["embedding_count"]
        fps = media.get("fps") or 0
        duration = media.get("duration_seconds") or 0
        total_frames = int(fps * duration) if fps and duration else 0
        if total_frames > 0:
            coverage_pct = round(embedding_count / total_frames * 100, 1)
        else:
            coverage_pct = 0
        html = templates.get_template("partials/tab_embeddings.html").render(
            embedding_count=embedding_count,
            total_frames=total_frames,
            coverage_pct=coverage_pct,
        )

    return HTMLResponse(html)


@router.get("/api/media/{media_id}")
def api_media_detail(request: Request, media_id: str):
    """Return detail for a single media file as JSON."""
    db = get_db(request)
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
    db = get_db(request)
    try:
        media = db.get_media(media_id)
        if media is None:
            raise HTTPException(status_code=404, detail="Media not found")
        transcript = db.get_transcript(media_id)
    finally:
        db.close()
    if transcript is None:
        raise HTTPException(status_code=404, detail="Transcript not found")
    segments: list = []
    if transcript["segments_json"]:
        try:
            segments = json.loads(str(transcript["segments_json"]))
        except (json.JSONDecodeError, TypeError):
            pass
    return {"language": transcript["language"], "segments": segments}


@router.get("/api/media/{media_id}/detections")
def api_media_detections(request: Request, media_id: str):
    """Return aggregated detection summary for a media file."""
    db = get_db(request)
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
        dets: list = []
        if row["detections_json"]:
            try:
                dets = json.loads(str(row["detections_json"]))
            except (json.JSONDecodeError, TypeError):
                pass
        total += len(dets)
        for d in dets:
            cls = d.get("class", "unknown")
            classes[cls] = classes.get(cls, 0) + 1
    return {
        "total_detections": total,
        "classes": classes,
        "frame_count": len(det_rows),
    }
