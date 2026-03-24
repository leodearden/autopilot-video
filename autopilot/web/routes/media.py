"""Media list page, detail page, and API endpoints."""

from __future__ import annotations

import json
import math
from functools import partial
from typing import NamedTuple, cast

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse

from autopilot.web.deps import get_db, is_htmx

router = APIRouter()


def _parse_metadata_json(media: dict) -> dict:
    """Parse metadata_json from a media row, returning {} on any error."""
    raw = media.get("metadata_json")
    if raw:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


class DetectionSummary(NamedTuple):
    """Aggregated detection counts from a list of detection rows."""

    total: int
    classes: dict[str, int]
    frame_details: list[dict]


def _aggregate_detections(det_rows: list[dict]) -> DetectionSummary:
    """Aggregate detection rows into total count, per-class counts, and per-frame details."""
    total = 0
    classes: dict[str, int] = {}
    frame_details: list[dict] = []
    for row in det_rows:
        dets: list = []
        raw = row.get("detections_json")
        if raw:
            try:
                if isinstance(raw, str):
                    dets = json.loads(raw)
                elif isinstance(raw, list):
                    dets = raw
                else:
                    dets = json.loads(str(raw))
            except (json.JSONDecodeError, TypeError):
                pass
        if not isinstance(dets, list):
            dets = []
        total += len(dets)
        frame_details.append({"number": row["frame_number"], "count": len(dets)})
        for d in dets:
            cls = d.get("class", "unknown")
            classes[cls] = classes.get(cls, 0) + 1
    return DetectionSummary(total=total, classes=classes, frame_details=frame_details)


def _format_seconds(seconds: float | None, *, pad_hours: bool = False) -> str:
    """Format seconds into a human-readable time string.

    pad_hours=False: M:SS for <1hr, H:MM:SS for >=1hr, None -> '--:--'
    pad_hours=True:  00:MM:SS always, None -> '--:--:--'
    """
    if seconds is None:
        return "--:--:--" if pad_hours else "--:--"
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if pad_hours:
        return f"{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


_format_duration = partial(_format_seconds, pad_hours=False)
_format_timestamp = partial(_format_seconds, pad_hours=True)


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
    extra_metadata = _parse_metadata_json(media)

    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "media/detail.html",
        {
            "media": media,
            "extra_metadata": extra_metadata,
            "format_duration": _format_duration,
        },
    )


_VALID_TABS = {"metadata", "transcript", "detections", "faces", "audio_events", "embeddings"}


@router.get("/media/{media_id}/tab/{tab_name}")
def media_tab(request: Request, media_id: str, tab_name: str):
    """Render a single tab partial for HTMX swap."""
    if tab_name not in _VALID_TABS:
        raise HTTPException(status_code=404, detail="Invalid tab")
    db = get_db(request)
    try:
        media = db.get_media(media_id)
        if media is None:
            raise HTTPException(status_code=404, detail="Media not found")

        templates = request.app.state.templates

        if tab_name == "metadata":
            extra_metadata = _parse_metadata_json(media)
            html = templates.get_template("partials/tab_metadata.html").render(
                media=media,
                extra_metadata=extra_metadata,
                format_duration=_format_duration,
            )
        elif tab_name == "transcript":
            transcript = db.get_transcript(media_id)
            segments = []
            if transcript and transcript.get("segments_json"):
                try:
                    segments = json.loads(cast(str, transcript["segments_json"]))
                except (json.JSONDecodeError, TypeError):
                    pass
            html = templates.get_template("partials/tab_transcript.html").render(
                transcript=transcript,
                segments=segments,
                format_timestamp=_format_timestamp,
            )
        elif tab_name == "detections":
            det_rows = db.get_detections_for_media(media_id)
            summary = _aggregate_detections(det_rows)
            class_counts = sorted(summary.classes.items(), key=lambda x: x[1], reverse=True)
            html = templates.get_template("partials/tab_detections.html").render(
                detections=det_rows,
                total_detections=summary.total,
                frame_count=len(det_rows),
                class_counts=class_counts,
                frame_details=summary.frame_details,
            )
        elif tab_name == "faces":
            raw_faces = db.get_faces_for_media(media_id)
            # Strip embedding BLOBs (presentation concern, matching get_media_detail)
            faces = [{k: v for k, v in f.items() if k != "embedding"} for f in raw_faces]
            # Build face_clusters with str keys and stripped BLOBs
            cluster_ids = [
                cast(int, f["cluster_id"]) for f in faces if f.get("cluster_id") is not None
            ]
            raw_clusters = db.get_face_clusters_by_ids(cluster_ids)
            face_clusters: dict[str, dict] = {
                str(cid): {k: v for k, v in c.items() if k != "representative_embedding"}
                for cid, c in raw_clusters.items()
            }
            # Group faces by cluster_id
            cluster_groups: dict[int | None, int] = {}
            for f in faces:
                raw_cid = f.get("cluster_id")
                cid: int | None = cast(int, raw_cid) if raw_cid is not None else None
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
            audio_rows = db.get_audio_events_for_media(media_id)
            events = []
            for row in audio_rows:
                parsed = []
                if row.get("events_json"):
                    try:
                        parsed = json.loads(cast(str, row["events_json"]))
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
            embedding_count = db.count_embeddings_for_media(media_id)
            fps: float = cast(float, media.get("fps")) or 0
            duration: float = cast(float, media.get("duration_seconds")) or 0
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
    finally:
        db.close()

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
        return {"language": None, "segments": []}
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
    summary = _aggregate_detections(det_rows)
    return {
        "total_detections": summary.total,
        "classes": summary.classes,
        "frame_count": len(det_rows),
    }
