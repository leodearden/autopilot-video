"""Pipeline detail pages: overview, stage detail, and job list."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from autopilot.db import CatalogDB
from autopilot.web.deps import get_db

router = APIRouter()

_PIPELINE_STAGES = CatalogDB._PIPELINE_STAGES

_STATUS_COLORS = {
    "idle": "gray",
    "pending": "gray",
    "running": "blue",
    "waiting": "amber",
    "done": "green",
    "error": "red",
}


def _compute_stage_status(counts: dict[str, int]) -> str:
    """Derive aggregate status from job status counts."""
    if counts.get("error", 0) > 0:
        return "error"
    if counts.get("running", 0) > 0:
        return "running"
    total = sum(counts.values())
    if total == 0:
        return "idle"
    if counts.get("done", 0) == total:
        return "done"
    return "waiting"


def _format_duration(seconds: float | int | None) -> str:
    """Format seconds as H:MM:SS or M:SS, or '--:--' if None."""
    if seconds is None:
        return "--:--"
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _build_stages(
    db: CatalogDB, run_id: str | None,
) -> list[dict[str, Any]]:
    """Build enriched stage dicts with job counts and status."""
    gates = {str(g["stage"]): dict(g) for g in db.get_all_gates()}
    stages: list[dict[str, Any]] = []
    for name in _PIPELINE_STAGES:
        counts = db.count_jobs_by_status(name, run_id=run_id) if run_id else {}
        total = sum(counts.values())
        done = counts.get("done", 0)
        status = _compute_stage_status(counts)
        gate = gates.get(name, {})
        stages.append({
            "name": name,
            "status": status,
            "status_color": _STATUS_COLORS.get(status, "gray"),
            "done": done,
            "total": total,
            "counts": counts,
            "gate_mode": gate.get("mode", "auto"),
            "gate_status": gate.get("status", ""),
        })
    return stages


@router.get("/pipeline", response_class=HTMLResponse)
def pipeline_overview(request: Request) -> HTMLResponse:
    """Render the pipeline overview page with run info and stage summaries."""
    db = get_db(request)
    try:
        run = db.get_current_run()
        run_id = str(run["run_id"]) if run else None
        stages = _build_stages(db, run_id)
        runs = db.list_runs()

        templates = request.app.state.templates
        context = {
            "page_title": "Pipeline",
            "run": run,
            "stages": stages,
            "runs": runs,
        }
        return templates.TemplateResponse(
            request, "pipeline/index.html", context,
        )
    finally:
        db.close()


@router.get("/pipeline/stages", response_class=HTMLResponse)
def pipeline_stages(request: Request) -> HTMLResponse:
    """Render the stage detail page with per-stage job breakdowns."""
    db = get_db(request)
    try:
        run = db.get_current_run()
        run_id = str(run["run_id"]) if run else None
        stages = _build_stages(db, run_id)

        templates = request.app.state.templates
        context = {
            "page_title": "Pipeline Stages",
            "run": run,
            "stages": stages,
        }
        return templates.TemplateResponse(
            request, "pipeline/stages.html", context,
        )
    finally:
        db.close()


@router.get("/pipeline/jobs", response_class=HTMLResponse)
def pipeline_jobs(
    request: Request,
    stage: str | None = None,
    status: str | None = None,
) -> HTMLResponse:
    """Render the job list page with optional stage/status filtering."""
    db = get_db(request)
    try:
        run = db.get_current_run()
        run_id = str(run["run_id"]) if run else None
        kwargs: dict[str, str | None] = {"run_id": run_id}
        if stage:
            kwargs["stage"] = stage
        if status:
            kwargs["status"] = status
        jobs = db.list_jobs(**kwargs)

        templates = request.app.state.templates
        context = {
            "page_title": "Pipeline Jobs",
            "jobs": jobs,
            "filter_stage": stage,
            "filter_status": status,
            "pipeline_stages": _PIPELINE_STAGES,
        }
        return templates.TemplateResponse(
            request, "pipeline/jobs.html", context,
        )
    finally:
        db.close()


@router.get("/api/pipeline/jobs")
def api_pipeline_jobs(
    request: Request,
    stage: str | None = None,
    status: str | None = None,
) -> list[dict[str, object]]:
    """Return pipeline jobs as JSON with optional filtering."""
    db = get_db(request)
    try:
        run = db.get_current_run()
        run_id = str(run["run_id"]) if run else None
        kwargs: dict[str, str | None] = {"run_id": run_id}
        if stage:
            kwargs["stage"] = stage
        if status:
            kwargs["status"] = status
        return db.list_jobs(**kwargs)
    finally:
        db.close()
