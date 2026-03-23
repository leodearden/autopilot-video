"""Pipeline dashboard page and API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from autopilot.db import CatalogDB


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

router = APIRouter()

_PIPELINE_STAGES = CatalogDB._PIPELINE_STAGES

# Status-to-Tailwind color class mapping
_STATUS_COLORS = {
    "idle": "gray",
    "running": "blue",
    "waiting": "amber",
    "done": "green",
    "error": "red",
}


def _get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


def _compute_stage_status(counts: dict[str, int]) -> str:
    """Derive aggregate status from job status counts.

    Priority: error > running > done > waiting > idle.
    """
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


def _build_single_stage(
    db: CatalogDB,
    stage_name: str,
    run_id: str | None,
    gate: dict,
) -> dict:
    """Build a stage summary dict for a single pipeline stage."""
    if run_id:
        counts = db.count_jobs_by_status(stage_name, run_id=run_id)
    else:
        counts = {}

    total = sum(counts.values())
    done = counts.get("done", 0)
    status = _compute_stage_status(counts)

    return {
        "name": stage_name,
        "status": status,
        "status_color": _STATUS_COLORS.get(status, "gray"),
        "done": done,
        "total": total,
        "counts": counts,
        "gate_mode": gate.get("mode", "auto"),
    }


def _build_stage_data(
    db: CatalogDB,
    run_id: str | None,
    gates: dict[str, dict],
) -> list[dict]:
    """Build stage summary dicts for all pipeline stages."""
    return [
        _build_single_stage(db, name, run_id, gates.get(name, {}))
        for name in _PIPELINE_STAGES
    ]


@router.get("/")
def root_redirect() -> RedirectResponse:
    """Redirect root URL to the dashboard page."""
    return RedirectResponse(url="/dashboard", status_code=307)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request) -> HTMLResponse:
    """Render the pipeline dashboard page with stage cards."""
    db = _get_db(request)
    try:
        run = db.get_current_run()
        gates = {g["stage"]: g for g in db.get_all_gates()}
        run_id = run["run_id"] if run else None
        stages = _build_stage_data(db, run_id, gates)

        # Compute timeline data
        total_done = sum(s["done"] for s in stages)
        total_jobs = sum(s["total"] for s in stages)
        progress_pct = (
            int(total_done / total_jobs * 100) if total_jobs > 0 else 0
        )

        elapsed = _format_duration(
            run.get("wall_clock_seconds") if run else None
        )
        budget_remaining = _format_duration(
            run.get("budget_remaining_seconds") if run else None
        )

        templates = request.app.state.templates
        context = {
            "page_title": "Dashboard",
            "run": run,
            "stages": stages,
            "pipeline_stages": _PIPELINE_STAGES,
            "total_done": total_done,
            "total_jobs": total_jobs,
            "progress_pct": progress_pct,
            "elapsed": elapsed,
            "budget_remaining": budget_remaining,
        }
        return templates.TemplateResponse(request, "dashboard.html", context)
    finally:
        db.close()


@router.get("/api/run")
def api_run(request: Request) -> dict:
    """Return the current pipeline run as JSON."""
    db = _get_db(request)
    try:
        run = db.get_current_run()
        return {"run": run}
    finally:
        db.close()


@router.get("/api/stages")
def api_stages(request: Request) -> list[dict]:
    """Return stage summaries as a JSON list."""
    db = _get_db(request)
    try:
        run = db.get_current_run()
        gates = {g["stage"]: g for g in db.get_all_gates()}
        run_id = run["run_id"] if run else None
        stages = _build_stage_data(db, run_id, gates)
        return [
            {
                "name": s["name"],
                "status_counts": s["counts"],
                "gate_mode": s["gate_mode"],
            }
            for s in stages
        ]
    finally:
        db.close()


@router.get("/dashboard/stage/{stage_name}", response_class=HTMLResponse)
def stage_card_partial(request: Request, stage_name: str) -> HTMLResponse:
    """Render a single stage card partial for HTMX swap updates."""
    if stage_name not in _PIPELINE_STAGES:
        raise HTTPException(status_code=404, detail=f"Unknown stage: {stage_name}")

    db = _get_db(request)
    try:
        run = db.get_current_run()
        run_id = run["run_id"] if run else None
        gate = db.get_gate(stage_name) or {}
        stage = _build_single_stage(db, stage_name, run_id, gate)

        templates = request.app.state.templates
        template = templates.get_template("partials/stage_card.html")
        html = template.render(stage=stage)
        return HTMLResponse(content=html)
    finally:
        db.close()
