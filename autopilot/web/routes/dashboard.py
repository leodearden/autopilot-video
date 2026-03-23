"""Pipeline dashboard page and API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from autopilot.db import CatalogDB

router = APIRouter()

_PIPELINE_STAGES = CatalogDB._PIPELINE_STAGES


def _get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request) -> HTMLResponse:
    """Render the pipeline dashboard page with stage cards."""
    db = _get_db(request)
    try:
        run = db.get_current_run()
        gates = {g["stage"]: g for g in db.get_all_gates()}

        stages = []
        for name in _PIPELINE_STAGES:
            gate = gates.get(name, {})
            stages.append({
                "name": name,
                "gate_mode": gate.get("mode", "auto"),
            })

        templates = request.app.state.templates
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "page_title": "Dashboard",
                "run": run,
                "stages": stages,
                "pipeline_stages": _PIPELINE_STAGES,
            },
        )
    finally:
        db.close()
