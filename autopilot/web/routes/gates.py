"""Gate configuration page and API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from autopilot.db import CatalogDB

router = APIRouter()

_PIPELINE_STAGES = CatalogDB._PIPELINE_STAGES

VALID_MODES = ("auto", "pause", "notify")

# Lookup for sorting gates by pipeline order
_STAGE_ORDER = {s: i for i, s in enumerate(_PIPELINE_STAGES)}

_GATE_STATUS_COLORS = {
    "idle": "gray",
    "waiting": "amber",
    "approved": "green",
    "rejected": "red",
    "skipped": "blue",
}

# Stage transitions: list of (from_stage, to_stage, gate_stage) tuples.
# The first entry represents pipeline entry (gate controls starting ingest).
_STAGE_TRANSITIONS: list[tuple[str, str, str]] = []
for i, stage in enumerate(_PIPELINE_STAGES):
    if i == 0:
        _STAGE_TRANSITIONS.append(("start", stage, stage))
    else:
        _STAGE_TRANSITIONS.append((_PIPELINE_STAGES[i - 1], stage, stage))

# Presets: each maps stage → mode.  Stages not mentioned default to 'auto'.
GATE_PRESETS = {
    "full_auto": {s: "auto" for s in _PIPELINE_STAGES},
    "review_creative": {
        "narrate": "pause",
        "script": "pause",
        "upload": "pause",
    },
    "review_everything": {s: "pause" for s in _PIPELINE_STAGES},
    "review_before_render": {
        "source": "pause",
        "upload": "pause",
    },
}


def _get_db(request: Request) -> CatalogDB:
    """Create a CatalogDB connection from the app's db_path."""
    return CatalogDB(request.app.state.db_path)


@router.get("/api/gates")
def api_gates(request: Request) -> list[dict]:
    """Return all pipeline gates as a JSON list."""
    db = _get_db(request)
    try:
        gates = db.get_all_gates()
    finally:
        db.close()
    gates.sort(key=lambda g: _STAGE_ORDER.get(g["stage"], 999))
    return gates


@router.get("/api/gates/{stage}")
def api_gate_detail(request: Request, stage: str) -> dict:
    """Return a single gate by stage name."""
    if stage not in _STAGE_ORDER:
        raise HTTPException(status_code=404, detail=f"Unknown stage: {stage}")
    db = _get_db(request)
    try:
        gate = db.get_gate(stage)
    finally:
        db.close()
    if gate is None:
        raise HTTPException(status_code=404, detail=f"Gate not found: {stage}")
    return gate
