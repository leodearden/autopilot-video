"""Gate configuration page and API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

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


class GateUpdate(BaseModel):
    """Request body for updating a gate."""

    mode: Optional[Literal["auto", "pause", "notify"]] = None
    timeout_hours: Optional[float] = None

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"mode": "pause", "timeout_hours": 24.0}]},
    )


@router.put("/api/gates/{stage}")
def api_update_gate(request: Request, stage: str, body: GateUpdate) -> dict:
    """Update a gate's mode and/or timeout."""
    if stage not in _STAGE_ORDER:
        raise HTTPException(status_code=404, detail=f"Unknown stage: {stage}")
    db = _get_db(request)
    try:
        updates: dict[str, object] = {}
        if body.mode is not None:
            updates["mode"] = body.mode
        # timeout_hours: check if it was explicitly provided (even as None)
        if "timeout_hours" in (body.model_fields_set or set()):
            updates["timeout_hours"] = body.timeout_hours
        if updates:
            db.update_gate(stage, **updates)
            db.conn.commit()
        gate = db.get_gate(stage)
    finally:
        db.close()
    if gate is None:
        raise HTTPException(status_code=404, detail=f"Gate not found: {stage}")
    return gate


def _gate_decision(request: Request, stage: str, status: str) -> dict:
    """Apply a gate decision (approve/skip) and return updated gate."""
    if stage not in _STAGE_ORDER:
        raise HTTPException(status_code=404, detail=f"Unknown stage: {stage}")
    db = _get_db(request)
    try:
        db.update_gate(
            stage,
            status=status,
            decided_by="console",
            decided_at=datetime.now(timezone.utc).isoformat(),
        )
        db.conn.commit()
        gate = db.get_gate(stage)
    finally:
        db.close()
    if gate is None:
        raise HTTPException(status_code=404, detail=f"Gate not found: {stage}")
    return gate


@router.post("/api/gates/{stage}/approve")
def api_approve_gate(request: Request, stage: str) -> dict:
    """Approve a waiting gate."""
    return _gate_decision(request, stage, "approved")


@router.post("/api/gates/{stage}/skip")
def api_skip_gate(request: Request, stage: str) -> dict:
    """Skip a waiting gate."""
    return _gate_decision(request, stage, "skipped")


@router.put("/api/gates/preset/{preset_name}")
def api_apply_preset(request: Request, preset_name: str) -> list[dict]:
    """Apply a gate preset, updating all gate modes."""
    if preset_name not in GATE_PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_name}")
    preset = GATE_PRESETS[preset_name]
    db = _get_db(request)
    try:
        for stage in _PIPELINE_STAGES:
            mode = preset.get(stage, "auto")
            db.update_gate(stage, mode=mode)
        db.conn.commit()
        gates = db.get_all_gates()
    finally:
        db.close()
    gates.sort(key=lambda g: _STAGE_ORDER.get(g["stage"], 999))
    return gates
