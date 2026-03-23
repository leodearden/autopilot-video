"""Tests for gate configuration page and API endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app

# Constants mirrored from the route module (populated after pre-2)
VALID_MODES = ("auto", "pause", "notify")

GATE_PRESETS = {
    "full_auto": {s: "auto" for s in CatalogDB._PIPELINE_STAGES},
    "review_creative": {
        "narrate": "pause",
        "script": "pause",
        "upload": "pause",
    },
    "review_everything": {s: "pause" for s in CatalogDB._PIPELINE_STAGES},
    "review_before_render": {
        "source": "pause",
        "upload": "pause",
    },
}


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with a file-backed CatalogDB via tmp_path."""
    db_path = str(tmp_path / "catalog.db")
    # Seed the gate rows
    with CatalogDB(db_path) as db:
        db.init_default_gates()
    return create_app(db_path)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a TestClient for the app."""
    return TestClient(app)


class TestGateListAPI:
    """Tests for GET /api/gates endpoint."""

    def test_api_gates_returns_all_gates(self, client: TestClient) -> None:
        """GET /api/gates returns 200 with JSON list of 9 gate dicts."""
        response = client.get("/api/gates")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 9

        # Each gate has required keys
        required_keys = {"stage", "mode", "status", "timeout_hours", "decided_at", "decided_by"}
        for gate in data:
            assert required_keys.issubset(gate.keys()), f"Missing keys in {gate}"

        # Stages match _PIPELINE_STAGES order
        stages = [g["stage"] for g in data]
        assert stages == list(CatalogDB._PIPELINE_STAGES)

        # Default mode is 'auto'
        for gate in data:
            assert gate["mode"] == "auto"
