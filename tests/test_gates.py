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


class TestGateDetailAPI:
    """Tests for GET /api/gates/{stage} endpoint."""

    def test_api_gate_detail_returns_single_gate(self, client: TestClient) -> None:
        """GET /api/gates/analyze returns 200 with gate dict for 'analyze'."""
        response = client.get("/api/gates/analyze")
        assert response.status_code == 200
        data = response.json()
        assert data["stage"] == "analyze"
        assert data["mode"] == "auto"

    def test_api_gate_detail_unknown_stage_returns_404(self, client: TestClient) -> None:
        """GET /api/gates/nonexistent returns 404."""
        response = client.get("/api/gates/nonexistent")
        assert response.status_code == 404


class TestGateUpdateAPI:
    """Tests for PUT /api/gates/{stage} endpoint."""

    def test_update_gate_mode(self, client: TestClient) -> None:
        """PUT /api/gates/analyze with mode='pause' updates and persists."""
        response = client.put("/api/gates/analyze", json={"mode": "pause"})
        assert response.status_code == 200
        assert response.json()["mode"] == "pause"
        # Re-GET confirms persistence
        check = client.get("/api/gates/analyze")
        assert check.json()["mode"] == "pause"

    def test_update_gate_invalid_mode(self, client: TestClient) -> None:
        """PUT with invalid mode returns 422."""
        response = client.put("/api/gates/analyze", json={"mode": "invalid"})
        assert response.status_code == 422

    def test_update_gate_timeout(self, client: TestClient) -> None:
        """PUT /api/gates/render with timeout_hours=24.0 persists."""
        response = client.put("/api/gates/render", json={"timeout_hours": 24.0})
        assert response.status_code == 200
        assert response.json()["timeout_hours"] == 24.0

    def test_update_gate_clear_timeout(self, client: TestClient) -> None:
        """PUT with timeout_hours=null clears the timeout."""
        # Set a timeout first
        client.put("/api/gates/render", json={"timeout_hours": 12.0})
        # Clear it
        response = client.put("/api/gates/render", json={"timeout_hours": None})
        assert response.status_code == 200
        assert response.json()["timeout_hours"] is None

    def test_update_gate_unknown_stage(self, client: TestClient) -> None:
        """PUT /api/gates/nonexistent returns 404."""
        response = client.put("/api/gates/nonexistent", json={"mode": "auto"})
        assert response.status_code == 404


class TestGateApproveAPI:
    """Tests for POST /api/gates/{stage}/approve endpoint."""

    def test_approve_gate(self, client: TestClient) -> None:
        """POST /api/gates/classify/approve returns 200 with approved status."""
        response = client.post("/api/gates/classify/approve")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["decided_by"] == "console"
        assert data["decided_at"] is not None

    def test_approve_unknown_stage_returns_404(self, client: TestClient) -> None:
        """POST /api/gates/nonexistent/approve returns 404."""
        response = client.post("/api/gates/nonexistent/approve")
        assert response.status_code == 404


class TestGateSkipAPI:
    """Tests for POST /api/gates/{stage}/skip endpoint."""

    def test_skip_gate(self, client: TestClient) -> None:
        """POST /api/gates/classify/skip returns 200 with skipped status."""
        response = client.post("/api/gates/classify/skip")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "skipped"
        assert data["decided_by"] == "console"
        assert data["decided_at"] is not None

    def test_skip_unknown_stage_returns_404(self, client: TestClient) -> None:
        """POST /api/gates/nonexistent/skip returns 404."""
        response = client.post("/api/gates/nonexistent/skip")
        assert response.status_code == 404
