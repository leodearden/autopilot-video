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
    db = CatalogDB(db_path)
    try:
        db.init_default_gates()
    finally:
        db.close()
    return create_app(db_path)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a TestClient for the app."""
    return TestClient(app)
