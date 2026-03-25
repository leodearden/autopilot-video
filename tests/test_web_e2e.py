"""End-to-end integration tests for the full web console.

These tests verify cross-cutting workflows and data consistency across
multiple web console views. Individual endpoint tests live in their
respective test files (test_dashboard.py, test_gates.py, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app

PIPELINE_STAGES = (
    "ingest", "analyze", "classify", "narrate", "script",
    "edl", "source", "render", "upload",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_db_path(tmp_path: Path) -> str:
    """Return the path for a test catalog DB file."""
    return str(tmp_path / "catalog.db")


@pytest.fixture
def e2e_db(e2e_db_path: str) -> CatalogDB:
    """Create a CatalogDB backed by a real file for E2E tests."""
    db = CatalogDB(e2e_db_path)
    db.conn.isolation_level = None  # autocommit
    return db


@pytest.fixture
def e2e_app(e2e_db_path: str) -> FastAPI:
    """Create a FastAPI app pointing at the E2E DB."""
    return create_app(e2e_db_path)


@pytest.fixture
def e2e_client(e2e_app: FastAPI) -> TestClient:
    """Create a TestClient for E2E tests."""
    return TestClient(e2e_app)


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def _parse_sse_body(text: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts.

    Each event has 'id', 'event', and 'data' keys extracted from
    the SSE wire format.
    """
    events: list[dict] = []
    current: dict = {}
    for line in text.splitlines():
        if line.startswith("id:"):
            current["id"] = line[3:].strip()
        elif line.startswith("event:"):
            current["event"] = line[6:].strip()
        elif line.startswith("data:"):
            current["data"] = line[5:].strip()
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


# ---------------------------------------------------------------------------
# DB seeding helpers
# ---------------------------------------------------------------------------


def _seed_media(
    db: CatalogDB, media_id: str = "m-1", **overrides: object,
) -> None:
    """Insert a media file with sensible defaults, overridable via kwargs."""
    defaults: dict[str, object] = {
        "file_path": f"/video/{media_id}.mp4",
        "codec": "h264",
        "resolution_w": 1920,
        "resolution_h": 1080,
        "fps": 30.0,
        "duration_seconds": 120.0,
        "status": "analyzed",
    }
    defaults.update(overrides)
    file_path = defaults.pop("file_path")
    db.insert_media(media_id, str(file_path), **defaults)  # type: ignore[arg-type]


def _seed_narrative(
    db: CatalogDB, narrative_id: str = "n-1", **overrides: object,
) -> None:
    """Insert a narrative with sensible defaults, overridable via kwargs."""
    defaults: dict[str, object] = {
        "title": "Morning Walk",
        "description": "A walk in the park",
        "proposed_duration_seconds": 120.0,
        "activity_cluster_ids_json": '["c-1","c-2"]',
        "arc_notes": "peaceful start",
        "emotional_journey": "calm → happy",
        "status": "proposed",
    }
    defaults.update(overrides)
    db.insert_narrative(narrative_id, **defaults)  # type: ignore[arg-type]


def _seed_cluster(
    db: CatalogDB, cluster_id: str = "c-1", **overrides: object,
) -> None:
    """Insert an activity cluster with sensible defaults, overridable via kwargs."""
    defaults: dict[str, object] = {
        "label": "Morning Activity",
        "description": "Walking the dog",
        "time_start": "2025-01-01T08:00:00",
        "time_end": "2025-01-01T09:00:00",
        "location_label": "Park",
        "gps_center_lat": 37.7749,
        "gps_center_lon": -122.4194,
        "clip_ids_json": '["clip-1","clip-2"]',
    }
    defaults.update(overrides)
    db.insert_activity_cluster(cluster_id, **defaults)  # type: ignore[arg-type]


def _seed_pipeline(db: CatalogDB) -> None:
    """Seed a running pipeline: run, jobs across stages, gates, and events.

    Creates:
      - Run 'run-1' (running, current_stage=analyze)
      - Ingest: 5 done + 2 running jobs
      - Analyze: 1 running + 2 pending jobs
      - Classify: 1 error job
      - Default gates with analyze set to mode='manual'
      - Error event for classify stage
    """
    db.insert_run(
        "run-1",
        started_at="2025-06-15T10:00:00",
        config_snapshot='{"quality": "high"}',
        current_stage="analyze",
        status="running",
        budget_remaining_seconds=3600,
    )
    db.update_run("run-1", wall_clock_seconds=120.0)

    # ingest: 5 done + 2 running
    for i in range(5):
        db.insert_job(
            f"ingest-done-{i}", "ingest", "transcode",
            target_id=f"media-{i}", target_label=f"video_{i}.mp4",
            status="done", run_id="run-1", duration_seconds=10.0 + i,
        )
    for i in range(2):
        db.insert_job(
            f"ingest-run-{i}", "ingest", "transcode",
            target_id=f"media-{5 + i}", target_label=f"video_{5 + i}.mp4",
            status="running", run_id="run-1", progress_pct=50.0,
        )

    # analyze: 1 running + 2 pending
    db.insert_job(
        "analyze-run-0", "analyze", "asr",
        target_id="media-0", status="running", run_id="run-1",
    )
    for i in range(2):
        db.insert_job(
            f"analyze-pend-{i}", "analyze", "asr",
            target_id=f"media-{1 + i}", status="pending", run_id="run-1",
        )

    # classify: 1 error
    db.insert_job(
        "classify-err-0", "classify", "cluster",
        target_id="media-0", status="error",
        error_message="Clustering failed: OOM", run_id="run-1",
    )

    # gates
    db.init_default_gates()
    db.update_gate("analyze", mode="manual")

    # events
    db.insert_event("stage_error", stage="classify",
                     payload_json=json.dumps({"error": "OOM"}))


# ===========================================================================
# 1. App Startup Tests
# ===========================================================================


class TestAppStartup:
    """Verify basic app startup: health, root redirect, static files."""

    def test_health_returns_200_ok(self, e2e_client: TestClient) -> None:
        """GET /api/health returns 200 with {status: ok}."""
        resp = e2e_client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_root_redirects_to_dashboard(self, e2e_client: TestClient) -> None:
        """GET / returns 302/307 redirect to /dashboard."""
        resp = e2e_client.get("/", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert "/dashboard" in resp.headers["location"]

    def test_static_files_mounted(self, e2e_client: TestClient) -> None:
        """GET /static/app.css returns 200 (static mount is working)."""
        resp = e2e_client.get("/static/app.css")
        assert resp.status_code == 200
