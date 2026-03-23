"""Tests for the pipeline dashboard page, stage cards, and API endpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dashboard_db_path(tmp_path: Path) -> str:
    """Return the path for a test catalog DB file."""
    return str(tmp_path / "catalog.db")


@pytest.fixture
def dashboard_db(dashboard_db_path: str) -> CatalogDB:
    """Create a CatalogDB backed by a real file for dashboard tests."""
    db = CatalogDB(dashboard_db_path)
    db.conn.isolation_level = None  # autocommit
    return db


@pytest.fixture
def dashboard_seeded_db(dashboard_db: CatalogDB) -> CatalogDB:
    """DB seeded with a running pipeline run, jobs, gates, and events.

    Run: run_id='run-1', current_stage='analyze', status='running',
         budget_remaining_seconds=3600, wall_clock_seconds=120.

    Jobs by stage (all run_id='run-1'):
      - ingest: 5 done, 2 running  (7 total)
      - analyze: 1 running, 2 pending  (3 total)
      - classify: 1 error  (1 total)
      - remaining stages: no jobs

    Gates: default gates initialized, analyze set to mode='manual'.

    Events: a few error events for testing.
    """
    db = dashboard_db

    # Insert a running pipeline run
    db.insert_run(
        "run-1",
        started_at="2025-06-15T10:00:00",
        config_snapshot='{"quality": "high"}',
        current_stage="analyze",
        status="running",
        budget_remaining_seconds=3600,
    )
    db.update_run("run-1", wall_clock_seconds=120.0)

    # -- ingest jobs: 5 done + 2 running --
    for i in range(5):
        db.insert_job(
            f"ingest-done-{i}",
            "ingest",
            "transcode",
            target_id=f"media-{i}",
            target_label=f"video_{i}.mp4",
            status="done",
            run_id="run-1",
            duration_seconds=10.0 + i,
        )
    for i in range(2):
        db.insert_job(
            f"ingest-run-{i}",
            "ingest",
            "transcode",
            target_id=f"media-{5 + i}",
            target_label=f"video_{5 + i}.mp4",
            status="running",
            run_id="run-1",
            progress_pct=50.0,
        )

    # -- analyze jobs: 1 running + 2 pending --
    db.insert_job(
        "analyze-run-0",
        "analyze",
        "asr",
        target_id="media-0",
        status="running",
        run_id="run-1",
    )
    for i in range(2):
        db.insert_job(
            f"analyze-pend-{i}",
            "analyze",
            "asr",
            target_id=f"media-{1 + i}",
            status="pending",
            run_id="run-1",
        )

    # -- classify jobs: 1 error --
    db.insert_job(
        "classify-err-0",
        "classify",
        "cluster",
        target_id="media-0",
        status="error",
        error_message="Clustering failed: OOM",
        run_id="run-1",
    )

    # -- Initialize default gates, then customize analyze --
    db.init_default_gates()
    db.update_gate("analyze", mode="manual")

    # -- Insert a few error events --
    db.insert_event(
        "stage_error",
        stage="classify",
        payload_json=json.dumps({"error": "OOM"}),
    )
    db.insert_event(
        "stage_started",
        stage="analyze",
    )

    return db


@pytest.fixture
def dashboard_app(dashboard_seeded_db: CatalogDB, dashboard_db_path: str) -> FastAPI:
    """Create a FastAPI app pointing at the seeded dashboard DB."""
    return create_app(dashboard_db_path)


@pytest.fixture
def dashboard_client(dashboard_app: FastAPI) -> TestClient:
    """Create a TestClient for dashboard tests."""
    return TestClient(dashboard_app)


@pytest.fixture
def empty_app(dashboard_db: CatalogDB, dashboard_db_path: str) -> FastAPI:
    """App with DB that has no pipeline runs (empty state)."""
    # Init gates so the DB is valid but no runs exist
    dashboard_db.init_default_gates()
    return create_app(dashboard_db_path)


@pytest.fixture
def empty_client(empty_app: FastAPI) -> TestClient:
    """TestClient for empty-state dashboard tests."""
    return TestClient(empty_app)
