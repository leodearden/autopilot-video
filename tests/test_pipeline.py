"""Tests for pipeline detail pages: /pipeline, /pipeline/stages, /pipeline/jobs."""

from __future__ import annotations

from pathlib import Path

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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pipeline_db_path(tmp_path: Path) -> str:
    return str(tmp_path / "catalog.db")


@pytest.fixture
def pipeline_db(pipeline_db_path: str) -> CatalogDB:
    db = CatalogDB(pipeline_db_path)
    db.conn.isolation_level = None
    return db


@pytest.fixture
def seeded_db(pipeline_db: CatalogDB) -> CatalogDB:
    """DB seeded with a run, jobs across stages, and gates."""
    db = pipeline_db
    db.insert_run(
        "run-1",
        started_at="2025-06-15T10:00:00",
        current_stage="analyze",
        status="running",
    )
    # Ingest: 3 done, 1 running
    for i in range(3):
        db.insert_job(
            f"ingest-done-{i}", "ingest", "transcode",
            status="done", run_id="run-1",
        )
    db.insert_job(
        "ingest-run-0", "ingest", "transcode",
        target_label="video_x.mp4", status="running",
        run_id="run-1", progress_pct=42.0,
    )
    # Analyze: 1 pending
    db.insert_job(
        "analyze-pend-0", "analyze", "asr",
        status="pending", run_id="run-1",
    )
    # Classify: 1 error
    db.insert_job(
        "classify-err-0", "classify", "cluster",
        status="error", error_message="OOM", run_id="run-1",
    )
    db.init_default_gates()
    return db


@pytest.fixture
def app(seeded_db: CatalogDB, pipeline_db_path: str) -> FastAPI:
    return create_app(pipeline_db_path)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def empty_app(pipeline_db: CatalogDB, pipeline_db_path: str) -> FastAPI:
    pipeline_db.init_default_gates()
    return create_app(pipeline_db_path)


@pytest.fixture
def empty_client(empty_app: FastAPI) -> TestClient:
    return TestClient(empty_app)


# ---------------------------------------------------------------------------
# GET /pipeline
# ---------------------------------------------------------------------------


class TestPipelineOverview:
    """Tests for the pipeline overview page."""

    def test_returns_200_html(self, client: TestClient) -> None:
        resp = client.get("/pipeline")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_contains_all_stage_names(self, client: TestClient) -> None:
        resp = client.get("/pipeline")
        html = resp.text
        for stage in PIPELINE_STAGES:
            assert stage in html, f"Stage '{stage}' not in pipeline page"

    def test_shows_run_id(self, client: TestClient) -> None:
        resp = client.get("/pipeline")
        assert "run-1" in resp.text

    def test_empty_state_no_run(self, empty_client: TestClient) -> None:
        resp = empty_client.get("/pipeline")
        assert resp.status_code == 200
        assert "no active" in resp.text.lower()

    def test_extends_base_template(self, client: TestClient) -> None:
        resp = client.get("/pipeline")
        assert "Autopilot Video" in resp.text


# ---------------------------------------------------------------------------
# GET /pipeline/stages
# ---------------------------------------------------------------------------


class TestPipelineStages:
    """Tests for the stage detail page."""

    def test_returns_200_html(self, client: TestClient) -> None:
        resp = client.get("/pipeline/stages")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_contains_stage_names(self, client: TestClient) -> None:
        resp = client.get("/pipeline/stages")
        html = resp.text
        for stage in PIPELINE_STAGES:
            assert stage in html

    def test_shows_job_counts(self, client: TestClient) -> None:
        resp = client.get("/pipeline/stages")
        html = resp.text
        # Ingest has 3 done of 4 total
        assert "3/4" in html

    def test_empty_state(self, empty_client: TestClient) -> None:
        resp = empty_client.get("/pipeline/stages")
        assert resp.status_code == 200
        assert "no active" in resp.text.lower()


# ---------------------------------------------------------------------------
# GET /pipeline/jobs
# ---------------------------------------------------------------------------


class TestPipelineJobs:
    """Tests for the job list page."""

    def test_returns_200_html(self, client: TestClient) -> None:
        resp = client.get("/pipeline/jobs")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_shows_all_jobs(self, client: TestClient) -> None:
        resp = client.get("/pipeline/jobs")
        html = resp.text
        # 4 ingest + 1 analyze + 1 classify = 6 jobs
        assert "ingest-done-0" in html
        assert "analyze-pend-0" in html
        assert "classify-err-0" in html

    def test_filter_by_stage(self, client: TestClient) -> None:
        resp = client.get("/pipeline/jobs?stage=ingest")
        html = resp.text
        assert "ingest-done-0" in html
        assert "classify-err-0" not in html

    def test_filter_by_status(self, client: TestClient) -> None:
        resp = client.get("/pipeline/jobs?status=error")
        html = resp.text
        assert "classify-err-0" in html
        assert "ingest-done-0" not in html

    def test_empty_filter_result(self, client: TestClient) -> None:
        resp = client.get("/pipeline/jobs?stage=upload")
        assert resp.status_code == 200
        assert "No jobs found" in resp.text

    def test_shows_target_label(self, client: TestClient) -> None:
        resp = client.get("/pipeline/jobs?stage=ingest&status=running")
        assert "video_x.mp4" in resp.text


# ---------------------------------------------------------------------------
# GET /api/pipeline/jobs
# ---------------------------------------------------------------------------


class TestApiPipelineJobs:
    """Tests for the JSON pipeline jobs endpoint."""

    def test_returns_200_json(self, client: TestClient) -> None:
        resp = client.get("/api/pipeline/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 6  # 4 ingest + 1 analyze + 1 classify

    def test_filter_by_stage(self, client: TestClient) -> None:
        resp = client.get("/api/pipeline/jobs?stage=ingest")
        data = resp.json()
        assert len(data) == 4
        assert all(j["stage"] == "ingest" for j in data)

    def test_filter_by_status(self, client: TestClient) -> None:
        resp = client.get("/api/pipeline/jobs?status=done")
        data = resp.json()
        assert all(j["status"] == "done" for j in data)
