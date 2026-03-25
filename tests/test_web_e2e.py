"""End-to-end integration tests for the full web console.

These tests verify cross-cutting workflows and data consistency across
multiple web console views. Individual endpoint tests live in their
respective test files (test_dashboard.py, test_gates.py, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator
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
def e2e_db(e2e_db_path: str) -> Generator:
    """Create a CatalogDB backed by a real file for E2E tests."""
    db = CatalogDB(e2e_db_path)
    db.conn.isolation_level = None  # autocommit
    try:
        yield db
    finally:
        db.close()


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


# ===========================================================================
# 2. Dashboard Rendering Tests
# ===========================================================================


@pytest.fixture
def dashboard_seeded_db(e2e_db: CatalogDB) -> CatalogDB:
    """DB seeded with a full pipeline run for dashboard tests."""
    _seed_pipeline(e2e_db)
    return e2e_db


@pytest.fixture
def dashboard_app(dashboard_seeded_db: CatalogDB, e2e_db_path: str) -> FastAPI:
    """App backed by seeded dashboard DB."""
    return create_app(e2e_db_path)


@pytest.fixture
def dashboard_client(dashboard_app: FastAPI) -> TestClient:
    """TestClient for dashboard tests."""
    return TestClient(dashboard_app)


class TestDashboardRendering:
    """Verify dashboard page, stage cards, and /api/stages."""

    def test_dashboard_returns_html(self, dashboard_client: TestClient) -> None:
        """GET /dashboard returns 200 text/html."""
        resp = dashboard_client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_dashboard_contains_all_9_stages(
        self, dashboard_client: TestClient,
    ) -> None:
        """Dashboard HTML contains all 9 pipeline stage names."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        for stage in PIPELINE_STAGES:
            assert stage in html, f"Stage '{stage}' not in dashboard"

    def test_stage_cards_show_status(
        self, dashboard_client: TestClient,
    ) -> None:
        """Seeded DB with jobs shows running/error status in cards."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text.lower()
        assert "running" in html
        assert "error" in html

    def test_stage_cards_show_progress(
        self, dashboard_client: TestClient,
    ) -> None:
        """Ingest stage has 5 done and 7 total jobs via /api/stages."""
        resp = dashboard_client.get("/api/stages")
        assert resp.status_code == 200
        stages = resp.json()
        ingest = next(s for s in stages if s["name"] == "ingest")
        counts = ingest["status_counts"]
        assert counts["done"] == 5
        assert counts["done"] + counts["running"] == 7

    def test_api_stages_returns_9_objects(
        self, dashboard_client: TestClient,
    ) -> None:
        """GET /api/stages returns list of 9 with name/status_counts/gate_mode."""
        resp = dashboard_client.get("/api/stages")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 9
        for stage in data:
            assert "name" in stage
            assert "status_counts" in stage
            assert "gate_mode" in stage


# ===========================================================================
# 3. SSE Event Flow Tests
# ===========================================================================


@pytest.fixture
def sse_db(e2e_db_path: str) -> Generator:
    """CatalogDB for SSE tests (separate from e2e_db to avoid fixture reuse)."""
    db = CatalogDB(e2e_db_path)
    db.conn.isolation_level = None
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def sse_app(sse_db: CatalogDB, e2e_db_path: str) -> FastAPI:
    """App for SSE tests with direct DB access for event insertion."""
    return create_app(e2e_db_path)


class TestSSEEventFlow:
    """Verify SSE endpoint content-type, event delivery, and reconnection."""

    def test_sse_endpoint_returns_event_stream(self, sse_app: FastAPI) -> None:
        """GET /api/events returns 200 with text/event-stream content-type."""
        from autopilot.web.routes import sse as sse_module

        async def _finite_gen(request):
            yield {"data": "hello"}

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            resp = client.get("/api/events")
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

    def test_events_delivered_in_order(
        self, sse_db: CatalogDB, sse_app: FastAPI,
    ) -> None:
        """Insert 3 events, verify delivery order matches insertion order."""
        from autopilot.web.routes import sse as sse_module

        sse_db.insert_event("stage_started", stage="INGEST")
        sse_db.insert_event("job_started", stage="INGEST", job_id="job-1")
        sse_db.insert_event("stage_completed", stage="INGEST")

        async def _finite_gen(request):
            db = sse_module._get_db(request)
            try:
                events = db.get_events_since(0)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            resp = client.get("/api/events")
            events = _parse_sse_body(resp.text)

        assert len(events) == 3
        assert events[0]["event"] == "stage_started"
        assert events[1]["event"] == "job_started"
        assert events[2]["event"] == "stage_completed"

    def test_event_has_id_event_data_fields(
        self, sse_db: CatalogDB, sse_app: FastAPI,
    ) -> None:
        """Each SSE event has id, event, and data fields."""
        from autopilot.web.routes import sse as sse_module

        sse_db.insert_event("stage_started", stage="ANALYZE", job_id="j1")

        async def _finite_gen(request):
            db = sse_module._get_db(request)
            try:
                events = db.get_events_since(0)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            resp = client.get("/api/events")
            events = _parse_sse_body(resp.text)

        assert len(events) == 1
        ev = events[0]
        assert "id" in ev
        assert "event" in ev
        assert "data" in ev

    def test_last_event_id_resumes(
        self, sse_db: CatalogDB, sse_app: FastAPI,
    ) -> None:
        """Connect with Last-Event-ID set to the 3rd event, verify only later events arrive."""
        from autopilot.web.routes import sse as sse_module

        valid_types = [
            "stage_started", "job_started", "job_progress",
            "job_completed", "stage_completed",
        ]
        # Capture returned IDs instead of discarding them
        event_ids = [sse_db.insert_event(et, stage="INGEST") for et in valid_types]

        async def _finite_gen(request):
            db = sse_module._get_db(request)
            last_id = sse_module._get_last_event_id(request)
            try:
                events = db.get_events_since(last_id)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        # event_ids[2] is the 3rd inserted event (job_progress).
        # get_events_since uses WHERE event_id > ?, so events at
        # indices 3 and 4 (job_completed, stage_completed) are returned.
        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            resp = client.get(
                "/api/events",
                headers={"Last-Event-ID": str(event_ids[2])},
            )
            events = _parse_sse_body(resp.text)

        assert len(events) == 2
        assert events[0]["event"] == "job_completed"
        assert events[1]["event"] == "stage_completed"

    def test_invalid_last_event_id_gets_all(
        self, sse_db: CatalogDB, sse_app: FastAPI,
    ) -> None:
        """Non-numeric Last-Event-ID falls back to 0 (all events)."""
        from autopilot.web.routes import sse as sse_module

        valid_types = ["stage_started", "job_started", "job_completed"]
        for et in valid_types:
            sse_db.insert_event(et, stage="INGEST")

        async def _finite_gen(request):
            db = sse_module._get_db(request)
            last_id = sse_module._get_last_event_id(request)
            try:
                events = db.get_events_since(last_id)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            resp = client.get("/api/events", headers={"Last-Event-ID": "abc"})
            events = _parse_sse_body(resp.text)

        assert len(events) == 3


# ===========================================================================
# 4. Gate Workflow Tests
# ===========================================================================


@pytest.fixture
def gates_app(tmp_path: Path) -> FastAPI:
    """App with default gates initialized."""
    db_path = str(tmp_path / "gates.db")
    with CatalogDB(db_path) as db:
        db.init_default_gates()
    return create_app(db_path)


@pytest.fixture
def gates_client(gates_app: FastAPI) -> TestClient:
    """TestClient for gate workflow tests."""
    return TestClient(gates_app)


class TestGateWorkflow:
    """Verify gate CRUD: list, detail, update, approve, skip, presets."""

    def test_get_gates_returns_9_gates(self, gates_client: TestClient) -> None:
        """GET /api/gates returns list of 9 gates with default modes."""
        resp = gates_client.get("/api/gates")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 9
        for gate in data:
            assert gate["mode"] == "auto"

    def test_get_single_gate(self, gates_client: TestClient) -> None:
        """GET /api/gates/narrate returns gate dict."""
        resp = gates_client.get("/api/gates/narrate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stage"] == "narrate"
        assert data["mode"] == "auto"

    def test_update_gate_mode(self, gates_client: TestClient) -> None:
        """PUT /api/gates/narrate with mode='pause' persists."""
        resp = gates_client.put("/api/gates/narrate", json={"mode": "pause"})
        assert resp.status_code == 200
        assert resp.json()["mode"] == "pause"
        # Verify persistence
        check = gates_client.get("/api/gates/narrate")
        assert check.json()["mode"] == "pause"

    def test_approve_gate(self, gates_client: TestClient) -> None:
        """POST /api/gates/narrate/approve sets status to 'approved'."""
        resp = gates_client.post("/api/gates/narrate/approve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "approved"
        assert data["decided_by"] == "console"

    def test_skip_gate(self, gates_client: TestClient) -> None:
        """POST /api/gates/narrate/skip sets status to 'skipped'."""
        resp = gates_client.post("/api/gates/narrate/skip")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "skipped"
        assert data["decided_by"] == "console"

    def test_apply_preset_review_creative(
        self, gates_client: TestClient,
    ) -> None:
        """PUT review_creative preset pauses narrate, script, upload."""
        resp = gates_client.put("/api/gates/preset/review_creative")
        assert resp.status_code == 200
        modes = {g["stage"]: g["mode"] for g in resp.json()}
        assert modes["narrate"] == "pause"
        assert modes["script"] == "pause"
        assert modes["upload"] == "pause"
        for stage in ("ingest", "analyze", "classify", "edl", "source", "render"):
            assert modes[stage] == "auto"

    def test_apply_preset_full_auto(self, gates_client: TestClient) -> None:
        """PUT full_auto preset sets all gates to auto."""
        # First pause everything, then reset
        gates_client.put("/api/gates/preset/review_everything")
        resp = gates_client.put("/api/gates/preset/full_auto")
        assert resp.status_code == 200
        for gate in resp.json():
            assert gate["mode"] == "auto"


# ===========================================================================
# 5. Media Browsing Tests
# ===========================================================================


@pytest.fixture
def media_seeded_app(tmp_path: Path) -> FastAPI:
    """App seeded with 12 media files, transcripts, detections, faces, etc."""
    db_path = str(tmp_path / "media.db")
    with CatalogDB(db_path) as db:
        db.conn.isolation_level = None
        # 12 media files: 5 analyzed, 4 pending, 3 ingested
        for i in range(5):
            _seed_media(db, f"m-{i}", status="analyzed",
                        file_path=f"/video/video_{i}.mp4")
        for i in range(5, 9):
            _seed_media(db, f"m-{i}", status="pending",
                        file_path=f"/video/video_{i}.mp4")
        for i in range(9, 12):
            _seed_media(db, f"m-{i}", status="ingested",
                        file_path=f"/video/video_{i}.mp4")

        # Transcripts for m-0, m-1, m-2
        for mid in ("m-0", "m-1", "m-2"):
            segments = [
                {"start": 0.0, "end": 10.0, "text": "Hello", "speaker": "A"},
                {"start": 10.0, "end": 20.0, "text": "World", "speaker": "B"},
            ]
            db.upsert_transcript(mid, json.dumps(segments), "en")

        # Detections for m-0, m-1
        dets = [
            ("m-0", 0, json.dumps([{"class": "person", "confidence": 0.9}])),
            ("m-0", 30, json.dumps([{"class": "car", "confidence": 0.8}])),
            ("m-1", 0, json.dumps([{"class": "dog", "confidence": 0.7}])),
        ]
        db.batch_insert_detections(dets)

        # Faces for m-0
        faces: list[tuple[str, int, int, str, bytes | None, int | None]] = [
            ("m-0", 0, 0, json.dumps([10, 20, 100, 200]), None, None),
        ]
        db.batch_insert_faces(faces)

        # Embeddings for m-0, m-1
        embs = [
            ("m-0", 0, b"\x00" * 16),
            ("m-0", 30, b"\x00" * 16),
            ("m-1", 0, b"\x00" * 16),
        ]
        db.batch_insert_embeddings(embs)

        # Audio events for m-0
        audio = [
            ("m-0", 5.0, json.dumps({"type": "speech", "label": "voice"})),
        ]
        db.batch_insert_audio_events(audio)

    return create_app(db_path)


@pytest.fixture
def media_client(media_seeded_app: FastAPI) -> TestClient:
    """TestClient for media browsing tests."""
    return TestClient(media_seeded_app)


class TestMediaBrowsing:
    """Verify media list, search, detail, and tab endpoints."""

    def test_media_list_page_returns_html(
        self, media_client: TestClient,
    ) -> None:
        """GET /media returns 200 HTML."""
        resp = media_client.get("/media")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_media_api_returns_paginated(
        self, media_client: TestClient,
    ) -> None:
        """GET /api/media returns items + total."""
        resp = media_client.get("/api/media")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] == 12

    def test_media_filter_by_status(
        self, media_client: TestClient,
    ) -> None:
        """GET /api/media?status=analyzed returns only matching."""
        resp = media_client.get("/api/media", params={"status": "analyzed"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 5
        for item in data["items"]:
            assert item["status"] == "analyzed"

    def test_media_search_by_name(
        self, media_client: TestClient,
    ) -> None:
        """GET /api/media?q=video_3 returns matching item."""
        resp = media_client.get("/api/media", params={"q": "video_3"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        paths = [item["file_path"] for item in data["items"]]
        assert any("video_3" in p for p in paths)

    def test_media_detail_page(
        self, media_client: TestClient,
    ) -> None:
        """GET /media/{id} returns 200 HTML with media info."""
        resp = media_client.get("/media/m-0")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_media_detail_api(
        self, media_client: TestClient,
    ) -> None:
        """GET /api/media/{id} returns JSON with expected keys."""
        resp = media_client.get("/api/media/m-0")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("media", "transcript", "detections", "faces",
                     "audio_events", "embedding_count"):
            assert key in data, f"Missing key: {key}"
        assert data["embedding_count"] == 2

    def test_media_tab_transcript(
        self, media_client: TestClient,
    ) -> None:
        """GET /media/{id}/tab/transcript returns HTML with segment text."""
        resp = media_client.get("/media/m-0/tab/transcript")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Hello" in resp.text


# ===========================================================================
# 6. Narrative Review Tests
# ===========================================================================


@pytest.fixture
def narrative_seeded_app(tmp_path: Path) -> FastAPI:
    """App with narrate gate waiting and 3 seeded narratives."""
    db_path = str(tmp_path / "narrative.db")
    with CatalogDB(db_path) as db:
        db.init_default_gates()
        db.update_gate("narrate", status="waiting")
        db.conn.commit()
        _seed_narrative(db, "n-1", title="Morning Walk", status="proposed")
        _seed_narrative(db, "n-2", title="Sunset Hike", status="proposed")
        _seed_narrative(db, "n-3", title="Beach Day", status="approved")
    return create_app(db_path)


@pytest.fixture
def narrative_client(narrative_seeded_app: FastAPI) -> TestClient:
    """TestClient for narrative review tests."""
    return TestClient(narrative_seeded_app)


class TestNarrativeReview:
    """Verify review hub, narrative listing, approve, reject."""

    def test_review_hub_returns_html(
        self, narrative_client: TestClient,
    ) -> None:
        """GET /review returns 200 HTML."""
        resp = narrative_client.get("/review")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_review_hub_shows_waiting_gate(
        self, narrative_client: TestClient,
    ) -> None:
        """With narrate gate waiting, hub page shows 'narrate'."""
        resp = narrative_client.get("/review")
        assert "narrate" in resp.text.lower()

    def test_list_narratives_api(
        self, narrative_client: TestClient,
    ) -> None:
        """GET /api/narratives returns list of 3."""
        resp = narrative_client.get("/api/narratives")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3

    def test_filter_narratives_by_status(
        self, narrative_client: TestClient,
    ) -> None:
        """GET /api/narratives?status=proposed returns only proposed."""
        resp = narrative_client.get(
            "/api/narratives", params={"status": "proposed"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert all(n["status"] == "proposed" for n in data)

    def test_approve_narrative(
        self, narrative_client: TestClient,
    ) -> None:
        """POST /api/narratives/n-1/approve sets status=approved."""
        resp = narrative_client.post("/api/narratives/n-1/approve")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "approved"
        assert data["narrative_id"] == "n-1"

    def test_reject_narrative(
        self, narrative_client: TestClient,
    ) -> None:
        """POST /api/narratives/n-1/reject sets status=rejected."""
        resp = narrative_client.post("/api/narratives/n-1/reject")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rejected"
        assert data["narrative_id"] == "n-1"

    def test_narratives_page_shows_titles(
        self, narrative_client: TestClient,
    ) -> None:
        """GET /review/narratives shows all narrative titles."""
        resp = narrative_client.get("/review/narratives")
        assert resp.status_code == 200
        assert "Morning Walk" in resp.text
        assert "Sunset Hike" in resp.text
        assert "Beach Day" in resp.text


# ===========================================================================
# 7. Cluster Review Tests
# ===========================================================================


@pytest.fixture
def cluster_seeded_app(tmp_path: Path) -> FastAPI:
    """App seeded with 3 activity clusters."""
    db_path = str(tmp_path / "cluster.db")
    with CatalogDB(db_path) as db:
        _seed_cluster(db, "c-1", label="Morning Walk",
                      clip_ids_json='["clip-1","clip-2"]',
                      time_start="2025-01-01T08:00:00",
                      time_end="2025-01-01T09:00:00")
        _seed_cluster(db, "c-2", label="Lunch Break",
                      clip_ids_json='["clip-3","clip-4","clip-5"]',
                      time_start="2025-01-01T12:00:00",
                      time_end="2025-01-01T13:00:00")
        _seed_cluster(db, "c-3", label="Evening Run",
                      clip_ids_json='["clip-6"]',
                      time_start="2025-01-01T18:00:00",
                      time_end="2025-01-01T19:00:00")
    return create_app(db_path)


@pytest.fixture
def cluster_client(cluster_seeded_app: FastAPI) -> TestClient:
    """TestClient for cluster review tests."""
    return TestClient(cluster_seeded_app)


class TestClusterReview:
    """Verify cluster pages, list, relabel, exclude, merge."""

    def test_clusters_page_shows_cards(
        self, cluster_client: TestClient,
    ) -> None:
        """GET /review/clusters shows cluster labels."""
        resp = cluster_client.get("/review/clusters")
        assert resp.status_code == 200
        assert "Morning Walk" in resp.text
        assert "Lunch Break" in resp.text

    def test_list_clusters_api(
        self, cluster_client: TestClient,
    ) -> None:
        """GET /api/clusters returns list with clip_ids + clip_count."""
        resp = cluster_client.get("/api/clusters")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        for c in data:
            assert "clip_ids" in c
            assert "clip_count" in c

    def test_relabel_cluster(
        self, cluster_client: TestClient,
    ) -> None:
        """POST /api/clusters/c-1/relabel updates and persists."""
        resp = cluster_client.post(
            "/api/clusters/c-1/relabel",
            json={"label": "Dawn Walk", "description": "Early morning"},
        )
        assert resp.status_code == 200
        assert resp.json()["label"] == "Dawn Walk"
        # Verify persistence
        check = cluster_client.get("/api/clusters/c-1")
        assert check.json()["label"] == "Dawn Walk"

    def test_exclude_cluster(
        self, cluster_client: TestClient,
    ) -> None:
        """POST /api/clusters/c-3/exclude sets excluded=1 and persists."""
        resp = cluster_client.post("/api/clusters/c-3/exclude")
        assert resp.status_code == 200
        assert resp.json()["excluded"] == 1
        # Verify persistence (mirrors test_relabel_cluster pattern)
        check = cluster_client.get("/api/clusters/c-3")
        assert check.json()["excluded"] == 1

    def test_merge_clusters(
        self, cluster_client: TestClient,
    ) -> None:
        """POST /api/clusters/merge combines clips and extends time range."""
        resp = cluster_client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        # c-2 has more clips so it survives
        assert data["cluster_id"] == "c-2"
        assert data["clip_count"] == 5
        # Time range: min start, max end
        assert data["time_start"] == "2025-01-01T08:00:00"
        assert data["time_end"] == "2025-01-01T13:00:00"
        # c-1 deleted
        check = cluster_client.get("/api/clusters/c-1")
        assert check.status_code == 404


# ===========================================================================
# 8. Render / Script Review Tests
# ===========================================================================


@pytest.fixture
def render_seeded_app(tmp_path: Path) -> FastAPI:
    """App seeded with narrative, edit plan, script, upload, and render file."""
    db_path = str(tmp_path / "render.db")
    video_file = tmp_path / "render.mp4"
    video_file.write_bytes(b"\x00" * 512)

    with CatalogDB(db_path) as db:
        db.init_default_gates()
        _seed_narrative(db, "n-1", title="Morning Walk", status="approved")

        # Edit plan with validation and render path
        db.upsert_edit_plan(
            "n-1",
            edl_json='{"cuts": [{"in": 0, "out": 10}]}',
            otio_path=str(tmp_path / "edit.otio"),
            validation_json=json.dumps({
                "resolution": "1920x1080",
                "duration_seconds": 120.5,
                "codec": "h264",
                "passes": True,
            }),
            render_path=str(video_file),
        )

        # Narrative script with scenes
        db.upsert_narrative_script(
            "n-1",
            script_json=json.dumps({
                "scenes": [
                    {"title": "Opening", "duration": 15, "voiceover": "Welcome"},
                    {"title": "Main", "duration": 90, "voiceover": "The story"},
                ],
            }),
        )

        # Upload record
        db.insert_upload(
            "n-1",
            youtube_video_id="abc123",
            youtube_url="https://youtube.com/watch?v=abc123",
            uploaded_at="2025-01-15T10:30:00",
            privacy_status="unlisted",
        )
        db.conn.commit()

    return create_app(db_path)


@pytest.fixture
def render_client(render_seeded_app: FastAPI) -> TestClient:
    """TestClient for render/script review tests."""
    return TestClient(render_seeded_app)


class TestRenderScriptReview:
    """Verify render review index, detail, video stream, and uploads."""

    def test_render_review_index(
        self, render_client: TestClient,
    ) -> None:
        """GET /review/render returns 200 HTML listing narratives."""
        resp = render_client.get("/review/render")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_render_detail_shows_validation(
        self, render_client: TestClient,
    ) -> None:
        """GET /review/render/n-1 shows validation results."""
        resp = render_client.get("/review/render/n-1")
        assert resp.status_code == 200
        assert "1920x1080" in resp.text
        assert "h264" in resp.text

    def test_render_detail_shows_script_scenes(
        self, render_client: TestClient,
    ) -> None:
        """Render detail page contains scene data from narrative_scripts."""
        resp = render_client.get("/review/render/n-1")
        assert resp.status_code == 200
        # Both seeded scene titles must appear in the rendered detail page
        assert "Opening" in resp.text
        assert "Main" in resp.text

    def test_render_api_returns_json(
        self, render_client: TestClient,
    ) -> None:
        """GET /api/renders/n-1 returns narrative_id + validation dict."""
        resp = render_client.get("/api/renders/n-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["narrative_id"] == "n-1"
        assert isinstance(data["validation"], dict)
        assert data["validation"]["passes"] is True
        assert data["render_path"] is not None

    def test_render_video_returns_mp4(
        self, render_client: TestClient,
    ) -> None:
        """GET /api/renders/n-1/video returns video/mp4 when file exists."""
        resp = render_client.get("/api/renders/n-1/video")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"

    def test_uploads_page(
        self, render_client: TestClient,
    ) -> None:
        """GET /review/uploads shows upload data with YouTube URL."""
        resp = render_client.get("/review/uploads")
        assert resp.status_code == 200
        assert "abc123" in resp.text
        assert "youtube.com" in resp.text


# ===========================================================================
# 9. Job Status Tests (reuse dashboard_client with mixed jobs)
# ===========================================================================


class TestJobStatus:
    """Verify job management visibility through the dashboard API."""

    def test_stages_api_shows_job_counts(
        self, dashboard_client: TestClient,
    ) -> None:
        """GET /api/stages includes status_counts for stages with jobs."""
        resp = dashboard_client.get("/api/stages")
        assert resp.status_code == 200
        data = resp.json()
        by_name = {s["name"]: s for s in data}
        # Ingest: 5 done + 2 running = 7 total
        ingest = by_name["ingest"]["status_counts"]
        assert ingest.get("done", 0) == 5
        assert ingest.get("running", 0) == 2

    def test_stage_card_reflects_error_jobs(
        self, dashboard_client: TestClient,
    ) -> None:
        """Stage with error jobs shows error status."""
        resp = dashboard_client.get("/api/stages")
        by_name = {s["name"]: s for s in resp.json()}
        classify = by_name["classify"]["status_counts"]
        assert classify.get("error", 0) == 1

    def test_stage_card_reflects_running_jobs(
        self, dashboard_client: TestClient,
    ) -> None:
        """Stage with running jobs shows running status."""
        resp = dashboard_client.get("/api/stages")
        by_name = {s["name"]: s for s in resp.json()}
        analyze = by_name["analyze"]["status_counts"]
        assert analyze.get("running", 0) == 1
        assert analyze.get("pending", 0) == 2


# ===========================================================================
# 10. Full Pipeline Simulation Tests
# ===========================================================================


@pytest.fixture
def pipeline_simulation_app(tmp_path: Path) -> FastAPI:
    """App seeded with a pipeline run, one job per stage, and default gates."""
    db_path = str(tmp_path / "pipeline.db")
    with CatalogDB(db_path) as db:
        db.conn.isolation_level = None

        # Default gates
        db.init_default_gates()

        # Pipeline run
        db.insert_run(
            "sim-run-1",
            started_at="2025-06-15T10:00:00",
            config_snapshot='{"mode": "simulation"}',
            current_stage="ingest",
            status="running",
            budget_remaining_seconds=7200,
        )

        # One job per stage
        for stage in PIPELINE_STAGES:
            db.insert_job(
                f"sim-{stage}-0", stage, "task",
                target_id=f"media-{stage}",
                target_label=f"{stage}_video.mp4",
                status="pending",
                run_id="sim-run-1",
            )

        # Set all gates to waiting (simulating pipeline reaching each gate)
        for stage in PIPELINE_STAGES:
            db.update_gate(stage, status="waiting")

    return create_app(db_path)


@pytest.fixture
def pipeline_client(pipeline_simulation_app: FastAPI) -> TestClient:
    """TestClient for pipeline simulation tests."""
    return TestClient(pipeline_simulation_app)


class TestFullPipelineSimulation:
    """Verify end-to-end gate approval workflow across all 9 pipeline stages."""

    def test_complete_gate_workflow(
        self, pipeline_client: TestClient,
    ) -> None:
        """Walk through approving every gate in sequence.

        1. Apply 'review_everything' preset (all gates pause mode).
        2. Set all gates to 'waiting' via DB (simulating pipeline reaching gate).
        3. Approve each of the 9 gates sequentially via API.
        4. Verify each gate becomes 'approved' after approval.
        5. At the end, verify all 9 gates show 'approved'.
        """
        # Step 1: Apply review_everything preset → all gates mode='pause'
        resp = pipeline_client.put("/api/gates/preset/review_everything")
        assert resp.status_code == 200
        for gate in resp.json():
            assert gate["mode"] == "pause"

        # Step 2: Set all gates to 'waiting' status via DB
        # (the fixture's pipeline_simulation_db is responsible for this)
        # The fixture already sets status='waiting' for all gates.

        # Step 3 & 4: Approve each gate and verify
        for stage in PIPELINE_STAGES:
            resp = pipeline_client.post(f"/api/gates/{stage}/approve")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "approved", (
                f"Gate {stage} should be approved, got {data['status']}"
            )
            assert data["decided_by"] == "console"

            # Verify via GET
            check = pipeline_client.get(f"/api/gates/{stage}")
            assert check.status_code == 200
            assert check.json()["status"] == "approved"

        # Step 5: Final verification — all 9 gates approved
        resp = pipeline_client.get("/api/gates")
        assert resp.status_code == 200
        gates = resp.json()
        assert len(gates) == 9
        for gate in gates:
            assert gate["status"] == "approved", (
                f"Gate {gate['stage']} should be approved at end"
            )
