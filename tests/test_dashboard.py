"""Tests for the pipeline dashboard page, stage cards, and API endpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app
from tests.conftest import PIPELINE_STAGES

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


# ---------------------------------------------------------------------------
# Step 1: GET /dashboard basic page tests
# ---------------------------------------------------------------------------

class TestDashboardPage:
    """Tests for GET /dashboard HTML page."""

    def test_dashboard_returns_200_html(self, dashboard_client: TestClient) -> None:
        """GET /dashboard returns 200 with text/html."""
        resp = dashboard_client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_dashboard_contains_all_stage_names(
        self, dashboard_client: TestClient
    ) -> None:
        """Dashboard page contains all 9 pipeline stage names."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        for stage in PIPELINE_STAGES:
            assert stage in html, f"Stage '{stage}' not found in dashboard HTML"

    def test_dashboard_extends_base_template(
        self, dashboard_client: TestClient
    ) -> None:
        """Dashboard extends base.html (contains nav with 'Autopilot Video')."""
        resp = dashboard_client.get("/dashboard")
        assert "Autopilot Video" in resp.text

    def test_dashboard_has_title(self, dashboard_client: TestClient) -> None:
        """Dashboard page contains 'Dashboard' in the title."""
        resp = dashboard_client.get("/dashboard")
        assert "Dashboard" in resp.text

    def test_dashboard_no_active_run_shows_banner(
        self, empty_client: TestClient
    ) -> None:
        """When no active run, dashboard shows 'Pipeline not running' banner."""
        resp = empty_client.get("/dashboard")
        assert resp.status_code == 200
        assert "not running" in resp.text.lower()


# ---------------------------------------------------------------------------
# Step 3: Stage card content tests (with seeded data)
# ---------------------------------------------------------------------------


class TestStageCardContent:
    """Tests for stage card status, progress, and gate badges."""

    def test_ingest_card_shows_running_status(
        self, dashboard_client: TestClient
    ) -> None:
        """Ingest card shows 'running' status (has 2 running jobs)."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        # Find the ingest card section
        ingest_start = html.find('id="stage-ingest"')
        ingest_end = html.find("</div>", html.find("</div>", ingest_start) + 1) + 6
        ingest_card = html[ingest_start:ingest_end]
        assert "running" in ingest_card.lower()

    def test_analyze_card_shows_running_status(
        self, dashboard_client: TestClient
    ) -> None:
        """Analyze card shows 'running' status (has running jobs)."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        analyze_start = html.find('id="stage-analyze"')
        analyze_end = html.find("</div>", html.find("</div>", analyze_start) + 1) + 6
        analyze_card = html[analyze_start:analyze_end]
        assert "running" in analyze_card.lower()

    def test_classify_card_shows_error_status(
        self, dashboard_client: TestClient
    ) -> None:
        """Classify card shows 'error' status (has error job)."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        classify_start = html.find('id="stage-classify"')
        classify_end = html.find(
            "</div>", html.find("</div>", classify_start) + 1
        ) + 6
        classify_card = html[classify_start:classify_end]
        assert "error" in classify_card.lower()

    def test_ingest_card_shows_progress(
        self, dashboard_client: TestClient
    ) -> None:
        """Ingest card shows progress like '5/7' (5 done of 7 total)."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        ingest_start = html.find('id="stage-ingest"')
        ingest_end = html.find("</div>", html.find("</div>", ingest_start) + 1) + 6
        ingest_card = html[ingest_start:ingest_end]
        assert "5/7" in ingest_card

    def test_analyze_gate_badge_shows_manual(
        self, dashboard_client: TestClient
    ) -> None:
        """Analyze card gate badge shows 'manual' mode."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        analyze_start = html.find('id="stage-analyze"')
        analyze_end = html.find("</div>", html.find("</div>", analyze_start) + 1) + 6
        analyze_card = html[analyze_start:analyze_end]
        assert "manual" in analyze_card.lower()


# ---------------------------------------------------------------------------
# Step 5: Timeline bar tests
# ---------------------------------------------------------------------------


class TestTimelineBar:
    """Tests for the pipeline timeline/progress bar."""

    def test_dashboard_contains_overall_progress(
        self, dashboard_client: TestClient
    ) -> None:
        """Dashboard HTML contains overall progress percentage."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        # Overall: 5 done out of 11 total = ~45%
        assert "id=\"timeline-bar\"" in html

    def test_dashboard_shows_budget_info(
        self, dashboard_client: TestClient
    ) -> None:
        """Dashboard shows budget remaining display."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        # budget_remaining_seconds=3600 -> 1:00:00 or similar
        assert "3600" in html or "1:00:00" in html or "60:00" in html

    def test_dashboard_shows_run_duration(
        self, dashboard_client: TestClient
    ) -> None:
        """Dashboard shows elapsed run duration."""
        resp = dashboard_client.get("/dashboard")
        html = resp.text
        # wall_clock_seconds=120 -> 2:00
        assert "2:00" in html or "120" in html

    def test_timeline_bar_not_shown_when_no_run(
        self, empty_client: TestClient
    ) -> None:
        """Timeline bar is not rendered when no active run."""
        resp = empty_client.get("/dashboard")
        assert "id=\"timeline-bar\"" not in resp.text


# ---------------------------------------------------------------------------
# Step 7: JSON API endpoint tests
# ---------------------------------------------------------------------------


class TestApiRun:
    """Tests for GET /api/run JSON endpoint."""

    def test_api_run_returns_200_json(
        self, dashboard_client: TestClient
    ) -> None:
        """GET /api/run returns 200 with JSON containing 'run' key."""
        resp = dashboard_client.get("/api/run")
        assert resp.status_code == 200
        data = resp.json()
        assert "run" in data

    def test_api_run_active_run_fields(
        self, dashboard_client: TestClient
    ) -> None:
        """Active run has run_id, status, current_stage, wall_clock_seconds."""
        resp = dashboard_client.get("/api/run")
        run = resp.json()["run"]
        assert run is not None
        assert run["run_id"] == "run-1"
        assert run["status"] == "running"
        assert run["current_stage"] == "analyze"
        assert "wall_clock_seconds" in run

    def test_api_run_no_active_run(self, empty_client: TestClient) -> None:
        """When no active run, 'run' is null."""
        resp = empty_client.get("/api/run")
        assert resp.json()["run"] is None


class TestApiStages:
    """Tests for GET /api/stages JSON endpoint."""

    def test_api_stages_returns_200_with_9_stages(
        self, dashboard_client: TestClient
    ) -> None:
        """GET /api/stages returns 200 with JSON list of 9 stages."""
        resp = dashboard_client.get("/api/stages")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 9

    def test_api_stages_has_required_fields(
        self, dashboard_client: TestClient
    ) -> None:
        """Each stage object has name, status_counts, gate_mode."""
        resp = dashboard_client.get("/api/stages")
        for stage in resp.json():
            assert "name" in stage
            assert "status_counts" in stage
            assert "gate_mode" in stage


# ---------------------------------------------------------------------------
# Step 9: HTMX stage card partial endpoint tests
# ---------------------------------------------------------------------------


class TestStageCardPartial:
    """Tests for GET /dashboard/stage/{stage_name} partial endpoint."""

    def test_stage_partial_returns_200_html(
        self, dashboard_client: TestClient
    ) -> None:
        """GET /dashboard/stage/ingest returns 200 with HTML."""
        resp = dashboard_client.get("/dashboard/stage/ingest")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "ingest" in resp.text

    def test_stage_partial_has_id_for_swap(
        self, dashboard_client: TestClient
    ) -> None:
        """Response includes id='stage-ingest' for HTMX swap targeting."""
        resp = dashboard_client.get("/dashboard/stage/ingest")
        assert 'id="stage-ingest"' in resp.text

    def test_invalid_stage_returns_404(
        self, dashboard_client: TestClient
    ) -> None:
        """Invalid stage name returns 404."""
        resp = dashboard_client.get("/dashboard/stage/nonexistent")
        assert resp.status_code == 404

    def test_stage_partial_reflects_db_state(
        self,
        dashboard_seeded_db: CatalogDB,
        dashboard_db_path: str,
    ) -> None:
        """After updating a job to done, re-fetching shows updated progress."""
        db = dashboard_seeded_db
        # Update one running ingest job to done
        db.update_job("ingest-run-0", status="done")

        app = create_app(dashboard_db_path)
        client = TestClient(app)
        resp = client.get("/dashboard/stage/ingest")
        # Now 6/7 done
        assert "6/7" in resp.text


# ---------------------------------------------------------------------------
# Step 11: SSE integration tests (content-based checks on app.js)
# ---------------------------------------------------------------------------

_APP_JS = Path(__file__).resolve().parent.parent / "autopilot" / "web" / "static" / "app.js"


def _extract_js_function(content: str, func_name: str) -> str:
    """Extract the body of a named JS function by brace-matching.

    Locates ``function <func_name>`` in *content* using a word-boundary regex
    (avoids prefix collisions e.g. 'makeStageHandler' vs 'makeStageHandlerV2'),
    finds the opening ``{``, then brace-counts to the matching ``}``.

    Returns the outermost ``{ ... }`` block (braces included).

    Raises :class:`AssertionError` with *func_name* in the message if the
    function is not found or if the braces are unbalanced.
    """
    match = re.search(rf'\bfunction\s+{re.escape(func_name)}\b', content)
    assert match is not None, f"{func_name} function not found in source"
    body_start = content.find("{", match.end())
    assert body_start != -1, f"opening brace not found for {func_name}"
    depth = 0
    i = body_start
    while i < len(content):
        if content[i] == "{":
            depth += 1
        elif content[i] == "}":
            depth -= 1
            if depth == 0:
                return content[body_start : i + 1]
        i += 1
    raise AssertionError(f"unbalanced braces for {func_name} (depth {depth} at end)")


class TestSSEIntegration:
    """Verify app.js contains SSE event handlers for dashboard updates."""

    def test_app_js_uses_connect_sse(self) -> None:
        """app.js contains EventSource or connectSSE usage for '/api/events'."""
        content = _APP_JS.read_text()
        assert "connectSSE" in content or "EventSource" in content
        assert "/api/events" in content

    def test_app_js_has_stage_event_handlers(self) -> None:
        """app.js has event handlers for stage_started, stage_completed, and job_progress."""
        content = _APP_JS.read_text()
        assert "stage_started" in content
        assert "stage_completed" in content
        assert "job_progress" in content

    def test_app_js_triggers_htmx_stage_update(self) -> None:
        """Handlers contain htmx.ajax or fetch call to /dashboard/stage/ for card updates."""
        content = _APP_JS.read_text()
        assert "htmx.ajax" in content or "/dashboard/stage/" in content

    def test_app_js_has_debounce_timer_map(self) -> None:
        """app.js declares _refreshTimers map and DEBOUNCE_MS constant with positive value."""
        content = _APP_JS.read_text()
        assert "_refreshTimers" in content, "_refreshTimers not found in app.js"
        assert "DEBOUNCE_MS" in content, "DEBOUNCE_MS not found in app.js"
        match = re.search(r"\bDEBOUNCE_MS\s*=\s*(\d+)", content)
        assert match is not None, "DEBOUNCE_MS numeric literal not found in app.js"
        assert int(match.group(1)) > 0, "DEBOUNCE_MS must be a positive integer"


# ---------------------------------------------------------------------------
# Step 13: GET / redirect test
# ---------------------------------------------------------------------------


class TestRootRedirect:
    """Tests for GET / redirecting to /dashboard."""

    def test_root_redirects_to_dashboard(
        self, dashboard_client: TestClient
    ) -> None:
        """GET / returns redirect (307 or 302) to /dashboard."""
        resp = dashboard_client.get("/", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert "/dashboard" in resp.headers["location"]
