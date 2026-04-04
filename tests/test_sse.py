"""Tests for the SSE event endpoint at GET /api/events."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app


@pytest.fixture
def sse_db(tmp_path: Path) -> CatalogDB:
    """Create a CatalogDB backed by a real file so the app can connect to it."""
    db_path = str(tmp_path / "catalog.db")
    db = CatalogDB(db_path)
    db.conn.isolation_level = None  # autocommit for test convenience
    db._test_path = db_path  # type: ignore[attr-defined]  # stash for fixture use
    return db


@pytest.fixture
def sse_app(sse_db: CatalogDB) -> FastAPI:
    """Create a FastAPI app pointing at the same DB as sse_db."""
    return create_app(sse_db._test_path)  # type: ignore[attr-defined]


@pytest.fixture
def sse_client(sse_app: FastAPI):
    """Create a TestClient for SSE tests."""
    return TestClient(sse_app)


def _parse_sse_body(text: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts.

    Each event has 'id', 'event', and 'data' keys extracted from
    the SSE wire format.
    """
    events = []
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


class TestSSEEndpointBasic:
    """Tests for the basic SSE endpoint behavior."""

    def test_sse_endpoint_returns_200_event_stream(self, sse_app) -> None:
        """GET /api/events returns 200 with content-type text/event-stream.

        We patch the generator to yield one event then stop, avoiding the
        infinite stream that would hang tests.
        """
        from autopilot.web.routes import sse as sse_module

        async def _finite_gen(request):
            yield {"data": "hello"}

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            response = client.get("/api/events")
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]


class TestSSEEventDelivery:
    """Tests for SSE event delivery from the database."""

    def test_events_delivered_in_order(self, sse_db, sse_app) -> None:
        """Events inserted into DB are delivered in correct order via SSE."""
        from autopilot.web.routes import sse as sse_module

        sse_db.insert_event("stage_started", stage="INGEST")
        sse_db.insert_event("job_started", stage="INGEST", job_id="job-1")
        sse_db.insert_event("stage_completed", stage="INGEST")

        async def _finite_gen(request):
            db = sse_module.get_db(request)
            try:
                events = db.get_events_since(0)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            response = client.get("/api/events")
            events = _parse_sse_body(response.text)

        assert len(events) == 3
        assert events[0]["event"] == "stage_started"
        assert events[1]["event"] == "job_started"
        assert events[2]["event"] == "stage_completed"

    def test_event_has_correct_sse_fields(self, sse_db, sse_app) -> None:
        """Each SSE event has id, event, and data fields."""
        from autopilot.web.routes import sse as sse_module

        sse_db.insert_event("stage_started", stage="ANALYZE", job_id="j1")

        async def _finite_gen(request):
            db = sse_module.get_db(request)
            try:
                events = db.get_events_since(0)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            response = client.get("/api/events")
            events = _parse_sse_body(response.text)

        assert len(events) == 1
        ev = events[0]
        assert "id" in ev
        assert "event" in ev
        assert "data" in ev
        assert ev["event"] == "stage_started"
        data = json.loads(ev["data"])
        assert data["event_type"] == "stage_started"
        assert data["stage"] == "ANALYZE"
        assert data["job_id"] == "j1"

    def test_event_data_payload_structure(self, sse_db, sse_app) -> None:
        """JSON data payload contains event_type, stage, job_id, and parsed payload."""
        from autopilot.web.routes import sse as sse_module

        sse_db.insert_event(
            "stage_completed",
            stage="INGEST",
            payload_json=json.dumps({"elapsed": 1.23}),
        )

        async def _finite_gen(request):
            db = sse_module.get_db(request)
            try:
                events = db.get_events_since(0)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            response = client.get("/api/events")
            events = _parse_sse_body(response.text)

        data = json.loads(events[0]["data"])
        assert data["event_type"] == "stage_completed"
        assert data["stage"] == "INGEST"
        assert data["elapsed"] == 1.23


class TestSSEReconnection:
    """Tests for Last-Event-ID reconnection support."""

    def test_last_event_id_header_resumes_from_position(self, sse_db, sse_app) -> None:
        """Connecting with Last-Event-ID: 3 skips events 1-3, sends 4-5."""
        from autopilot.web.routes import sse as sse_module

        # Insert 5 events (IDs will be 1-5)
        for i in range(5):
            sse_db.insert_event(f"event_{i}", stage="INGEST")

        async def _finite_gen(request):
            db = sse_module.get_db(request)
            last_id = sse_module._get_last_event_id(request)
            try:
                events = db.get_events_since(last_id)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            response = client.get("/api/events", headers={"Last-Event-ID": "3"})
            events = _parse_sse_body(response.text)

        assert len(events) == 2
        assert events[0]["event"] == "event_3"
        assert events[1]["event"] == "event_4"

    def test_missing_last_event_id_starts_from_zero(self, sse_db, sse_app) -> None:
        """Without Last-Event-ID header, all events are received."""
        from autopilot.web.routes import sse as sse_module

        for i in range(3):
            sse_db.insert_event(f"event_{i}", stage="INGEST")

        async def _finite_gen(request):
            db = sse_module.get_db(request)
            last_id = sse_module._get_last_event_id(request)
            try:
                events = db.get_events_since(last_id)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            response = client.get("/api/events")
            events = _parse_sse_body(response.text)

        assert len(events) == 3

    def test_invalid_last_event_id_starts_from_zero(self, sse_db, sse_app) -> None:
        """Non-numeric Last-Event-ID falls back to 0 (all events)."""
        from autopilot.web.routes import sse as sse_module

        for i in range(3):
            sse_db.insert_event(f"event_{i}", stage="INGEST")

        async def _finite_gen(request):
            db = sse_module.get_db(request)
            last_id = sse_module._get_last_event_id(request)
            try:
                events = db.get_events_since(last_id)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            response = client.get("/api/events", headers={"Last-Event-ID": "abc"})
            events = _parse_sse_body(response.text)

        assert len(events) == 3


class TestSSEPruning:
    """Tests for periodic event pruning."""

    def test_old_events_pruned_after_polls(self, sse_db) -> None:
        """Events older than 24 hours are pruned after sufficient poll cycles."""
        # Insert an old event via raw SQL (>24h ago)
        sse_db.conn.execute(
            "INSERT INTO pipeline_events (event_type, stage, created_at) "
            "VALUES (?, ?, datetime('now', '-25 hours'))",
            ("old_event", "INGEST"),
        )
        # Insert a recent event
        sse_db.insert_event("recent_event", stage="INGEST")

        # Verify both exist
        all_events = sse_db.conn.execute("SELECT * FROM pipeline_events").fetchall()
        assert len(all_events) == 2

        # Prune should remove old event
        sse_db.prune_events(hours=24)

        remaining = sse_db.conn.execute("SELECT * FROM pipeline_events").fetchall()
        assert len(remaining) == 1
        assert remaining[0]["event_type"] == "recent_event"

    def test_recent_events_not_pruned(self, sse_db) -> None:
        """Recent events are not affected by pruning."""
        for i in range(3):
            sse_db.insert_event(f"event_{i}", stage="INGEST")

        sse_db.prune_events(hours=24)

        remaining = sse_db.conn.execute("SELECT * FROM pipeline_events").fetchall()
        assert len(remaining) == 3


class TestSSEEventTypes:
    """Tests for event type handling and the VALID_EVENT_TYPES constant."""

    ALL_EVENT_TYPES = [
        "stage_started",
        "stage_completed",
        "stage_error",
        "job_started",
        "job_completed",
        "job_error",
        "job_progress",
        "gate_waiting",
        "gate_approved",
        "gate_skipped",
        "run_completed",
        "run_failed",
        "notification",
    ]

    def test_all_event_types_forwarded(self, sse_db, sse_app) -> None:
        """Each of the 13 event types is forwarded correctly via SSE."""
        from autopilot.web.routes import sse as sse_module

        for etype in self.ALL_EVENT_TYPES:
            sse_db.insert_event(etype, stage="TEST")

        async def _finite_gen(request):
            db = sse_module.get_db(request)
            try:
                events = db.get_events_since(0)
                for ev in events:
                    yield sse_module._format_event(ev)
            finally:
                db.close()

        with patch.object(sse_module, "_event_generator", _finite_gen):
            client = TestClient(sse_app)
            response = client.get("/api/events")
            events = _parse_sse_body(response.text)

        received_types = [ev["event"] for ev in events]
        for etype in self.ALL_EVENT_TYPES:
            assert etype in received_types, f"Event type '{etype}' not received"

    def test_valid_event_types_constant_defined(self) -> None:
        """VALID_EVENT_TYPES constant exists and contains all expected types."""
        from autopilot.web.routes.sse import VALID_EVENT_TYPES

        for etype in self.ALL_EVENT_TYPES:
            assert etype in VALID_EVENT_TYPES, f"'{etype}' missing from VALID_EVENT_TYPES"


# ---------------------------------------------------------------------------
# TestSseDepsImportRefactor — task-137 step-3
# ---------------------------------------------------------------------------

class TestSseDepsImportRefactor:
    """Verify sse.py uses get_db from deps and private copy is removed."""

    def test_no_private_get_db(self) -> None:
        """_get_db should be removed from sse module."""
        from autopilot.web.routes import sse
        assert not hasattr(sse, "_get_db"), (
            "_get_db should be removed in favor of get_db from deps"
        )

    def test_sse_get_db_is_deps_get_db(self) -> None:
        """sse.get_db should be the same object as deps.get_db."""
        from autopilot.web import deps
        from autopilot.web.routes import sse
        assert sse.get_db is deps.get_db, (
            "sse.get_db should be imported from deps"
        )
