"""Tests for review hub and narrative review routes + DB methods."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app
from tests.conftest import _seed_narrative

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> Iterator[CatalogDB]:
    """Create a CatalogDB with a temp file-backed database."""
    db_path = str(tmp_path / "catalog.db")
    _db = CatalogDB(db_path)
    yield _db  # type: ignore[misc]
    _db.close()


# ---------------------------------------------------------------------------
# TestUpdateNarrative — step-1
# ---------------------------------------------------------------------------

class TestUpdateNarrative:
    """Tests for CatalogDB.update_narrative(**kwargs)."""

    def test_update_title(self, db: CatalogDB) -> None:
        """update_narrative updates the title field."""
        _seed_narrative(db, "n-1", title="Old Title")
        db.update_narrative("n-1", title="New Title")
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "New Title"

    def test_update_description(self, db: CatalogDB) -> None:
        """update_narrative updates the description field."""
        _seed_narrative(db, "n-1", description="Old Desc")
        db.update_narrative("n-1", description="New Desc")
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["description"] == "New Desc"

    def test_update_proposed_duration_seconds(self, db: CatalogDB) -> None:
        """update_narrative updates proposed_duration_seconds."""
        _seed_narrative(db, "n-1", proposed_duration_seconds=60.0)
        db.update_narrative("n-1", proposed_duration_seconds=180.0)
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["proposed_duration_seconds"] == 180.0

    def test_leaves_unmentioned_fields_unchanged(self, db: CatalogDB) -> None:
        """update_narrative only updates explicitly provided fields."""
        _seed_narrative(db, "n-1", title="Keep Me", description="Change Me")
        db.update_narrative("n-1", description="Changed")
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "Keep Me"
        assert row["description"] == "Changed"

    def test_noop_on_empty_kwargs(self, db: CatalogDB) -> None:
        """update_narrative with no kwargs is a no-op (no error)."""
        _seed_narrative(db, "n-1", title="Same")
        db.update_narrative("n-1")
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "Same"

    def test_update_multiple_fields(self, db: CatalogDB) -> None:
        """update_narrative can update multiple fields at once."""
        _seed_narrative(db, "n-1", title="Old", description="Old", proposed_duration_seconds=60.0)
        db.update_narrative("n-1", title="New", description="New", proposed_duration_seconds=90.0)
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "New"
        assert row["description"] == "New"
        assert row["proposed_duration_seconds"] == 90.0


# ---------------------------------------------------------------------------
# TestUpdateNarrativeWhitelist — step-27
# ---------------------------------------------------------------------------

class TestUpdateNarrativeWhitelist:
    """Tests for column whitelist validation in update_narrative."""

    def test_rejects_disallowed_column(self, db: CatalogDB) -> None:
        """update_narrative raises ValueError for a column not in the whitelist."""
        _seed_narrative(db, "n-1")
        with pytest.raises(ValueError, match="rowid"):
            db.update_narrative("n-1", rowid=999)

    def test_rejects_sql_injection_key(self, db: CatalogDB) -> None:
        """update_narrative raises ValueError for SQL-injection column name."""
        _seed_narrative(db, "n-1")
        with pytest.raises(ValueError):
            db.update_narrative("n-1", **{"x = 1; DROP TABLE narratives; --": "bad"})

    def test_rejects_unknown_column(self, db: CatalogDB) -> None:
        """update_narrative raises ValueError for an arbitrary unknown column."""
        _seed_narrative(db, "n-1")
        with pytest.raises(ValueError, match="nonexistent_col"):
            db.update_narrative("n-1", nonexistent_col="value")

    def test_accepts_all_legitimate_columns(self, db: CatalogDB) -> None:
        """update_narrative accepts all legitimate mutable columns without error."""
        _seed_narrative(db, "n-1")
        # Each of these should succeed without raising
        db.update_narrative("n-1", title="T")
        db.update_narrative("n-1", description="D")
        db.update_narrative("n-1", proposed_duration_seconds=99.0)
        db.update_narrative("n-1", activity_cluster_ids_json='["c-3"]')
        db.update_narrative("n-1", arc_notes="notes")
        db.update_narrative("n-1", emotional_journey="joy")
        db.update_narrative("n-1", status="approved")
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "T"
        assert row["status"] == "approved"


# ---------------------------------------------------------------------------
# Web fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with seeded narratives."""
    db_path = str(tmp_path / "catalog.db")
    with CatalogDB(db_path) as _db:
        _db.init_default_gates()
        _seed_narrative(_db, "n-1", title="Morning Walk", status="proposed")
        _seed_narrative(_db, "n-2", title="Sunset Hike", status="approved")
        _seed_narrative(_db, "n-3", title="Beach Day", status="proposed")
        _db.insert_activity_cluster(
            "c-test-1",
            label="Morning Jog",
            description="Jogging in the park",
            clip_ids_json='["v1","v2"]',
        )
        _db.conn.commit()
    return create_app(db_path)


@pytest.fixture
def seeded_client(seeded_app: FastAPI) -> TestClient:
    """Create a TestClient for the seeded app."""
    return TestClient(seeded_app)


# ---------------------------------------------------------------------------
# TestReviewRouter — step-3
# ---------------------------------------------------------------------------

class TestReviewRouter:
    """Tests for review_router registration and GET /review."""

    def test_review_router_importable(self) -> None:
        """review_router is importable from autopilot.web.routes."""
        from autopilot.web.routes import review_router  # noqa: F401

    def test_get_review_returns_200(self, client: TestClient) -> None:
        """GET /review returns HTTP 200."""
        response = client.get("/review")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# TestListNarrativesAPI — step-5
# ---------------------------------------------------------------------------

class TestListNarrativesAPI:
    """Tests for GET /api/narratives endpoint."""

    def test_returns_json_list(self, seeded_client: TestClient) -> None:
        """GET /api/narratives returns a JSON list of narratives."""
        response = seeded_client.get("/api/narratives")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3

    def test_filter_by_status(self, seeded_client: TestClient) -> None:
        """GET /api/narratives?status=proposed returns only proposed narratives."""
        response = seeded_client.get("/api/narratives", params={"status": "proposed"})
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(n["status"] == "proposed" for n in data)

    def test_empty_list_when_no_narratives(self, client: TestClient) -> None:
        """GET /api/narratives returns empty list when no narratives exist."""
        response = client.get("/api/narratives")
        assert response.status_code == 200
        data = response.json()
        assert data == []


# ---------------------------------------------------------------------------
# TestGetNarrativeAPI — step-7
# ---------------------------------------------------------------------------

class TestGetNarrativeAPI:
    """Tests for GET /api/narratives/{id} endpoint."""

    def test_returns_narrative_dict(self, seeded_client: TestClient) -> None:
        """GET /api/narratives/n-1 returns the narrative as a dict."""
        response = seeded_client.get("/api/narratives/n-1")
        assert response.status_code == 200
        data = response.json()
        assert data["narrative_id"] == "n-1"
        assert data["title"] == "Morning Walk"

    def test_parses_cluster_ids(self, seeded_client: TestClient) -> None:
        """Response includes parsed activity_cluster_ids list."""
        response = seeded_client.get("/api/narratives/n-1")
        assert response.status_code == 200
        data = response.json()
        assert data["activity_cluster_ids"] == ["c-1", "c-2"]

    def test_404_for_nonexistent(self, seeded_client: TestClient) -> None:
        """GET /api/narratives/nonexistent returns 404."""
        response = seeded_client.get("/api/narratives/nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# TestApproveNarrativeAPI — step-9
# ---------------------------------------------------------------------------

class TestApproveNarrativeAPI:
    """Tests for POST /api/narratives/{id}/approve endpoint."""

    def test_approve_sets_status(self, seeded_client: TestClient) -> None:
        """POST /api/narratives/n-1/approve sets status to approved."""
        response = seeded_client.post("/api/narratives/n-1/approve")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["narrative_id"] == "n-1"

    def test_approve_404_for_missing(self, seeded_client: TestClient) -> None:
        """POST /api/narratives/nonexistent/approve returns 404."""
        response = seeded_client.post("/api/narratives/nonexistent/approve")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# TestRejectNarrativeAPI — step-11
# ---------------------------------------------------------------------------

class TestRejectNarrativeAPI:
    """Tests for POST /api/narratives/{id}/reject endpoint."""

    def test_reject_sets_status(self, seeded_client: TestClient) -> None:
        """POST /api/narratives/n-1/reject sets status to rejected."""
        response = seeded_client.post("/api/narratives/n-1/reject")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "rejected"
        assert data["narrative_id"] == "n-1"

    def test_reject_404_for_missing(self, seeded_client: TestClient) -> None:
        """POST /api/narratives/nonexistent/reject returns 404."""
        response = seeded_client.post("/api/narratives/nonexistent/reject")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# TestUpdateNarrativeAPI — step-13
# ---------------------------------------------------------------------------

class TestUpdateNarrativeAPI:
    """Tests for PUT /api/narratives/{id} endpoint."""

    def test_update_title_and_description(self, seeded_client: TestClient) -> None:
        """PUT /api/narratives/n-1 updates title and description."""
        response = seeded_client.put(
            "/api/narratives/n-1",
            json={"title": "Evening Walk", "description": "A sunset stroll"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Evening Walk"
        assert data["description"] == "A sunset stroll"

    def test_update_duration(self, seeded_client: TestClient) -> None:
        """PUT /api/narratives/n-1 updates proposed_duration_seconds."""
        response = seeded_client.put(
            "/api/narratives/n-1",
            json={"proposed_duration_seconds": 300.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["proposed_duration_seconds"] == 300.0

    def test_rejects_unknown_fields(self, seeded_client: TestClient) -> None:
        """PUT /api/narratives/n-1 with unknown field returns 422."""
        response = seeded_client.put(
            "/api/narratives/n-1",
            json={"unknown_field": "bad"},
        )
        assert response.status_code == 422

    def test_404_for_missing(self, seeded_client: TestClient) -> None:
        """PUT /api/narratives/nonexistent returns 404."""
        response = seeded_client.put(
            "/api/narratives/nonexistent",
            json={"title": "Does not exist"},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# TestBulkApproveAPI — step-15
# ---------------------------------------------------------------------------

class TestBulkApproveAPI:
    """Tests for POST /api/narratives/bulk-approve endpoint."""

    def test_bulk_approve_multiple(self, seeded_client: TestClient) -> None:
        """POST /api/narratives/bulk-approve approves multiple narratives."""
        response = seeded_client.post(
            "/api/narratives/bulk-approve",
            json={"ids": ["n-1", "n-3"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["approved"] == 2

    def test_bulk_approve_empty_list(self, seeded_client: TestClient) -> None:
        """POST /api/narratives/bulk-approve with empty list returns 0."""
        response = seeded_client.post(
            "/api/narratives/bulk-approve",
            json={"ids": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["approved"] == 0


# ---------------------------------------------------------------------------
# TestBulkApproveCorrectness — step-29
# ---------------------------------------------------------------------------

class TestBulkApproveCorrectness:
    """Tests for bulk-approve persistence and accurate row count."""

    def test_bulk_approve_persists_status(self, seeded_client: TestClient) -> None:
        """After bulk-approve, approved narratives are persisted in the DB."""
        # n-1 and n-3 are "proposed" in seeded_app
        response = seeded_client.post(
            "/api/narratives/bulk-approve",
            json={"ids": ["n-1", "n-3"]},
        )
        assert response.status_code == 200
        assert response.json()["approved"] == 2
        # Verify DB persistence via GET
        approved = seeded_client.get(
            "/api/narratives", params={"status": "approved"},
        )
        approved_ids = {n["narrative_id"] for n in approved.json()}
        assert "n-1" in approved_ids
        assert "n-3" in approved_ids

    def test_bulk_approve_nonexistent_ids(self, seeded_client: TestClient) -> None:
        """Bulk-approve with nonexistent IDs returns approved: 0."""
        response = seeded_client.post(
            "/api/narratives/bulk-approve",
            json={"ids": ["does-not-exist-1", "does-not-exist-2"]},
        )
        assert response.status_code == 200
        assert response.json()["approved"] == 0

    def test_bulk_approve_mixed_existing_nonexistent(
        self, seeded_client: TestClient,
    ) -> None:
        """Bulk-approve with mix of valid and nonexistent IDs counts only valid."""
        response = seeded_client.post(
            "/api/narratives/bulk-approve",
            json={"ids": ["n-1", "does-not-exist"]},
        )
        assert response.status_code == 200
        # Only n-1 exists, so approved should be 1
        assert response.json()["approved"] == 1


# ---------------------------------------------------------------------------
# Review Hub page fixtures — step-17
# ---------------------------------------------------------------------------

@pytest.fixture
def waiting_gate_app(tmp_path: Path) -> FastAPI:
    """App with narrate gate set to 'waiting' and proposed narratives."""
    db_path = str(tmp_path / "catalog.db")
    with CatalogDB(db_path) as _db:
        _db.init_default_gates()
        _db.update_gate("narrate", status="waiting")
        _db.conn.commit()
        _seed_narrative(_db, "n-1", title="Morning Walk", status="proposed")
        _seed_narrative(_db, "n-2", title="Sunset Hike", status="proposed")
    return create_app(db_path)


@pytest.fixture
def waiting_gate_client(waiting_gate_app: FastAPI) -> TestClient:
    return TestClient(waiting_gate_app)


# ---------------------------------------------------------------------------
# TestReviewHub — step-17
# ---------------------------------------------------------------------------

class TestReviewHub:
    """Tests for GET /review hub page showing pending gates."""

    def test_hub_returns_200_html(self, client: TestClient) -> None:
        """GET /review returns 200 with HTML content."""
        response = client.get("/review")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_hub_empty_state(self, client: TestClient) -> None:
        """Hub shows 'No pending reviews' when no gates are waiting."""
        response = client.get("/review")
        assert "No pending reviews" in response.text

    def test_hub_shows_waiting_gate(self, waiting_gate_client: TestClient) -> None:
        """Hub shows gate card when narrate gate is waiting."""
        response = waiting_gate_client.get("/review")
        assert response.status_code == 200
        assert "narrate" in response.text.lower()

    def test_hub_links_to_narrative_review(
        self, waiting_gate_client: TestClient,
    ) -> None:
        """Hub has link to /review/narratives when narrate gate waiting."""
        response = waiting_gate_client.get("/review")
        assert "/review/narratives" in response.text

    def test_hub_shows_proposed_count(
        self, waiting_gate_client: TestClient,
    ) -> None:
        """Hub shows count of proposed narratives for narrate gate."""
        response = waiting_gate_client.get("/review")
        # Should show "2" somewhere for the 2 proposed narratives
        assert "2" in response.text


# ---------------------------------------------------------------------------
# TestNarrativesPage — step-19
# ---------------------------------------------------------------------------

class TestNarrativesPage:
    """Tests for GET /review/narratives page."""

    def test_page_returns_200_html(self, seeded_client: TestClient) -> None:
        """GET /review/narratives returns 200 with HTML content."""
        response = seeded_client.get("/review/narratives")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_page_contains_narrative_titles(
        self, seeded_client: TestClient,
    ) -> None:
        """Page shows narrative titles."""
        response = seeded_client.get("/review/narratives")
        assert "Morning Walk" in response.text
        assert "Sunset Hike" in response.text
        assert "Beach Day" in response.text

    def test_page_shows_approve_reject_buttons(
        self, seeded_client: TestClient,
    ) -> None:
        """Page has approve and reject buttons for proposed narratives."""
        response = seeded_client.get("/review/narratives")
        assert "approve" in response.text.lower()
        assert "reject" in response.text.lower()

    def test_page_shows_status_badges(
        self, seeded_client: TestClient,
    ) -> None:
        """Page shows status badges (proposed, approved)."""
        response = seeded_client.get("/review/narratives")
        assert "proposed" in response.text.lower()
        assert "approved" in response.text.lower()


# ---------------------------------------------------------------------------
# TestNarrativePartialHTMX — step-21
# ---------------------------------------------------------------------------

class TestNarrativePartialHTMX:
    """Tests for HTMX partial responses from approve/reject endpoints."""

    def test_approve_htmx_returns_html(self, seeded_client: TestClient) -> None:
        """POST /api/narratives/n-1/approve with HX-Request returns HTML."""
        response = seeded_client.post(
            "/api/narratives/n-1/approve",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # HTML should contain the status badge, not JSON
        assert "approved" in response.text.lower()
        assert "narrative-n-1" in response.text

    def test_approve_htmx_contains_status_badge(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX approve response contains updated status badge."""
        response = seeded_client.post(
            "/api/narratives/n-1/approve",
            headers={"HX-Request": "true"},
        )
        # Should have green badge for approved status
        assert "bg-green-900" in response.text

    def test_reject_htmx_returns_html(self, seeded_client: TestClient) -> None:
        """POST /api/narratives/n-1/reject with HX-Request returns HTML."""
        response = seeded_client.post(
            "/api/narratives/n-1/reject",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "rejected" in response.text.lower()

    def test_reject_htmx_contains_red_badge(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX reject response contains red status badge."""
        response = seeded_client.post(
            "/api/narratives/n-1/reject",
            headers={"HX-Request": "true"},
        )
        assert "bg-red-900" in response.text


# ---------------------------------------------------------------------------
# Gate Approval fixtures — step-23
# ---------------------------------------------------------------------------

@pytest.fixture
def all_decided_app(tmp_path: Path) -> FastAPI:
    """App with narrate gate waiting and all narratives decided (none proposed)."""
    db_path = str(tmp_path / "catalog.db")
    with CatalogDB(db_path) as _db:
        _db.init_default_gates()
        _db.update_gate("narrate", status="waiting")
        _db.conn.commit()
        _seed_narrative(_db, "n-1", title="Morning Walk", status="approved")
        _seed_narrative(_db, "n-2", title="Sunset Hike", status="rejected")
    return create_app(db_path)


@pytest.fixture
def all_decided_client(all_decided_app: FastAPI) -> TestClient:
    return TestClient(all_decided_app)


# ---------------------------------------------------------------------------
# TestGateApprovalIntegration — step-23
# ---------------------------------------------------------------------------

class TestGateApprovalIntegration:
    """Tests for Approve Gate button on narrative review page."""

    def test_no_approve_gate_when_proposed_remain(
        self, waiting_gate_client: TestClient,
    ) -> None:
        """Page does NOT show Approve Gate button when proposed narratives remain."""
        response = waiting_gate_client.get("/review/narratives")
        assert "Approve Gate" not in response.text

    def test_shows_approve_gate_when_all_decided(
        self, all_decided_client: TestClient,
    ) -> None:
        """Page shows Approve Gate button when no proposed narratives remain."""
        response = all_decided_client.get("/review/narratives")
        assert "Approve Gate" in response.text

    def test_approve_gate_targets_gates_api(
        self, all_decided_client: TestClient,
    ) -> None:
        """Approve Gate button targets POST /api/gates/narrate/approve."""
        response = all_decided_client.get("/review/narratives")
        assert "/api/gates/narrate/approve" in response.text


# ---------------------------------------------------------------------------
# TestUpdateNarrativeHTMX — step-25
# ---------------------------------------------------------------------------

class TestUpdateNarrativeHTMX:
    """Tests for PUT /api/narratives/{id} with HTMX returning HTML partial."""

    def test_put_htmx_returns_html(self, seeded_client: TestClient) -> None:
        """PUT /api/narratives/n-1 with HX-Request returns HTML partial."""
        response = seeded_client.put(
            "/api/narratives/n-1",
            json={"title": "Updated Title"},
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_put_htmx_contains_updated_title(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX PUT response contains the updated title in HTML."""
        response = seeded_client.put(
            "/api/narratives/n-1",
            json={"title": "Updated Title"},
            headers={"HX-Request": "true"},
        )
        assert "Updated Title" in response.text

    def test_put_htmx_contains_narrative_id(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX PUT response contains the narrative div ID for swap."""
        response = seeded_client.put(
            "/api/narratives/n-1",
            json={"title": "Updated Title"},
            headers={"HX-Request": "true"},
        )
        assert "narrative-n-1" in response.text

    def test_put_without_htmx_returns_json(
        self, seeded_client: TestClient,
    ) -> None:
        """PUT /api/narratives/n-1 without HX-Request returns JSON."""
        response = seeded_client.put(
            "/api/narratives/n-1",
            json={"title": "Updated Title"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"


# ---------------------------------------------------------------------------
# TestSafeJsonList — task-72 step-1
# ---------------------------------------------------------------------------

class TestSafeJsonList:
    """Tests for _safe_json_list helper."""

    def test_valid_json_list(self) -> None:
        """Valid JSON list string returns parsed list."""
        from autopilot.web.routes.review import _safe_json_list

        assert _safe_json_list('["a", "b", "c"]') == ["a", "b", "c"]

    def test_none_returns_empty(self) -> None:
        """None input returns empty list."""
        from autopilot.web.routes.review import _safe_json_list

        assert _safe_json_list(None) == []

    def test_empty_string_returns_empty(self) -> None:
        """Empty string input returns empty list."""
        from autopilot.web.routes.review import _safe_json_list

        assert _safe_json_list("") == []

    def test_malformed_json_returns_empty(self) -> None:
        """Malformed JSON returns empty list instead of raising."""
        from autopilot.web.routes.review import _safe_json_list

        assert _safe_json_list("{not-json") == []

    def test_json_string_returns_empty(self) -> None:
        """JSON string (not a list) returns empty list."""
        from autopilot.web.routes.review import _safe_json_list

        assert _safe_json_list('"hello"') == []

    def test_json_integer_returns_empty(self) -> None:
        """JSON integer (not a list) returns empty list."""
        from autopilot.web.routes.review import _safe_json_list

        assert _safe_json_list("42") == []

    def test_json_object_returns_empty(self) -> None:
        """JSON object (not a list) returns empty list."""
        from autopilot.web.routes.review import _safe_json_list

        assert _safe_json_list('{"a": 1}') == []


# ---------------------------------------------------------------------------
# TestParseNarrativeSafety — task-72 step-3
# ---------------------------------------------------------------------------

class TestParseNarrativeSafety:
    """Tests for _parse_narrative handling malformed JSON gracefully."""

    def test_malformed_json_returns_empty_cluster_ids(self) -> None:
        """_parse_narrative with malformed JSON returns empty list, not crash."""
        from autopilot.web.routes.review import _parse_narrative

        result = _parse_narrative({"activity_cluster_ids_json": "{not-json"})
        assert result["activity_cluster_ids"] == []

    def test_non_list_json_returns_empty_cluster_ids(self) -> None:
        """_parse_narrative with non-list JSON returns empty list."""
        from autopilot.web.routes.review import _parse_narrative

        result = _parse_narrative({"activity_cluster_ids_json": '"just-a-string"'})
        assert result["activity_cluster_ids"] == []

    def test_none_json_returns_empty_cluster_ids(self) -> None:
        """_parse_narrative with None JSON value returns empty list."""
        from autopilot.web.routes.review import _parse_narrative

        result = _parse_narrative({"activity_cluster_ids_json": None})
        assert result["activity_cluster_ids"] == []

    def test_valid_json_list_still_works(self) -> None:
        """_parse_narrative with valid JSON list still parses correctly."""
        from autopilot.web.routes.review import _parse_narrative

        result = _parse_narrative({"activity_cluster_ids_json": '["c-1", "c-2"]'})
        assert result["activity_cluster_ids"] == ["c-1", "c-2"]


# ---------------------------------------------------------------------------
# TestNarrativeAPIMalformedJSON — task-72 step-5
# ---------------------------------------------------------------------------

@pytest.fixture
def malformed_json_app(tmp_path: Path) -> FastAPI:
    """App with narratives that have malformed/non-list JSON in cluster IDs."""
    db_path = str(tmp_path / "catalog.db")
    with CatalogDB(db_path) as _db:
        _db.init_default_gates()
        _seed_narrative(
            _db, "n-bad-json",
            title="Bad JSON",
            activity_cluster_ids_json="{bad",
        )
        _seed_narrative(
            _db, "n-not-list",
            title="Not A List",
            activity_cluster_ids_json='"just-a-string"',
        )
    return create_app(db_path)


@pytest.fixture
def malformed_json_client(malformed_json_app: FastAPI) -> TestClient:
    return TestClient(malformed_json_app)


class TestNarrativeAPIMalformedJSON:
    """API-level tests for graceful handling of bad JSON in cluster IDs."""

    def test_malformed_json_returns_200_with_empty_cluster_ids(
        self, malformed_json_client: TestClient,
    ) -> None:
        """GET /api/narratives/n-bad-json returns 200 with empty cluster IDs."""
        response = malformed_json_client.get("/api/narratives/n-bad-json")
        assert response.status_code == 200
        data = response.json()
        assert data["activity_cluster_ids"] == []

    def test_non_list_json_returns_200_with_empty_cluster_ids(
        self, malformed_json_client: TestClient,
    ) -> None:
        """GET /api/narratives/n-not-list returns 200 with empty cluster IDs."""
        response = malformed_json_client.get("/api/narratives/n-not-list")
        assert response.status_code == 200
        data = response.json()
        assert data["activity_cluster_ids"] == []


# ---------------------------------------------------------------------------
# TestRenderPartialRefactor — task-74 step-3
# ---------------------------------------------------------------------------

class TestRenderPartialRefactor:
    """Verify review.py uses render_partial from deps and _render_narrative_partial is removed."""

    def test_no_private_render_narrative_partial(self) -> None:
        """_render_narrative_partial should be removed from review module."""
        from autopilot.web.routes import review
        assert not hasattr(review, "_render_narrative_partial"), (
            "_render_narrative_partial should be removed in favor of render_partial"
        )

    def test_no_private_render_cluster_partial(self) -> None:
        """_render_cluster_partial should be removed from review module."""
        from autopilot.web.routes import review
        assert not hasattr(review, "_render_cluster_partial"), (
            "_render_cluster_partial should be removed in favor of render_partial"
        )

    def test_review_imports_render_partial(self) -> None:
        """review module should import render_partial from deps."""
        import inspect

        from autopilot.web.routes import review as review_mod
        source = inspect.getsource(review_mod)
        assert "render_partial" in source, (
            "review.py should use render_partial from deps"
        )


# ---------------------------------------------------------------------------
# TestClusterRelabelHTMX — task-74 step-11
# ---------------------------------------------------------------------------

class TestClusterRelabelHTMX:
    """Tests for POST /api/clusters/{id}/relabel with HTMX returning HTML partial."""

    def test_relabel_htmx_returns_html(self, seeded_client: TestClient) -> None:
        """POST /api/clusters/c-test-1/relabel with HX-Request returns HTML."""
        response = seeded_client.post(
            "/api/clusters/c-test-1/relabel",
            json={"label": "Updated Label"},
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "cluster-c-test-1" in response.text
        assert "Updated Label" in response.text

    def test_relabel_htmx_preserves_clip_count(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX relabel response preserves the clip count from parsed cluster."""
        response = seeded_client.post(
            "/api/clusters/c-test-1/relabel",
            json={"label": "Updated Label"},
            headers={"HX-Request": "true"},
        )
        assert "2 clips" in response.text


# ---------------------------------------------------------------------------
# TestClusterExcludeHTMX — task-74 step-11
# ---------------------------------------------------------------------------

class TestClusterExcludeHTMX:
    """Tests for POST /api/clusters/{id}/exclude with HTMX returning HTML partial."""

    def test_exclude_htmx_returns_html(self, seeded_client: TestClient) -> None:
        """POST /api/clusters/c-test-1/exclude with HX-Request returns HTML."""
        response = seeded_client.post(
            "/api/clusters/c-test-1/exclude",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "cluster-c-test-1" in response.text

    def test_exclude_htmx_contains_excluded_badge(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX exclude response contains the excluded badge styling."""
        response = seeded_client.post(
            "/api/clusters/c-test-1/exclude",
            headers={"HX-Request": "true"},
        )
        assert "bg-red-900" in response.text


# ---------------------------------------------------------------------------
# TestGetNarrativeEditHTMX — task-100 step-1
# ---------------------------------------------------------------------------

class TestGetNarrativeEditHTMX:
    """Tests for GET /api/narratives/{id}?edit=1 with HTMX returning edit form."""

    def test_get_htmx_edit_returns_html(self, seeded_client: TestClient) -> None:
        """GET /api/narratives/n-1?edit=1 with HX-Request returns text/html."""
        response = seeded_client.get(
            "/api/narratives/n-1?edit=1",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_get_htmx_edit_contains_input_fields(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX edit response contains input fields for title, description, duration."""
        response = seeded_client.get(
            "/api/narratives/n-1?edit=1",
            headers={"HX-Request": "true"},
        )
        assert 'name="title"' in response.text
        assert 'name="description"' in response.text
        assert 'name="proposed_duration_seconds"' in response.text

    def test_get_htmx_edit_prepopulates_values(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX edit response prepopulates fields with current narrative values."""
        response = seeded_client.get(
            "/api/narratives/n-1?edit=1",
            headers={"HX-Request": "true"},
        )
        assert "Morning Walk" in response.text
        assert "A walk in the park" in response.text

    def test_get_htmx_edit_has_save_button(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX edit response contains Save button with hx-put to update endpoint."""
        response = seeded_client.get(
            "/api/narratives/n-1?edit=1",
            headers={"HX-Request": "true"},
        )
        assert 'hx-put="/api/narratives/n-1"' in response.text

    def test_get_htmx_edit_has_cancel_button(
        self, seeded_client: TestClient,
    ) -> None:
        """HTMX edit response contains Cancel button that swaps back to read-only card."""
        response = seeded_client.get(
            "/api/narratives/n-1?edit=1",
            headers={"HX-Request": "true"},
        )
        # Cancel button should GET the narrative without ?edit=1 to get the card back
        assert 'hx-get="/api/narratives/n-1"' in response.text
        assert "Cancel" in response.text

    def test_get_htmx_no_edit_returns_card(
        self, seeded_client: TestClient,
    ) -> None:
        """GET /api/narratives/n-1 with HX-Request (no ?edit=1) returns read-only card."""
        response = seeded_client.get(
            "/api/narratives/n-1",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Morning Walk" in response.text
        # Read-only card has Approve button, not input fields
        assert "Approve" in response.text
        assert 'name="title"' not in response.text

    def test_get_no_htmx_returns_json(self, seeded_client: TestClient) -> None:
        """GET /api/narratives/n-1 without HX-Request returns JSON."""
        response = seeded_client.get("/api/narratives/n-1")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        data = response.json()
        assert data["title"] == "Morning Walk"

    def test_narrative_card_edit_button_uses_edit_param(
        self, seeded_client: TestClient,
    ) -> None:
        """The Edit button in the read-only card includes ?edit=1 in its hx-get URL."""
        response = seeded_client.get(
            "/api/narratives/n-1",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "edit=1" in response.text

    def test_get_htmx_edit_not_found_returns_404(
        self, seeded_client: TestClient,
    ) -> None:
        """GET /api/narratives/nonexistent?edit=1 with HX-Request returns 404."""
        response = seeded_client.get(
            "/api/narratives/nonexistent?edit=1",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Zero-duration fixtures — task-167
# ---------------------------------------------------------------------------

@pytest.fixture
def zero_duration_app(tmp_path: Path) -> FastAPI:
    """App with narratives seeded with proposed_duration_seconds=0 and None."""
    db_path = str(tmp_path / "catalog.db")
    with CatalogDB(db_path) as _db:
        _db.init_default_gates()
        _seed_narrative(
            _db, "n-zero",
            title="Zero Duration",
            proposed_duration_seconds=0,
        )
        _seed_narrative(
            _db, "n-none",
            title="No Duration",
            proposed_duration_seconds=None,
        )
    return create_app(db_path)


@pytest.fixture
def zero_duration_client(zero_duration_app: FastAPI) -> TestClient:
    """TestClient for the zero-duration app."""
    return TestClient(zero_duration_app)


# ---------------------------------------------------------------------------
# TestEditFormZeroDuration — task-167 step-1
# ---------------------------------------------------------------------------

class TestEditFormZeroDuration:
    """Tests for edit form rendering of zero and None duration values."""

    def test_edit_form_renders_zero_duration(
        self, zero_duration_client: TestClient,
    ) -> None:
        """Edit form for narrative with proposed_duration_seconds=0 shows value="0"."""
        response = zero_duration_client.get(
            "/api/narratives/n-zero?edit=1",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        # DB stores REAL, so 0 renders as "0" or "0.0" — either is acceptable
        assert 'value="0"' in response.text or 'value="0.0"' in response.text

    def test_edit_form_renders_empty_for_none_duration(
        self, zero_duration_client: TestClient,
    ) -> None:
        """Edit form for narrative with proposed_duration_seconds=None shows empty value."""
        response = zero_duration_client.get(
            "/api/narratives/n-none?edit=1",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        # The input value should be empty, not the string 'None'
        assert 'value="None"' not in response.text
        # Duration input should have an empty value attribute
        assert 'name="proposed_duration_seconds"' in response.text

    def test_edit_form_js_does_not_use_falsy_or_null(
        self, zero_duration_client: TestClient,
    ) -> None:
        """hx-vals JS uses explicit NaN/empty check, not '|| null' for duration."""
        response = zero_duration_client.get(
            "/api/narratives/n-zero?edit=1",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        # The hx-vals should NOT use the falsy '|| null' pattern for duration
        assert "|| null" not in response.text
        # Should use explicit NaN check instead
        assert "isNaN" in response.text
