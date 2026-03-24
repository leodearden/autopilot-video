"""Tests for review hub and narrative review routes + DB methods."""

from __future__ import annotations

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
def db(tmp_path: Path) -> CatalogDB:
    """Create a CatalogDB with a temp file-backed database."""
    db_path = str(tmp_path / "catalog.db")
    _db = CatalogDB(db_path)
    yield _db  # type: ignore[misc]
    _db.close()


def _seed_narrative(db: CatalogDB, narrative_id: str = "n-1", **overrides: object) -> None:
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
    db.conn.commit()


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
# Web fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with a file-backed CatalogDB via tmp_path."""
    db_path = str(tmp_path / "catalog.db")
    with CatalogDB(db_path) as _db:
        _db.init_default_gates()
    return create_app(db_path)


@pytest.fixture
def seeded_app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with seeded narratives."""
    db_path = str(tmp_path / "catalog.db")
    with CatalogDB(db_path) as _db:
        _db.init_default_gates()
        _seed_narrative(_db, "n-1", title="Morning Walk", status="proposed")
        _seed_narrative(_db, "n-2", title="Sunset Hike", status="approved")
        _seed_narrative(_db, "n-3", title="Beach Day", status="proposed")
    return create_app(db_path)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a TestClient for the app."""
    return TestClient(app)


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
