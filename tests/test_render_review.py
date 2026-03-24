"""Tests for render review and upload status routes + DB methods."""

from __future__ import annotations

import json
from collections.abc import Iterator
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
def db(tmp_path: Path) -> Iterator[CatalogDB]:
    """Create a CatalogDB with a temp file-backed database."""
    db_path = str(tmp_path / "catalog.db")
    _db = CatalogDB(db_path)
    yield _db  # type: ignore[misc]
    _db.close()


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with a temp database and default gates."""
    db_path = str(tmp_path / "app.db")
    with CatalogDB(db_path) as _db:
        _db.init_default_gates()
    return create_app(db_path=db_path)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


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
    db.conn.commit()


def _seed_edit_plan(
    db: CatalogDB,
    narrative_id: str = "n-1",
    *,
    seed_narrative: bool = True,
    **overrides: object,
) -> None:
    """Insert an edit plan with sensible defaults.

    Also inserts a narrative by default (set seed_narrative=False to skip).
    """
    if seed_narrative:
        _seed_narrative(db, narrative_id)
    defaults: dict[str, object] = {
        "edl_json": '{"cuts": []}',
        "otio_path": "/tmp/edit.otio",
        "validation_json": json.dumps({
            "resolution": "1920x1080",
            "duration_seconds": 120.5,
            "codec": "h264",
            "passes": True,
        }),
        "render_path": "/tmp/render.mp4",
    }
    defaults.update(overrides)
    db.upsert_edit_plan(narrative_id, **defaults)  # type: ignore[arg-type]
    db.conn.commit()


def _seed_upload(
    db: CatalogDB,
    narrative_id: str = "n-1",
    *,
    seed_narrative: bool = True,
    **overrides: object,
) -> None:
    """Insert an upload record with sensible defaults.

    Also inserts a narrative by default (set seed_narrative=False to skip).
    """
    if seed_narrative:
        _seed_narrative(db, narrative_id)
    defaults: dict[str, object] = {
        "youtube_video_id": "abc123",
        "youtube_url": "https://youtube.com/watch?v=abc123",
        "uploaded_at": "2025-01-15T10:30:00",
        "privacy_status": "unlisted",
    }
    defaults.update(overrides)
    db.insert_upload(narrative_id, **defaults)  # type: ignore[arg-type]
    db.conn.commit()


# ---------------------------------------------------------------------------
# TestListEditPlans — step-1
# ---------------------------------------------------------------------------


class TestListEditPlans:
    """Tests for CatalogDB.list_edit_plans()."""

    def test_returns_empty_list_when_none_exist(self, db: CatalogDB) -> None:
        """list_edit_plans returns an empty list when no edit plans exist."""
        result = db.list_edit_plans()
        assert result == []

    def test_returns_all_edit_plans_with_narrative_title(
        self, db: CatalogDB,
    ) -> None:
        """list_edit_plans returns all edit plans joined with narrative title."""
        _seed_edit_plan(db, "n-1")
        _seed_narrative(db, "n-2", title="Sunset Hike")
        _seed_edit_plan(db, "n-2", seed_narrative=False)
        result = db.list_edit_plans()
        assert len(result) == 2
        ids = {r["narrative_id"] for r in result}
        assert ids == {"n-1", "n-2"}

    def test_includes_all_columns(self, db: CatalogDB) -> None:
        """list_edit_plans includes all edit_plan columns plus narrative_title."""
        _seed_edit_plan(db, "n-1")
        result = db.list_edit_plans()
        assert len(result) == 1
        row = result[0]
        assert row["narrative_id"] == "n-1"
        assert row["edl_json"] == '{"cuts": []}'
        assert row["otio_path"] == "/tmp/edit.otio"
        assert row["validation_json"] is not None
        assert row["render_path"] == "/tmp/render.mp4"
        assert row["narrative_title"] == "Morning Walk"

    def test_narrative_title_is_none_when_narrative_deleted(
        self, db: CatalogDB,
    ) -> None:
        """narrative_title is None when narrative is missing (LEFT JOIN)."""
        _seed_edit_plan(db, "n-1")
        # Temporarily disable FK checks to delete the narrative
        db.conn.execute("PRAGMA foreign_keys = OFF")
        db.conn.execute("DELETE FROM narratives WHERE narrative_id = 'n-1'")
        db.conn.commit()
        db.conn.execute("PRAGMA foreign_keys = ON")
        result = db.list_edit_plans()
        assert len(result) == 1
        assert result[0]["narrative_title"] is None


# ---------------------------------------------------------------------------
# TestListUploads — step-3
# ---------------------------------------------------------------------------


class TestListUploads:
    """Tests for CatalogDB.list_uploads()."""

    def test_returns_empty_list_when_none_exist(self, db: CatalogDB) -> None:
        """list_uploads returns an empty list when no uploads exist."""
        result = db.list_uploads()
        assert result == []

    def test_returns_all_uploads_with_narrative_title(
        self, db: CatalogDB,
    ) -> None:
        """list_uploads returns all uploads joined with narrative title."""
        _seed_upload(db, "n-1")
        _seed_narrative(db, "n-2", title="Sunset Hike")
        _seed_upload(db, "n-2", seed_narrative=False, youtube_video_id="xyz789")
        result = db.list_uploads()
        assert len(result) == 2
        ids = {r["narrative_id"] for r in result}
        assert ids == {"n-1", "n-2"}

    def test_includes_all_columns(self, db: CatalogDB) -> None:
        """list_uploads includes all upload columns plus narrative_title."""
        _seed_upload(db, "n-1")
        result = db.list_uploads()
        assert len(result) == 1
        row = result[0]
        assert row["narrative_id"] == "n-1"
        assert row["youtube_video_id"] == "abc123"
        assert row["youtube_url"] == "https://youtube.com/watch?v=abc123"
        assert row["uploaded_at"] == "2025-01-15T10:30:00"
        assert row["privacy_status"] == "unlisted"
        assert row["narrative_title"] == "Morning Walk"

    def test_ordered_by_uploaded_at_desc(self, db: CatalogDB) -> None:
        """list_uploads returns uploads ordered by uploaded_at descending."""
        _seed_upload(db, "n-1", uploaded_at="2025-01-10T08:00:00")
        _seed_narrative(db, "n-2", title="Later Upload")
        _seed_upload(
            db, "n-2", seed_narrative=False,
            uploaded_at="2025-01-15T10:00:00", youtube_video_id="def456",
        )
        result = db.list_uploads()
        assert result[0]["narrative_id"] == "n-2"  # later first
        assert result[1]["narrative_id"] == "n-1"

    def test_narrative_title_is_none_when_narrative_deleted(
        self, db: CatalogDB,
    ) -> None:
        """narrative_title is None when narrative is missing (LEFT JOIN)."""
        _seed_upload(db, "n-1")
        db.conn.execute("PRAGMA foreign_keys = OFF")
        db.conn.execute("DELETE FROM narratives WHERE narrative_id = 'n-1'")
        db.conn.commit()
        db.conn.execute("PRAGMA foreign_keys = ON")
        result = db.list_uploads()
        assert len(result) == 1
        assert result[0]["narrative_title"] is None


# ---------------------------------------------------------------------------
# TestApiGetRender — step-5
# ---------------------------------------------------------------------------


class TestApiGetRender:
    """Tests for GET /api/renders/{narrative_id}."""

    def test_returns_render_data(self, tmp_path: Path) -> None:
        """Returns JSON with edit plan data and narrative title."""
        db_path = str(tmp_path / "app.db")
        with CatalogDB(db_path) as _db:
            _db.init_default_gates()
            _seed_narrative(_db, "n-1", title="Morning Walk")
            _seed_edit_plan(_db, "n-1", seed_narrative=False)
        app = create_app(db_path=db_path)
        client = TestClient(app)
        resp = client.get("/api/renders/n-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["narrative_id"] == "n-1"
        assert data["render_path"] == "/tmp/render.mp4"
        assert data["narrative_title"] == "Morning Walk"
        # validation should be parsed from JSON string to dict
        assert isinstance(data["validation"], dict)
        assert data["validation"]["passes"] is True

    def test_returns_404_for_nonexistent(self, client: TestClient) -> None:
        """Returns 404 for a narrative_id with no edit plan."""
        resp = client.get("/api/renders/no-such-id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestApiStreamVideo — step-7
# ---------------------------------------------------------------------------


class TestApiStreamVideo:
    """Tests for GET /api/renders/{narrative_id}/video."""

    def test_returns_video_when_file_exists(self, tmp_path: Path) -> None:
        """Returns 200 with video/mp4 content-type when render file exists."""
        video_file = tmp_path / "render.mp4"
        video_file.write_bytes(b"\x00" * 1024)  # mock video content
        db_path = str(tmp_path / "app.db")
        with CatalogDB(db_path) as _db:
            _db.init_default_gates()
            _seed_narrative(_db, "n-1")
            _seed_edit_plan(
                _db, "n-1", seed_narrative=False,
                render_path=str(video_file),
            )
        app = create_app(db_path=db_path)
        client = TestClient(app)
        resp = client.get("/api/renders/n-1/video")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"

    def test_returns_404_when_no_edit_plan(self, client: TestClient) -> None:
        """Returns 404 when no edit plan exists for the narrative."""
        resp = client.get("/api/renders/no-such-id/video")
        assert resp.status_code == 404

    def test_returns_404_when_render_path_is_null(
        self, tmp_path: Path,
    ) -> None:
        """Returns 404 when render_path is null (render not done)."""
        db_path = str(tmp_path / "app.db")
        with CatalogDB(db_path) as _db:
            _db.init_default_gates()
            _seed_narrative(_db, "n-1")
            _seed_edit_plan(
                _db, "n-1", seed_narrative=False, render_path=None,
            )
        app = create_app(db_path=db_path)
        client = TestClient(app)
        resp = client.get("/api/renders/n-1/video")
        assert resp.status_code == 404

    def test_returns_404_when_file_missing(self, tmp_path: Path) -> None:
        """Returns 404 when render_path points to a nonexistent file."""
        db_path = str(tmp_path / "app.db")
        with CatalogDB(db_path) as _db:
            _db.init_default_gates()
            _seed_narrative(_db, "n-1")
            _seed_edit_plan(
                _db, "n-1", seed_narrative=False,
                render_path="/nonexistent/video.mp4",
            )
        app = create_app(db_path=db_path)
        client = TestClient(app)
        resp = client.get("/api/renders/n-1/video")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestApiListUploads — step-9
# ---------------------------------------------------------------------------


class TestApiListUploads:
    """Tests for GET /api/uploads."""

    def test_returns_empty_list(self, client: TestClient) -> None:
        """Returns empty JSON list when no uploads exist."""
        resp = client.get("/api/uploads")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_uploads_with_narrative_title(
        self, tmp_path: Path,
    ) -> None:
        """Returns JSON list of uploads with narrative_title joined."""
        db_path = str(tmp_path / "app.db")
        with CatalogDB(db_path) as _db:
            _db.init_default_gates()
            _seed_upload(_db, "n-1")
            _seed_narrative(_db, "n-2", title="Sunset Hike")
            _seed_upload(
                _db, "n-2", seed_narrative=False,
                youtube_video_id="xyz789",
                youtube_url="https://youtube.com/watch?v=xyz789",
            )
        app = create_app(db_path=db_path)
        client = TestClient(app)
        resp = client.get("/api/uploads")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        titles = {d["narrative_title"] for d in data}
        assert titles == {"Morning Walk", "Sunset Hike"}

    def test_includes_all_fields(self, tmp_path: Path) -> None:
        """Each upload includes all expected fields."""
        db_path = str(tmp_path / "app.db")
        with CatalogDB(db_path) as _db:
            _db.init_default_gates()
            _seed_upload(_db, "n-1")
        app = create_app(db_path=db_path)
        client = TestClient(app)
        resp = client.get("/api/uploads")
        data = resp.json()
        assert len(data) == 1
        row = data[0]
        assert row["narrative_id"] == "n-1"
        assert row["youtube_video_id"] == "abc123"
        assert row["youtube_url"] == "https://youtube.com/watch?v=abc123"
        assert row["uploaded_at"] == "2025-01-15T10:30:00"
        assert row["privacy_status"] == "unlisted"
        assert row["narrative_title"] == "Morning Walk"
