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
