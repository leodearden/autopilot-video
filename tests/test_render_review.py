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
