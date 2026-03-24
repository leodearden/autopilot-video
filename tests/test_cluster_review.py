"""Tests for cluster review routes and DB methods."""

from __future__ import annotations

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
    db.conn.commit()


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with a temp database."""
    db_path = str(tmp_path / "app.db")
    return create_app(db_path=db_path)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client for the app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# TestGetActivityCluster — step-1
# ---------------------------------------------------------------------------

class TestGetActivityCluster:
    """Tests for CatalogDB.get_activity_cluster(cluster_id)."""

    def test_returns_dict_for_existing_cluster(self, db: CatalogDB) -> None:
        """get_activity_cluster returns a dict with all fields for an existing cluster."""
        _seed_cluster(db, "c-1", label="Walk", description="Morning walk")
        result = db.get_activity_cluster("c-1")
        assert result is not None
        assert result["cluster_id"] == "c-1"
        assert result["label"] == "Walk"
        assert result["description"] == "Morning walk"
        assert result["time_start"] == "2025-01-01T08:00:00"
        assert result["time_end"] == "2025-01-01T09:00:00"
        assert result["location_label"] == "Park"
        assert result["gps_center_lat"] == 37.7749
        assert result["gps_center_lon"] == -122.4194
        assert result["clip_ids_json"] == '["clip-1","clip-2"]'

    def test_returns_none_for_missing_cluster(self, db: CatalogDB) -> None:
        """get_activity_cluster returns None for a non-existent cluster_id."""
        result = db.get_activity_cluster("nonexistent")
        assert result is None

    def test_excluded_defaults_to_zero(self, db: CatalogDB) -> None:
        """Newly inserted cluster has excluded=0 by default."""
        _seed_cluster(db, "c-1")
        result = db.get_activity_cluster("c-1")
        assert result is not None
        assert result["excluded"] == 0


# ---------------------------------------------------------------------------
# TestDeleteActivityCluster — step-3
# ---------------------------------------------------------------------------

class TestDeleteActivityCluster:
    """Tests for CatalogDB.delete_activity_cluster(cluster_id)."""

    def test_cluster_removed_after_delete(self, db: CatalogDB) -> None:
        """delete_activity_cluster removes the cluster; get returns None."""
        _seed_cluster(db, "c-1")
        db.delete_activity_cluster("c-1")
        db.conn.commit()
        assert db.get_activity_cluster("c-1") is None

    def test_no_error_deleting_missing_id(self, db: CatalogDB) -> None:
        """delete_activity_cluster does not raise for a non-existent id."""
        db.delete_activity_cluster("nonexistent")  # should not raise
