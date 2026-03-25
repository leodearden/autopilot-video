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


# ---------------------------------------------------------------------------
# TestUpdateActivityClusterWhitelist — step-5
# ---------------------------------------------------------------------------

class TestUpdateActivityClusterWhitelist:
    """Tests for column whitelist on update_activity_cluster."""

    def test_rejects_disallowed_column(self, db: CatalogDB) -> None:
        """update_activity_cluster raises ValueError for columns not in whitelist."""
        _seed_cluster(db, "c-1")
        with pytest.raises(ValueError, match="Disallowed column"):
            db.update_activity_cluster("c-1", evil_column="hacked")

    def test_accepts_all_valid_columns(self, db: CatalogDB) -> None:
        """update_activity_cluster accepts all whitelisted columns."""
        _seed_cluster(db, "c-1")
        db.update_activity_cluster(
            "c-1",
            label="New Label",
            description="New desc",
            time_start="2025-02-01T08:00:00",
            time_end="2025-02-01T09:00:00",
            location_label="Beach",
            gps_center_lat=34.0,
            gps_center_lon=-118.0,
            clip_ids_json='["clip-3"]',
            excluded=1,
        )
        db.conn.commit()
        result = db.get_activity_cluster("c-1")
        assert result is not None
        assert result["label"] == "New Label"
        assert result["excluded"] == 1


# ---------------------------------------------------------------------------
# Helpers — seed clusters via HTTP test client
# ---------------------------------------------------------------------------

def _seed_cluster_via_db(app: FastAPI, cluster_id: str = "c-1", **overrides: object) -> None:
    """Insert a cluster directly into the app's database."""
    db = CatalogDB(app.state.db_path)
    try:
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
    finally:
        db.close()


# ---------------------------------------------------------------------------
# TestApiListClusters — step-7
# ---------------------------------------------------------------------------

class TestApiListClusters:
    """Tests for GET /api/clusters."""

    def test_returns_list_with_parsed_clip_ids(self, app: FastAPI, client: TestClient) -> None:
        """GET /api/clusters returns clusters with clip_ids list and clip_count."""
        _seed_cluster_via_db(app, "c-1", clip_ids_json='["clip-1","clip-2","clip-3"]')
        resp = client.get("/api/clusters")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        cluster = data[0]
        assert cluster["cluster_id"] == "c-1"
        assert cluster["clip_ids"] == ["clip-1", "clip-2", "clip-3"]
        assert cluster["clip_count"] == 3
        assert "clip_ids_json" not in cluster

    def test_returns_empty_list_when_no_clusters(self, client: TestClient) -> None:
        """GET /api/clusters returns empty list with no data."""
        resp = client.get("/api/clusters")
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# TestApiGetCluster — step-9
# ---------------------------------------------------------------------------

class TestApiGetCluster:
    """Tests for GET /api/clusters/{cluster_id}."""

    def test_returns_single_cluster_with_parsed_clip_ids(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """GET /api/clusters/{id} returns cluster with clip_ids and clip_count."""
        _seed_cluster_via_db(app, "c-1", clip_ids_json='["clip-1","clip-2"]')
        resp = client.get("/api/clusters/c-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_id"] == "c-1"
        assert data["clip_ids"] == ["clip-1", "clip-2"]
        assert data["clip_count"] == 2
        assert "clip_ids_json" not in data

    def test_returns_404_for_missing_cluster(self, client: TestClient) -> None:
        """GET /api/clusters/{id} returns 404 for non-existent cluster."""
        resp = client.get("/api/clusters/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestApiRelabelCluster — step-11
# ---------------------------------------------------------------------------

class TestApiRelabelCluster:
    """Tests for POST /api/clusters/{cluster_id}/relabel."""

    def test_updates_label_and_description(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """POST relabel updates label and description, returns updated cluster."""
        _seed_cluster_via_db(app, "c-1", label="Old", description="Old desc")
        resp = client.post(
            "/api/clusters/c-1/relabel",
            json={"label": "New Label", "description": "New desc"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "New Label"
        assert data["description"] == "New desc"
        assert data["cluster_id"] == "c-1"
        # Verify persistence via follow-up GET
        get_resp = client.get("/api/clusters/c-1")
        assert get_resp.status_code == 200
        persisted = get_resp.json()
        assert persisted["label"] == "New Label"
        assert persisted["description"] == "New desc"
        assert persisted["cluster_id"] == "c-1"

    def test_returns_404_for_missing_cluster(self, client: TestClient) -> None:
        """POST relabel returns 404 for non-existent cluster."""
        resp = client.post(
            "/api/clusters/nonexistent/relabel",
            json={"label": "X"},
        )
        assert resp.status_code == 404

    def test_rejects_extra_fields(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """POST relabel rejects extra fields via Pydantic validation (422)."""
        _seed_cluster_via_db(app, "c-1")
        resp = client.post(
            "/api/clusters/c-1/relabel",
            json={"label": "X", "evil": "field"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# TestApiExcludeCluster — step-13
# ---------------------------------------------------------------------------

class TestApiExcludeCluster:
    """Tests for POST /api/clusters/{cluster_id}/exclude."""

    def test_sets_excluded_flag(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """POST exclude sets excluded=1 and returns updated cluster."""
        _seed_cluster_via_db(app, "c-1")
        resp = client.post("/api/clusters/c-1/exclude")
        assert resp.status_code == 200
        data = resp.json()
        assert data["excluded"] == 1
        assert data["cluster_id"] == "c-1"
        # Verify persistence via follow-up GET
        get_resp = client.get("/api/clusters/c-1")
        assert get_resp.status_code == 200
        persisted = get_resp.json()
        assert persisted["excluded"] == 1
        assert persisted["cluster_id"] == "c-1"

    def test_returns_404_for_missing_cluster(self, client: TestClient) -> None:
        """POST exclude returns 404 for non-existent cluster."""
        resp = client.post("/api/clusters/nonexistent/exclude")
        assert resp.status_code == 404

    def test_idempotent_exclude(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Excluding already-excluded cluster succeeds."""
        _seed_cluster_via_db(app, "c-1")
        client.post("/api/clusters/c-1/exclude")
        resp = client.post("/api/clusters/c-1/exclude")
        assert resp.status_code == 200
        assert resp.json()["excluded"] == 1


# ---------------------------------------------------------------------------
# TestApiMergeClusters — step-15
# ---------------------------------------------------------------------------

class TestApiMergeClusters:
    """Tests for POST /api/clusters/merge."""

    def test_merges_two_clusters(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Merge 2 clusters: combined clips, label from largest, time range merged."""
        _seed_cluster_via_db(
            app, "c-1",
            label="Small",
            clip_ids_json='["clip-1"]',
            time_start="2025-01-01T08:00:00",
            time_end="2025-01-01T09:00:00",
        )
        _seed_cluster_via_db(
            app, "c-2",
            label="Large",
            clip_ids_json='["clip-2","clip-3","clip-4"]',
            time_start="2025-01-01T10:00:00",
            time_end="2025-01-01T12:00:00",
        )
        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Largest cluster (c-2) survives
        assert data["cluster_id"] == "c-2"
        assert data["label"] == "Large"
        # Combined clips
        assert sorted(data["clip_ids"]) == ["clip-1", "clip-2", "clip-3", "clip-4"]
        assert data["clip_count"] == 4
        # Time range: min start, max end
        assert data["time_start"] == "2025-01-01T08:00:00"
        assert data["time_end"] == "2025-01-01T12:00:00"
        # Other cluster deleted
        get_resp = client.get("/api/clusters/c-1")
        assert get_resp.status_code == 404

    def test_merges_three_clusters(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Merge 3 clusters into the largest one."""
        _seed_cluster_via_db(app, "c-1", clip_ids_json='["a"]')
        _seed_cluster_via_db(app, "c-2", clip_ids_json='["b","c"]')
        _seed_cluster_via_db(app, "c-3", clip_ids_json='["d","e","f"]')
        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2", "c-3"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_id"] == "c-3"
        assert data["clip_count"] == 6
        # Others deleted
        assert client.get("/api/clusters/c-1").status_code == 404
        assert client.get("/api/clusters/c-2").status_code == 404

    def test_rejects_fewer_than_two_ids(self, client: TestClient) -> None:
        """Merge with fewer than 2 cluster_ids returns 422."""
        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1"]},
        )
        assert resp.status_code == 422

    def test_returns_404_for_missing_cluster(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Merge returns 404 if any cluster_id not found."""
        _seed_cluster_via_db(app, "c-1")
        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "nonexistent"]},
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# TestClustersPage — step-17
# ---------------------------------------------------------------------------

class TestClustersPage:
    """Tests for GET /review/clusters HTML page."""

    def test_renders_cluster_cards(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Page shows cluster card labels and clip counts."""
        _seed_cluster_via_db(
            app, "c-1", label="Morning Walk", clip_ids_json='["a","b","c"]',
        )
        resp = client.get("/review/clusters")
        assert resp.status_code == 200
        body = resp.text
        assert "Morning Walk" in body
        assert "3 clips" in body

    def test_empty_state(self, client: TestClient) -> None:
        """Page shows 'No clusters' message when empty."""
        resp = client.get("/review/clusters")
        assert resp.status_code == 200
        assert "No clusters" in resp.text

    def test_excluded_badge(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Excluded clusters show 'excluded' badge."""
        _seed_cluster_via_db(app, "c-1", label="Hidden")
        # Exclude via API
        client.post("/api/clusters/c-1/exclude")
        resp = client.get("/review/clusters")
        assert resp.status_code == 200
        assert "excluded" in resp.text


# ---------------------------------------------------------------------------
# TestHtmxClusterResponses — step-19
# ---------------------------------------------------------------------------

class TestHtmxClusterResponses:
    """Tests for HTMX partial responses on cluster routes."""

    def test_relabel_htmx_returns_html_partial(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """POST relabel with HX-Request returns HTML with cluster div and updated label."""
        _seed_cluster_via_db(app, "c-1", label="Old Label")
        resp = client.post(
            "/api/clusters/c-1/relabel",
            json={"label": "New Label"},
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        body = resp.text
        assert 'id="cluster-c-1"' in body
        assert "New Label" in body

    def test_exclude_htmx_returns_html_partial(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """POST exclude with HX-Request returns HTML partial with 'excluded' badge."""
        _seed_cluster_via_db(app, "c-1")
        resp = client.post(
            "/api/clusters/c-1/exclude",
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "excluded" in resp.text


# ---------------------------------------------------------------------------
# TestReviewHubClassifyGate — step-21
# ---------------------------------------------------------------------------

def _setup_classify_gate_app(tmp_path: Path, clusters: int = 2, excluded: int = 0) -> FastAPI:
    """Create an app with classify gate waiting and some clusters."""
    db_path = str(tmp_path / "catalog.db")
    with CatalogDB(db_path) as _db:
        _db.init_default_gates()
        _db.update_gate("classify", status="waiting")
        _db.conn.commit()
        for i in range(clusters):
            _db.insert_activity_cluster(
                f"c-{i}",
                label=f"Cluster {i}",
                description=f"Description {i}",
                time_start="2025-01-01T08:00:00",
                time_end="2025-01-01T09:00:00",
                location_label="Park",
                gps_center_lat=37.7749,
                gps_center_lon=-122.4194,
                clip_ids_json='["clip-1"]',
            )
        # Exclude some clusters
        for i in range(excluded):
            _db.update_activity_cluster(f"c-{i}", excluded=1)
        _db.conn.commit()
    return create_app(db_path=db_path)


class TestReviewHubClassifyGate:
    """Tests for review hub integration with classify gate."""

    def test_hub_shows_classify_gate_with_pending_count(self, tmp_path: Path) -> None:
        """Hub shows classify card with link and non-excluded cluster count."""
        app = _setup_classify_gate_app(tmp_path, clusters=3, excluded=1)
        client = TestClient(app)
        resp = client.get("/review")
        assert resp.status_code == 200
        body = resp.text
        # Gate stage visible
        assert "classify" in body.lower()
        # Link to cluster review page
        assert "/review/clusters" in body
        # Pending count: 3 clusters - 1 excluded = 2
        assert "2" in body
        assert "activity clusters" in body
