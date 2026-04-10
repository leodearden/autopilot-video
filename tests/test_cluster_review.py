"""Tests for cluster review routes and DB methods."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app
from tests.conftest import _seed_cluster, _seed_cluster_via_db

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
    """Create a FastAPI app with a temp database."""
    db_path = str(tmp_path / "app.db")
    return create_app(db_path=db_path)


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
# TestParseClusterMalformedJson — task-63 step-1
# ---------------------------------------------------------------------------


class TestParseClusterMalformedJson:
    """Tests for _parse_cluster handling of malformed JSON in clip_ids_json."""

    def test_malformed_json_returns_empty_clip_ids(self) -> None:
        """_parse_cluster returns empty clip_ids and clip_count=0 for invalid JSON."""
        from autopilot.web.routes.review import _parse_cluster

        row: dict[str, object] = {"clip_ids_json": "{bad json"}
        result = _parse_cluster(row)
        assert result["clip_ids"] == []
        assert result["clip_count"] == 0

    def test_none_clip_ids_json_returns_empty_list(self) -> None:
        """_parse_cluster returns empty clip_ids when clip_ids_json is None."""
        from autopilot.web.routes.review import _parse_cluster

        row: dict[str, object] = {"clip_ids_json": None}
        result = _parse_cluster(row)
        assert result["clip_ids"] == []
        assert result["clip_count"] == 0

    def test_valid_json_still_works(self) -> None:
        """_parse_cluster still parses valid JSON correctly."""
        from autopilot.web.routes.review import _parse_cluster

        row: dict[str, object] = {"clip_ids_json": '["a","b"]'}
        result = _parse_cluster(row)
        assert result["clip_ids"] == ["a", "b"]
        assert result["clip_count"] == 2


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
# TestCountNonExcludedClusters — task-65 step-5
# ---------------------------------------------------------------------------

class TestCountNonExcludedClusters:
    """Tests for CatalogDB.count_non_excluded_clusters()."""

    def test_counts_non_excluded_with_mixed_rows(self, db: CatalogDB) -> None:
        """Returns count of non-excluded clusters when mix of excluded/non-excluded."""
        _seed_cluster(db, "c-1")
        _seed_cluster(db, "c-2")
        _seed_cluster(db, "c-3")
        db.update_activity_cluster("c-2", excluded=1)
        db.conn.commit()
        assert db.count_non_excluded_clusters() == 2

    def test_returns_0_when_all_excluded(self, db: CatalogDB) -> None:
        """Returns 0 when every cluster is excluded."""
        _seed_cluster(db, "c-1")
        _seed_cluster(db, "c-2")
        db.update_activity_cluster("c-1", excluded=1)
        db.update_activity_cluster("c-2", excluded=1)
        db.conn.commit()
        assert db.count_non_excluded_clusters() == 0

    def test_returns_0_when_table_empty(self, db: CatalogDB) -> None:
        """Returns 0 when activity_clusters table has no rows."""
        assert db.count_non_excluded_clusters() == 0


# ---------------------------------------------------------------------------
# TestBatchDeleteActivityClusters — task-65 step-3
# ---------------------------------------------------------------------------

class TestBatchDeleteActivityClusters:
    """Tests for CatalogDB.batch_delete_activity_clusters(cluster_ids)."""

    def test_deletes_multiple_clusters(self, db: CatalogDB) -> None:
        """batch_delete_activity_clusters removes all specified clusters."""
        _seed_cluster(db, "c-1")
        _seed_cluster(db, "c-2")
        _seed_cluster(db, "c-3")
        result = db.batch_delete_activity_clusters(["c-1", "c-2"])
        db.conn.commit()
        assert result == 2
        assert db.get_activity_cluster("c-1") is None
        assert db.get_activity_cluster("c-2") is None
        assert db.get_activity_cluster("c-3") is not None

    def test_returns_0_on_empty_list(self, db: CatalogDB) -> None:
        """batch_delete_activity_clusters returns 0 for empty input."""
        _seed_cluster(db, "c-1")
        result = db.batch_delete_activity_clusters([])
        assert result == 0
        assert db.get_activity_cluster("c-1") is not None

    def test_skips_nonexistent_ids(self, db: CatalogDB) -> None:
        """batch_delete_activity_clusters silently skips non-existent IDs."""
        _seed_cluster(db, "c-1")
        result = db.batch_delete_activity_clusters(["c-1", "nonexistent"])
        db.conn.commit()
        assert result == 1
        assert db.get_activity_cluster("c-1") is None


# ---------------------------------------------------------------------------
# TestUpdateActivityClusterRowcount — task-65 step-1
# ---------------------------------------------------------------------------

class TestUpdateActivityClusterRowcount:
    """Tests that update_activity_cluster returns int rowcount."""

    def test_returns_1_for_existing_cluster(self, db: CatalogDB) -> None:
        """update_activity_cluster returns 1 when updating an existing row."""
        _seed_cluster(db, "c-1", label="Old")
        result = db.update_activity_cluster("c-1", label="New")
        assert result == 1

    def test_returns_0_for_nonexistent_cluster(self, db: CatalogDB) -> None:
        """update_activity_cluster returns 0 for a non-existent cluster_id."""
        result = db.update_activity_cluster("nonexistent", label="X")
        assert result == 0

    def test_returns_0_for_empty_kwargs(self, db: CatalogDB) -> None:
        """update_activity_cluster returns 0 for empty kwargs (early return)."""
        _seed_cluster(db, "c-1")
        result = db.update_activity_cluster("c-1")
        assert result == 0


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

    def test_excluded_clusters_still_listed(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """GET /api/clusters includes excluded clusters with excluded==1."""
        _seed_cluster_via_db(app, "c-1", label="Visible")
        _seed_cluster_via_db(app, "c-2", label="Hidden")
        # Exclude c-1
        client.post("/api/clusters/c-1/exclude")
        resp = client.get("/api/clusters")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        by_id = {c["cluster_id"]: c for c in data}
        assert by_id["c-1"]["excluded"] == 1
        assert by_id["c-2"]["excluded"] == 0


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

    def test_relabel_noop_empty_body(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """POST relabel with empty body {} is a valid no-op returning 200."""
        _seed_cluster_via_db(app, "c-1", label="Keep", description="Same")
        resp = client.post("/api/clusters/c-1/relabel", json={})
        assert resp.status_code == 200
        # Verify original values are unchanged via GET
        get_resp = client.get("/api/clusters/c-1")
        assert get_resp.status_code == 200
        persisted = get_resp.json()
        assert persisted["label"] == "Keep"
        assert persisted["description"] == "Same"

    def test_empty_body_nonexistent_cluster_returns_404(
        self, client: TestClient,
    ) -> None:
        """POST relabel with empty body {} on nonexistent cluster returns 404.

        When no fields are set, the UPDATE is skipped and _update_and_respond_cluster's
        SELECT finds nothing, yielding 404.
        """
        resp = client.post("/api/clusters/nonexistent/relabel", json={})
        assert resp.status_code == 404


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

    def test_rejects_empty_cluster_ids_list(self, client: TestClient) -> None:
        """Merge with empty cluster_ids list returns 422."""
        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": []},
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
        assert "nonexistent" in resp.json()["detail"]

    def test_merge_deduplicates_clip_ids(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Merge deduplicates shared clip_ids across clusters."""
        _seed_cluster_via_db(
            app, "c-1", clip_ids_json='["a","b"]',
        )
        _seed_cluster_via_db(
            app, "c-2", clip_ids_json='["b","c","d"]',
        )
        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        # 4 unique clips, not 5 (b is shared)
        assert len(data["clip_ids"]) == 4
        assert len(data["clip_ids"]) == len(set(data["clip_ids"]))
        assert set(data["clip_ids"]) == {"a", "b", "c", "d"}

    def test_merge_rolls_back_on_failure(
        self, app: FastAPI, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Merge rolls back all changes if batch_delete_activity_clusters fails."""
        _seed_cluster_via_db(app, "c-1", clip_ids_json='["a"]', label="Original-1")
        _seed_cluster_via_db(
            app, "c-2", clip_ids_json='["b","c"]', label="Original-2",
        )

        # Make batch_delete_activity_clusters fail after update has been applied
        original_batch_delete = CatalogDB.batch_delete_activity_clusters

        def _exploding_batch_delete(
            self: CatalogDB, cluster_ids: list[str],
        ) -> None:
            raise RuntimeError("simulated DB failure")

        monkeypatch.setattr(
            CatalogDB, "batch_delete_activity_clusters", _exploding_batch_delete,
        )

        err_client = TestClient(app, raise_server_exceptions=False)
        resp = err_client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 500

        # Restore original batch delete so we can read
        monkeypatch.setattr(
            CatalogDB, "batch_delete_activity_clusters", original_batch_delete,
        )

        # Both clusters should still exist with original data (rolled back)
        read_client = TestClient(app)
        r1 = read_client.get("/api/clusters/c-1")
        assert r1.status_code == 200
        assert r1.json()["label"] == "Original-1"

        r2 = read_client.get("/api/clusters/c-2")
        assert r2.status_code == 200
        assert r2.json()["label"] == "Original-2"
        # c-2 clip_ids should be unchanged (update was rolled back)
        assert r2.json()["clip_ids"] == ["b", "c"]

    def test_merge_malformed_timestamp_returns_422(
        self, app: FastAPI,
    ) -> None:
        """Merge returns 422 with actionable detail for malformed time_start."""
        _seed_cluster_via_db(
            app, "c-1",
            clip_ids_json='["a"]',
            time_start="2025-06-15T22:00:00+10:00",
            time_end="2025-06-15T23:00:00+10:00",
        )
        _seed_cluster_via_db(
            app, "c-2",
            clip_ids_json='["b","c"]',
            time_start="not-a-timestamp",
            time_end="2025-06-16T02:00:00+00:00",
        )
        err_client = TestClient(app, raise_server_exceptions=False)
        resp = err_client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "Invalid timestamp format" in detail
        assert "not-a-timestamp" in detail

    def test_merge_malformed_time_end_returns_422(
        self, app: FastAPI,
    ) -> None:
        """Merge returns 422 with actionable detail for malformed time_end."""
        _seed_cluster_via_db(
            app, "c-1",
            clip_ids_json='["a"]',
            time_start="2025-06-15T08:00:00",
            time_end="2025-06-15T09:00:00",
        )
        _seed_cluster_via_db(
            app, "c-2",
            clip_ids_json='["b","c"]',
            time_start="2025-06-15T10:00:00",
            time_end="2025/13/45",
        )
        err_client = TestClient(app, raise_server_exceptions=False)
        resp = err_client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "Invalid timestamp format" in detail
        assert "2025/13/45" in detail

    def test_merge_timestamp_comparison_chronological(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Merge uses chronological (not lexicographic) timestamp comparison."""
        # c-1: starts at 12:00 UTC (22:00+10:00), ends 13:00 UTC
        _seed_cluster_via_db(
            app, "c-1",
            clip_ids_json='["a"]',
            time_start="2025-06-15T22:00:00+10:00",
            time_end="2025-06-15T23:00:00+10:00",
        )
        # c-2: starts at 14:00 UTC, ends 02:00 UTC next day
        _seed_cluster_via_db(
            app, "c-2",
            clip_ids_json='["b","c"]',
            time_start="2025-06-15T14:00:00+00:00",
            time_end="2025-06-16T02:00:00+00:00",
        )
        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        # c-1 starts earlier chronologically (12:00 UTC < 14:00 UTC)
        assert data["time_start"] == "2025-06-15T22:00:00+10:00"
        # c-2 ends later chronologically (02:00 UTC next day > 13:00 UTC)
        assert data["time_end"] == "2025-06-16T02:00:00+00:00"

    def test_rejects_extra_fields(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Merge with extra fields returns 422 (extra='forbid')."""
        _seed_cluster_via_db(app, "c-1")
        _seed_cluster_via_db(app, "c-2")
        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"], "evil": "field"},
        )
        assert resp.status_code == 422

    def test_merge_with_null_clip_ids_json(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Merge where one cluster has NULL clip_ids_json succeeds."""
        _seed_cluster_via_db(
            app, "c-1", clip_ids_json=None,
        )
        _seed_cluster_via_db(
            app, "c-2", clip_ids_json='["a","b"]',
        )
        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        # c-2 is largest (2 clips vs 0), so it survives
        assert data["cluster_id"] == "c-2"
        assert sorted(data["clip_ids"]) == ["a", "b"]
        assert data["clip_count"] == 2
        # c-1 (null clips) is deleted
        assert client.get("/api/clusters/c-1").status_code == 404


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
# TestClassifyGateApproveCondition — task-63 step-3
# ---------------------------------------------------------------------------


class TestClassifyGateApproveCondition:
    """Tests for show_approve_gate logic on /review/clusters."""

    def test_shows_approve_when_non_excluded_clusters_exist(
        self, tmp_path: Path,
    ) -> None:
        """Approve gate button shows when reviewed (non-excluded) clusters exist."""
        app = _setup_classify_gate_app(tmp_path, clusters=2, excluded=0)
        client = TestClient(app)
        resp = client.get("/review/clusters")
        assert resp.status_code == 200
        assert "Approve Gate" in resp.text

    def test_hides_approve_when_all_clusters_excluded(
        self, tmp_path: Path,
    ) -> None:
        """Approve gate button hidden when ALL clusters are excluded."""
        app = _setup_classify_gate_app(tmp_path, clusters=2, excluded=2)
        client = TestClient(app)
        resp = client.get("/review/clusters")
        assert resp.status_code == 200
        assert "Approve Gate" not in resp.text


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
# TestClusterCardRelabelInputs — task-63 step-11
# ---------------------------------------------------------------------------


class TestClusterCardRelabelInputs:
    """Tests for cluster_card.html relabel input fields."""

    def test_non_excluded_card_has_label_input(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Non-excluded cluster card contains an input with name='label' and current value."""
        _seed_cluster_via_db(app, "c-1", label="Morning Walk", description="Walk in the park")
        resp = client.post(
            "/api/clusters/c-1/relabel",
            json={"label": "Morning Walk"},
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert 'name="label"' in body
        assert 'value="Morning Walk"' in body

    def test_non_excluded_card_has_description_input(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Non-excluded cluster card contains a textarea with name='description'."""
        _seed_cluster_via_db(app, "c-1", label="Morning Walk", description="Walk in the park")
        resp = client.post(
            "/api/clusters/c-1/relabel",
            json={"label": "Morning Walk"},
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 200
        body = resp.text
        assert 'name="description"' in body
        assert "Walk in the park" in body

    def test_relabel_button_sends_json_body(
        self, app: FastAPI, client: TestClient,
    ) -> None:
        """Relabel button has hx-vals with js: prefix to send JSON."""
        _seed_cluster_via_db(app, "c-1", label="Test", description="Desc")
        resp = client.post(
            "/api/clusters/c-1/relabel",
            json={"label": "Test"},
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 200
        body = resp.text
        # Button should have hx-vals='js:...' to gather input values as JSON
        assert "hx-vals" in body
        assert "js:" in body


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


# ---------------------------------------------------------------------------
# TestGetActivityClustersByIds — task 80, steps 1-3
# ---------------------------------------------------------------------------


class TestGetActivityClustersByIds:
    """Tests for CatalogDB.get_activity_clusters_by_ids(cluster_ids)."""

    def test_returns_dict_keyed_by_cluster_id(self, db: CatalogDB) -> None:
        """Seed 3 clusters, fetch 2 by ID, assert dict has exactly 2 keys."""
        _seed_cluster(db, "c-1", label="Alpha")
        _seed_cluster(db, "c-2", label="Beta")
        _seed_cluster(db, "c-3", label="Gamma")
        result = db.get_activity_clusters_by_ids(["c-1", "c-3"])
        assert isinstance(result, dict)
        assert set(result.keys()) == {"c-1", "c-3"}
        assert result["c-1"]["label"] == "Alpha"
        assert result["c-3"]["label"] == "Gamma"

    def test_returns_empty_dict_for_empty_input(self, db: CatalogDB) -> None:
        """Calling with [] returns {}."""
        result = db.get_activity_clusters_by_ids([])
        assert result == {}

    def test_omits_nonexistent_ids(self, db: CatalogDB) -> None:
        """Fetch [c-1, nonexistent] when only c-1 exists; dict has only c-1."""
        _seed_cluster(db, "c-1", label="Only")
        result = db.get_activity_clusters_by_ids(["c-1", "nonexistent"])
        assert set(result.keys()) == {"c-1"}
        assert result["c-1"]["label"] == "Only"


# ---------------------------------------------------------------------------
# TestApiMergeClustersUseBatchFetch — task 80, step-5
# ---------------------------------------------------------------------------


class TestApiMergeClustersUseBatchFetch:
    """Verify merge route uses batch fetch instead of per-cluster N+1 queries."""

    def _install_batch_spy(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> list[list[str]]:
        """Install N+1 trap and batch-fetch spy, return mutable args list."""
        def _n1_trap(self: object, cluster_id: str) -> None:
            raise AssertionError(f"N+1 detected: get_activity_cluster({cluster_id!r})")

        monkeypatch.setattr(CatalogDB, "get_activity_cluster", _n1_trap)

        batch_args: list[list[str]] = []
        _original_batch = CatalogDB.get_activity_clusters_by_ids

        def _batch_spy(self: object, cluster_ids: list[str]) -> dict:  # type: ignore[type-arg]
            batch_args.append(list(cluster_ids))
            return _original_batch(self, cluster_ids)  # type: ignore[arg-type]

        monkeypatch.setattr(CatalogDB, "get_activity_clusters_by_ids", _batch_spy)
        return batch_args

    def test_install_batch_spy_returns_list_and_traps_n1(
        self, app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_install_batch_spy returns empty list, traps N+1, and records batch calls."""
        batch_args = self._install_batch_spy(monkeypatch)

        # (a) Returns an empty list initially
        assert batch_args == []

        # (b) Single-cluster fetch raises AssertionError with 'N+1 detected'
        db = CatalogDB(app.state.db_path)
        with pytest.raises(AssertionError, match="N\\+1 detected"):
            db.get_activity_cluster("any-id")

        # (c) Batch fetch records call args and delegates to original
        _seed_cluster_via_db(app, "c-spy", label="Spy")
        result = db.get_activity_clusters_by_ids(["c-spy"])
        assert len(batch_args) == 1
        assert batch_args[0] == ["c-spy"]
        assert "c-spy" in result

    def test_merge_uses_batch_fetch(
        self, app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Monkeypatch get_activity_cluster to raise; merge still succeeds via batch path."""
        _seed_cluster_via_db(
            app, "c-1", label="Small", clip_ids_json='["clip-1"]',
        )
        _seed_cluster_via_db(
            app, "c-2", label="Large", clip_ids_json='["clip-2","clip-3"]',
        )

        batch_args = self._install_batch_spy(monkeypatch)

        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["cluster_id"] == "c-2"
        assert sorted(data["clip_ids"]) == ["clip-1", "clip-2", "clip-3"]
        # Exact call count: initial fetch + post-merge re-fetch
        assert len(batch_args) == 2, f"Expected 2 batch calls, got {len(batch_args)}"
        # First call: fetch all requested cluster IDs
        assert batch_args[0] == ["c-1", "c-2"]
        # Second call: post-merge re-fetch of the surviving (largest) cluster
        assert batch_args[1] == ["c-2"]

    def test_merge_equal_size_clusters_first_in_request_wins(
        self, app: FastAPI, client: TestClient, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When clusters tie on clip_count, the first in request order survives.

        Python's max() returns the first element with the maximum key value.
        Since parsed_clusters preserves body.cluster_ids order, the first
        cluster_id in the request is the deterministic tie-breaking survivor.
        """
        _seed_cluster_via_db(
            app, "c-1", label="Alpha", clip_ids_json='["clip-a","clip-b"]',
        )
        _seed_cluster_via_db(
            app, "c-2", label="Beta", clip_ids_json='["clip-c","clip-d"]',
        )

        batch_args = self._install_batch_spy(monkeypatch)

        resp = client.post(
            "/api/clusters/merge",
            json={"cluster_ids": ["c-1", "c-2"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Tie-breaking: first in request order (c-1) survives
        assert data["cluster_id"] == "c-1"
        # All 4 clips from both clusters are merged
        assert sorted(data["clip_ids"]) == ["clip-a", "clip-b", "clip-c", "clip-d"]
        # Exact batch call count: initial fetch + post-merge re-fetch
        assert len(batch_args) == 2, f"Expected 2 batch calls, got {len(batch_args)}"
        assert batch_args[0] == ["c-1", "c-2"]
        # Second call: re-fetch of the survivor (c-1, first in request)
        assert batch_args[1] == ["c-1"]


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
        assert "2 activity clusters" in body
