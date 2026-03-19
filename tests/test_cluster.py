"""Tests for activity clustering (autopilot.organize.cluster)."""

from __future__ import annotations

import inspect
import json
import math

import pytest


# -- DB helper tests (prerequisite) -------------------------------------------


class TestDBHelpers:
    """Verify CatalogDB helper methods needed by cluster/classify modules."""

    def test_list_all_media_returns_all(self, catalog_db):
        """list_all_media() returns all inserted media files."""
        catalog_db.insert_media("vid1", "/tmp/vid1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.insert_media("vid2", "/tmp/vid2.mp4", created_at="2025-01-01T11:00:00")

        result = catalog_db.list_all_media()
        assert len(result) == 2
        ids = {r["id"] for r in result}
        assert ids == {"vid1", "vid2"}

    def test_list_all_media_empty(self, catalog_db):
        """list_all_media() returns empty list when no media exists."""
        result = catalog_db.list_all_media()
        assert result == []

    def test_get_audio_events_for_media(self, catalog_db):
        """get_audio_events_for_media() returns all audio events for a media file."""
        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        catalog_db.batch_insert_audio_events([
            ("vid1", 0.0, json.dumps([{"class": "Speech", "probability": 0.9}])),
            ("vid1", 1.0, json.dumps([{"class": "Music", "probability": 0.7}])),
        ])

        result = catalog_db.get_audio_events_for_media("vid1")
        assert len(result) == 2
        timestamps = sorted(float(r["timestamp_seconds"]) for r in result)
        assert timestamps == [0.0, 1.0]

    def test_get_audio_events_for_media_empty(self, catalog_db):
        """get_audio_events_for_media() returns empty list when no events exist."""
        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        result = catalog_db.get_audio_events_for_media("vid1")
        assert result == []

    def test_get_detections_for_media(self, catalog_db):
        """get_detections_for_media() returns all detections for a media file."""
        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        catalog_db.batch_insert_detections([
            ("vid1", 0, json.dumps([{"class": "person", "confidence": 0.95}])),
            ("vid1", 30, json.dumps([{"class": "car", "confidence": 0.88}])),
        ])

        result = catalog_db.get_detections_for_media("vid1")
        assert len(result) == 2
        frames = sorted(int(r["frame_number"]) for r in result)
        assert frames == [0, 30]

    def test_get_detections_for_media_empty(self, catalog_db):
        """get_detections_for_media() returns empty list when no detections exist."""
        catalog_db.insert_media("vid1", "/tmp/vid1.mp4")
        result = catalog_db.get_detections_for_media("vid1")
        assert result == []

    def test_clear_activity_clusters(self, catalog_db):
        """clear_activity_clusters() deletes all activity clusters."""
        catalog_db.insert_activity_cluster(
            "c1", time_start="2025-01-01T10:00:00", clip_ids_json='["vid1"]',
        )
        catalog_db.insert_activity_cluster(
            "c2", time_start="2025-01-01T11:00:00", clip_ids_json='["vid2"]',
        )

        assert len(catalog_db.get_activity_clusters()) == 2

        catalog_db.clear_activity_clusters()

        assert len(catalog_db.get_activity_clusters()) == 0

    def test_clear_activity_clusters_empty(self, catalog_db):
        """clear_activity_clusters() succeeds when no clusters exist."""
        catalog_db.clear_activity_clusters()  # Should not raise
        assert len(catalog_db.get_activity_clusters()) == 0


# -- Step 1: Public API and helpers -------------------------------------------


class TestPublicAPI:
    """Verify public API surface and type signatures."""

    def test_exports_importable(self):
        """ClusterError, cluster_activities, ActivityCluster are importable."""
        from autopilot.organize.cluster import (
            ActivityCluster,
            ClusterError,
            cluster_activities,
        )

        assert ClusterError is not None
        assert cluster_activities is not None
        assert ActivityCluster is not None

    def test_cluster_error_is_exception(self):
        """ClusterError is a subclass of Exception with message."""
        from autopilot.organize.cluster import ClusterError

        assert issubclass(ClusterError, Exception)
        err = ClusterError("test message")
        assert str(err) == "test message"

    def test_activity_cluster_fields(self):
        """ActivityCluster has expected fields."""
        from autopilot.organize.cluster import ActivityCluster

        ac = ActivityCluster(
            cluster_id="c1",
            clip_ids=["vid1", "vid2"],
            time_start="2025-01-01T10:00:00",
            time_end="2025-01-01T11:00:00",
            gps_center_lat=18.7,
            gps_center_lon=98.9,
        )
        assert ac.cluster_id == "c1"
        assert ac.clip_ids == ["vid1", "vid2"]
        assert ac.time_start == "2025-01-01T10:00:00"
        assert ac.time_end == "2025-01-01T11:00:00"
        assert ac.gps_center_lat == 18.7
        assert ac.gps_center_lon == 98.9

    def test_activity_cluster_optional_gps(self):
        """ActivityCluster accepts None for GPS fields."""
        from autopilot.organize.cluster import ActivityCluster

        ac = ActivityCluster(
            cluster_id="c1",
            clip_ids=["vid1"],
            time_start="2025-01-01T10:00:00",
            time_end="2025-01-01T10:30:00",
        )
        assert ac.gps_center_lat is None
        assert ac.gps_center_lon is None

    def test_cluster_activities_signature(self):
        """cluster_activities has db parameter and returns list."""
        from autopilot.organize.cluster import cluster_activities

        sig = inspect.signature(cluster_activities)
        params = list(sig.parameters.keys())
        assert "db" in params


class TestHaversine:
    """Tests for _haversine() distance helper."""

    def test_known_distance(self):
        """New York to London is approximately 5570 km."""
        from autopilot.organize.cluster import _haversine

        # NYC: 40.7128, -74.0060
        # London: 51.5074, -0.1278
        d = _haversine(40.7128, -74.0060, 51.5074, -0.1278)
        assert abs(d - 5_570_000) < 50_000  # within 50km

    def test_zero_distance(self):
        """Same point -> 0 meters."""
        from autopilot.organize.cluster import _haversine

        d = _haversine(18.7883, 98.9853, 18.7883, 98.9853)
        assert d == 0.0

    def test_antipodal(self):
        """Opposite sides of Earth -> ~20015 km (half circumference)."""
        from autopilot.organize.cluster import _haversine

        d = _haversine(0.0, 0.0, 0.0, 180.0)
        assert abs(d - 20_015_000) < 100_000  # within 100km

    def test_short_distance(self):
        """Two points ~100m apart in same city."""
        from autopilot.organize.cluster import _haversine

        # Two points ~100m apart in Chiang Mai
        d = _haversine(18.7880, 98.9850, 18.7889, 98.9850)
        assert 50 < d < 200  # roughly 100m

    def test_symmetric(self):
        """d(a, b) == d(b, a)."""
        from autopilot.organize.cluster import _haversine

        d1 = _haversine(40.0, -74.0, 51.5, -0.1)
        d2 = _haversine(51.5, -0.1, 40.0, -74.0)
        assert d1 == pytest.approx(d2)


class TestTemporalSpatialClustering:
    """Tests for _temporal_spatial_cluster() private helper."""

    def test_clips_within_threshold_cluster_together(self):
        """Clips within 30min and 500m should be in the same cluster."""
        from autopilot.organize.cluster import _temporal_spatial_cluster

        clips = [
            {"id": "v1", "created_at": "2025-01-01T10:00:00", "gps_lat": 18.788, "gps_lon": 98.985},
            {"id": "v2", "created_at": "2025-01-01T10:10:00", "gps_lat": 18.788, "gps_lon": 98.985},
            {"id": "v3", "created_at": "2025-01-01T10:20:00", "gps_lat": 18.789, "gps_lon": 98.985},
        ]
        clusters = _temporal_spatial_cluster(clips)
        assert len(clusters) == 1
        assert set(clusters[0]) == {"v1", "v2", "v3"}

    def test_clips_far_apart_separate(self):
        """Clips > 30min apart should be in different clusters."""
        from autopilot.organize.cluster import _temporal_spatial_cluster

        clips = [
            {"id": "v1", "created_at": "2025-01-01T10:00:00", "gps_lat": 18.788, "gps_lon": 98.985},
            {"id": "v2", "created_at": "2025-01-01T12:00:00", "gps_lat": 18.788, "gps_lon": 98.985},
        ]
        clusters = _temporal_spatial_cluster(clips)
        assert len(clusters) == 2

    def test_spatial_separation(self):
        """Clips at same time but far apart spatially should separate."""
        from autopilot.organize.cluster import _temporal_spatial_cluster

        clips = [
            {"id": "v1", "created_at": "2025-01-01T10:00:00", "gps_lat": 18.788, "gps_lon": 98.985},
            {"id": "v2", "created_at": "2025-01-01T10:05:00", "gps_lat": 19.500, "gps_lon": 99.500},
        ]
        clusters = _temporal_spatial_cluster(clips)
        assert len(clusters) == 2

    def test_missing_gps_falls_back_to_temporal(self):
        """Clips with None GPS cluster by time only."""
        from autopilot.organize.cluster import _temporal_spatial_cluster

        clips = [
            {"id": "v1", "created_at": "2025-01-01T10:00:00", "gps_lat": None, "gps_lon": None},
            {"id": "v2", "created_at": "2025-01-01T10:10:00", "gps_lat": None, "gps_lon": None},
            {"id": "v3", "created_at": "2025-01-01T14:00:00", "gps_lat": None, "gps_lon": None},
        ]
        clusters = _temporal_spatial_cluster(clips)
        assert len(clusters) == 2
        # First two should cluster together
        cluster_with_v1 = [c for c in clusters if "v1" in c][0]
        assert "v2" in cluster_with_v1

    def test_single_clip(self):
        """Single clip -> single cluster."""
        from autopilot.organize.cluster import _temporal_spatial_cluster

        clips = [
            {"id": "v1", "created_at": "2025-01-01T10:00:00", "gps_lat": 18.788, "gps_lon": 98.985},
        ]
        clusters = _temporal_spatial_cluster(clips)
        assert len(clusters) == 1
        assert clusters[0] == ["v1"]

    def test_empty_clips(self):
        """Empty input -> empty output."""
        from autopilot.organize.cluster import _temporal_spatial_cluster

        clusters = _temporal_spatial_cluster([])
        assert clusters == []


# -- Step 3: Semantic refinement tests ----------------------------------------


class TestSemanticRefinement:
    """Tests for _semantic_refine() helper."""

    def _make_embedding(self, values: list[float]) -> bytes:
        """Create a bytes blob from a float32 list."""
        import numpy as np
        return np.array(values, dtype=np.float32).tobytes()

    def test_no_split_when_embeddings_similar(self, catalog_db):
        """Cluster with consistent embeddings should not be split."""
        from autopilot.organize.cluster import _semantic_refine

        # Create 3 clips with nearly identical embeddings
        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4", created_at="2025-01-01T10:05:00",
        )
        catalog_db.insert_media(
            "v3", "/tmp/v3.mp4", created_at="2025-01-01T10:10:00",
        )

        emb = self._make_embedding([1.0, 0.0, 0.0, 0.0])
        catalog_db.batch_insert_embeddings([
            ("v1", 0, emb),
            ("v2", 0, emb),
            ("v3", 0, emb),
        ])

        clip_ids = ["v1", "v2", "v3"]
        clips_data = {
            "v1": {"id": "v1", "created_at": "2025-01-01T10:00:00"},
            "v2": {"id": "v2", "created_at": "2025-01-01T10:05:00"},
            "v3": {"id": "v3", "created_at": "2025-01-01T10:10:00"},
        }

        result = _semantic_refine(clip_ids, clips_data, catalog_db)
        assert len(result) == 1
        assert set(result[0]) == {"v1", "v2", "v3"}

    def test_split_on_discontinuity(self, catalog_db):
        """Cluster with embedding discontinuity should be split."""
        from autopilot.organize.cluster import _semantic_refine

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4", created_at="2025-01-01T10:05:00",
        )
        catalog_db.insert_media(
            "v3", "/tmp/v3.mp4", created_at="2025-01-01T10:10:00",
        )
        catalog_db.insert_media(
            "v4", "/tmp/v4.mp4", created_at="2025-01-01T10:15:00",
        )

        # First two clips: [1, 0, 0, 0], last two: [0, 0, 0, 1] - very different
        emb_a = self._make_embedding([1.0, 0.0, 0.0, 0.0])
        emb_b = self._make_embedding([0.0, 0.0, 0.0, 1.0])
        catalog_db.batch_insert_embeddings([
            ("v1", 0, emb_a),
            ("v2", 0, emb_a),
            ("v3", 0, emb_b),
            ("v4", 0, emb_b),
        ])

        clip_ids = ["v1", "v2", "v3", "v4"]
        clips_data = {
            "v1": {"id": "v1", "created_at": "2025-01-01T10:00:00"},
            "v2": {"id": "v2", "created_at": "2025-01-01T10:05:00"},
            "v3": {"id": "v3", "created_at": "2025-01-01T10:10:00"},
            "v4": {"id": "v4", "created_at": "2025-01-01T10:15:00"},
        }

        result = _semantic_refine(clip_ids, clips_data, catalog_db)
        assert len(result) == 2

    def test_no_embeddings_no_split(self, catalog_db):
        """Cluster with no embeddings should not be split."""
        from autopilot.organize.cluster import _semantic_refine

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4", created_at="2025-01-01T10:05:00",
        )

        clip_ids = ["v1", "v2"]
        clips_data = {
            "v1": {"id": "v1", "created_at": "2025-01-01T10:00:00"},
            "v2": {"id": "v2", "created_at": "2025-01-01T10:05:00"},
        }

        result = _semantic_refine(clip_ids, clips_data, catalog_db)
        assert len(result) == 1
        assert set(result[0]) == {"v1", "v2"}

    def test_single_clip_no_split(self, catalog_db):
        """Single clip cluster is returned as-is."""
        from autopilot.organize.cluster import _semantic_refine

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
        )

        clip_ids = ["v1"]
        clips_data = {
            "v1": {"id": "v1", "created_at": "2025-01-01T10:00:00"},
        }

        result = _semantic_refine(clip_ids, clips_data, catalog_db)
        assert len(result) == 1
        assert result[0] == ["v1"]


# -- Step 5: cluster_activities end-to-end tests ------------------------------


class TestClusterActivities:
    """Tests for cluster_activities() end-to-end."""

    def test_basic_clustering(self, catalog_db):
        """Nearby clips grouped, distant clips separated."""
        from autopilot.organize.cluster import cluster_activities

        # Two close clips and one far away
        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4",
            created_at="2025-01-01T10:00:00", gps_lat=18.788, gps_lon=98.985,
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4",
            created_at="2025-01-01T10:10:00", gps_lat=18.788, gps_lon=98.985,
        )
        catalog_db.insert_media(
            "v3", "/tmp/v3.mp4",
            created_at="2025-01-01T14:00:00", gps_lat=19.500, gps_lon=99.500,
        )

        result = cluster_activities(catalog_db)
        assert len(result) == 2

        # Find which cluster has v1
        c1 = [c for c in result if "v1" in c.clip_ids][0]
        assert "v2" in c1.clip_ids
        assert "v3" not in c1.clip_ids

    def test_stores_in_db(self, catalog_db):
        """Clusters stored in activity_clusters table."""
        from autopilot.organize.cluster import cluster_activities

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4",
            created_at="2025-01-01T10:00:00", gps_lat=18.788, gps_lon=98.985,
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4",
            created_at="2025-01-01T10:10:00", gps_lat=18.788, gps_lon=98.985,
        )

        result = cluster_activities(catalog_db)
        assert len(result) == 1

        db_clusters = catalog_db.get_activity_clusters()
        assert len(db_clusters) == 1
        assert db_clusters[0]["cluster_id"] == result[0].cluster_id
        assert json.loads(str(db_clusters[0]["clip_ids_json"])) == result[0].clip_ids

    def test_correct_time_range(self, catalog_db):
        """time_start and time_end reflect clip range."""
        from autopilot.organize.cluster import cluster_activities

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4", created_at="2025-01-01T10:20:00",
        )

        result = cluster_activities(catalog_db)
        assert len(result) == 1
        assert "10:00:00" in result[0].time_start
        assert "10:20:00" in result[0].time_end

    def test_gps_center(self, catalog_db):
        """GPS center is mean of clip coordinates."""
        from autopilot.organize.cluster import cluster_activities

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4",
            created_at="2025-01-01T10:00:00", gps_lat=18.0, gps_lon=98.0,
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4",
            created_at="2025-01-01T10:10:00", gps_lat=18.002, gps_lon=98.002,
        )

        result = cluster_activities(catalog_db)
        assert len(result) == 1
        assert result[0].gps_center_lat == pytest.approx(18.001, abs=0.001)
        assert result[0].gps_center_lon == pytest.approx(98.001, abs=0.001)

    def test_idempotency(self, catalog_db):
        """Second call clears and re-creates clusters."""
        from autopilot.organize.cluster import cluster_activities

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
        )

        result1 = cluster_activities(catalog_db)
        result2 = cluster_activities(catalog_db)

        # Should have exactly 1 cluster after second call
        db_clusters = catalog_db.get_activity_clusters()
        assert len(db_clusters) == 1
        # cluster_id should change (new UUID)
        assert result1[0].cluster_id != result2[0].cluster_id

    def test_empty_db_returns_empty(self, catalog_db):
        """No media files -> empty list."""
        from autopilot.organize.cluster import cluster_activities

        result = cluster_activities(catalog_db)
        assert result == []

    def test_missing_gps_graceful(self, catalog_db):
        """All clips with None GPS still cluster by time."""
        from autopilot.organize.cluster import cluster_activities

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4", created_at="2025-01-01T10:10:00",
        )

        result = cluster_activities(catalog_db)
        assert len(result) == 1
        assert result[0].gps_center_lat is None
        assert result[0].gps_center_lon is None
