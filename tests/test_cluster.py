"""Tests for activity clustering (autopilot.organize.cluster)."""

from __future__ import annotations

import json

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
