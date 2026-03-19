"""Tests for activity classification/labeling (autopilot.organize.classify)."""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock, patch

import pytest


# -- Step 7: Public API and summary assembly tests ----------------------------


class TestPublicAPI:
    """Verify public API surface and type signatures."""

    def test_exports_importable(self):
        """ClassifyError and label_activities are importable."""
        from autopilot.organize.classify import (
            ClassifyError,
            label_activities,
        )

        assert ClassifyError is not None
        assert label_activities is not None

    def test_classify_error_is_exception(self):
        """ClassifyError is a subclass of Exception with message."""
        from autopilot.organize.classify import ClassifyError

        assert issubclass(ClassifyError, Exception)
        err = ClassifyError("test message")
        assert str(err) == "test message"

    def test_label_activities_signature(self):
        """label_activities has db and config parameters."""
        from autopilot.organize.classify import label_activities

        sig = inspect.signature(label_activities)
        params = list(sig.parameters.keys())
        assert "db" in params
        assert "config" in params


class TestSummaryAssembly:
    """Tests for _assemble_cluster_summary() helper."""

    def test_basic_summary(self, catalog_db):
        """Summary includes time range and clip count."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
            gps_lat=18.788, gps_lon=98.985,
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4", created_at="2025-01-01T10:30:00",
            gps_lat=18.789, gps_lon=98.986,
        )

        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": json.dumps(["v1", "v2"]),
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:30:00",
            "gps_center_lat": 18.7885,
            "gps_center_lon": 98.9855,
        }

        summary = _assemble_cluster_summary(cluster, catalog_db)
        assert "time_range" in summary
        assert "2025-01-01T10:00:00" in summary["time_range"]
        assert "2025-01-01T10:30:00" in summary["time_range"]

    def test_includes_transcripts(self, catalog_db):
        """Summary includes transcript excerpts when available."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.upsert_transcript(
            "v1",
            json.dumps([{"text": "Hello world", "start": 0.0, "end": 1.0}]),
            "en",
        )

        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": json.dumps(["v1"]),
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:00:00",
            "gps_center_lat": None,
            "gps_center_lon": None,
        }

        summary = _assemble_cluster_summary(cluster, catalog_db)
        assert "transcripts" in summary
        assert "Hello world" in summary["transcripts"]

    def test_includes_detections(self, catalog_db):
        """Summary includes top YOLO detection classes."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.batch_insert_detections([
            ("v1", 0, json.dumps([
                {"class": "person", "confidence": 0.95},
                {"class": "car", "confidence": 0.88},
            ])),
            ("v1", 30, json.dumps([
                {"class": "person", "confidence": 0.92},
                {"class": "bicycle", "confidence": 0.75},
            ])),
        ])

        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": json.dumps(["v1"]),
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:00:00",
            "gps_center_lat": None,
            "gps_center_lon": None,
        }

        summary = _assemble_cluster_summary(cluster, catalog_db)
        assert "detections" in summary
        assert "person" in summary["detections"]

    def test_includes_audio_events(self, catalog_db):
        """Summary includes top audio event classes."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.batch_insert_audio_events([
            ("v1", 0.0, json.dumps([
                {"class": "Speech", "probability": 0.9},
                {"class": "Music", "probability": 0.3},
            ])),
        ])

        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": json.dumps(["v1"]),
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:00:00",
            "gps_center_lat": None,
            "gps_center_lon": None,
        }

        summary = _assemble_cluster_summary(cluster, catalog_db)
        assert "audio_events" in summary
        assert "Speech" in summary["audio_events"]

    def test_includes_gps(self, catalog_db):
        """Summary includes GPS coordinates when available."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
            gps_lat=18.788, gps_lon=98.985,
        )

        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": json.dumps(["v1"]),
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:00:00",
            "gps_center_lat": 18.788,
            "gps_center_lon": 98.985,
        }

        summary = _assemble_cluster_summary(cluster, catalog_db)
        assert "gps" in summary
        assert "18.788" in summary["gps"]

    def test_empty_data_graceful(self, catalog_db):
        """Summary handles missing transcripts/detections/audio gracefully."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")

        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": json.dumps(["v1"]),
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:00:00",
            "gps_center_lat": None,
            "gps_center_lon": None,
        }

        summary = _assemble_cluster_summary(cluster, catalog_db)
        assert isinstance(summary, dict)
        assert "time_range" in summary
