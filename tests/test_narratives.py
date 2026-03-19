"""Tests for narrative organization (autopilot.organize.narratives)."""

from __future__ import annotations

import inspect
import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# -- Step 1: Public API surface tests ------------------------------------------


class TestPublicAPI:
    """Verify public API surface and type signatures."""

    def test_narrative_error_importable(self):
        """NarrativeError is importable from narratives module."""
        from autopilot.organize.narratives import NarrativeError

        assert NarrativeError is not None

    def test_narrative_error_is_exception(self):
        """NarrativeError is a subclass of Exception with message."""
        from autopilot.organize.narratives import NarrativeError

        assert issubclass(NarrativeError, Exception)
        err = NarrativeError("test message")
        assert str(err) == "test message"

    def test_narrative_dataclass_fields(self):
        """Narrative has all required fields."""
        from autopilot.organize.narratives import Narrative

        n = Narrative()
        assert hasattr(n, "narrative_id")
        assert hasattr(n, "title")
        assert hasattr(n, "description")
        assert hasattr(n, "proposed_duration_seconds")
        assert hasattr(n, "activity_cluster_ids")
        assert hasattr(n, "arc")
        assert hasattr(n, "emotional_journey")
        assert hasattr(n, "reasoning")
        assert hasattr(n, "status")

    def test_narrative_default_status(self):
        """Narrative defaults to status='proposed'."""
        from autopilot.organize.narratives import Narrative

        n = Narrative()
        assert n.status == "proposed"

    def test_narrative_activity_cluster_ids_is_list(self):
        """Narrative.activity_cluster_ids defaults to empty list."""
        from autopilot.organize.narratives import Narrative

        n = Narrative()
        assert isinstance(n.activity_cluster_ids, list)

    def test_narrative_arc_is_dict(self):
        """Narrative.arc defaults to empty dict."""
        from autopilot.organize.narratives import Narrative

        n = Narrative()
        assert isinstance(n.arc, dict)

    def test_build_master_storyboard_signature(self):
        """build_master_storyboard has db parameter and returns str."""
        from autopilot.organize.narratives import build_master_storyboard

        sig = inspect.signature(build_master_storyboard)
        params = list(sig.parameters.keys())
        assert "db" in params
        assert sig.return_annotation in (str, "str")

    def test_propose_narratives_signature(self):
        """propose_narratives has storyboard, db, config parameters."""
        from autopilot.organize.narratives import propose_narratives

        sig = inspect.signature(propose_narratives)
        params = list(sig.parameters.keys())
        assert "storyboard" in params
        assert "db" in params
        assert "config" in params

    def test_format_for_review_exists(self):
        """format_for_review function exists and is callable."""
        from autopilot.organize.narratives import format_for_review

        assert callable(format_for_review)

    def test_format_for_review_signature(self):
        """format_for_review has narratives parameter and returns str."""
        from autopilot.organize.narratives import format_for_review

        sig = inspect.signature(format_for_review)
        params = list(sig.parameters.keys())
        assert "narratives" in params
        assert sig.return_annotation in (str, "str")


# -- Step 3: _build_cluster_summary tests -------------------------------------


def _seed_cluster_with_full_data(catalog_db):
    """Seed DB with a cluster that has media, transcripts, detections, audio, faces."""
    # Media with duration
    catalog_db.insert_media(
        "v1", "/tmp/v1.mp4",
        created_at="2025-01-01T10:00:00",
        duration_seconds=120.0,
        gps_lat=18.788, gps_lon=98.985,
    )
    catalog_db.insert_media(
        "v2", "/tmp/v2.mp4",
        created_at="2025-01-01T10:05:00",
        duration_seconds=180.0,
        gps_lat=18.789, gps_lon=98.986,
    )

    # Transcripts
    catalog_db.upsert_transcript(
        "v1",
        json.dumps([
            {"text": "Look at the beautiful temple!", "start": 0.0, "end": 2.0},
            {"text": "This is incredible.", "start": 2.0, "end": 4.0},
        ]),
        "en",
    )
    catalog_db.upsert_transcript(
        "v2",
        json.dumps([{"text": "The monks are chanting.", "start": 0.0, "end": 3.0}]),
        "en",
    )

    # YOLO detections
    catalog_db.batch_insert_detections([
        ("v1", 0, json.dumps([
            {"class": "person", "confidence": 0.95},
            {"class": "building", "confidence": 0.88},
        ])),
        ("v1", 30, json.dumps([
            {"class": "person", "confidence": 0.92},
            {"class": "person", "confidence": 0.85},
        ])),
        ("v2", 0, json.dumps([
            {"class": "person", "confidence": 0.40},  # low confidence
            {"class": "building", "confidence": 0.30},
        ])),
    ])

    # Audio events
    catalog_db.batch_insert_audio_events([
        ("v1", 0.0, json.dumps([
            {"class": "Speech", "probability": 0.9},
            {"class": "Music", "probability": 0.3},
        ])),
        ("v2", 0.0, json.dumps([
            {"class": "Chanting", "probability": 0.85},
        ])),
    ])

    # Faces with clusters
    emb = np.zeros(512, dtype=np.float32).tobytes()
    catalog_db.batch_insert_faces([
        ("v1", 0, 0, json.dumps({"x": 10, "y": 10, "w": 50, "h": 50}), emb, 1),
        ("v2", 0, 0, json.dumps({"x": 20, "y": 20, "w": 60, "h": 60}), emb, 1),
    ])
    catalog_db.insert_face_cluster(1, label="Person A")

    # Activity cluster
    cluster = {
        "cluster_id": "c1",
        "clip_ids_json": json.dumps(["v1", "v2"]),
        "time_start": "2025-01-01T10:00:00",
        "time_end": "2025-01-01T10:08:00",
        "label": "Temple visit",
        "description": "Visiting a Buddhist temple",
        "gps_center_lat": 18.7885,
        "gps_center_lon": 98.9855,
    }
    return cluster


class TestBuildClusterSummary:
    """Tests for _build_cluster_summary() helper."""

    def test_full_data_returns_all_keys(self, catalog_db):
        """Summary with full data has all required keys."""
        from autopilot.organize.narratives import _build_cluster_summary

        cluster = _seed_cluster_with_full_data(catalog_db)
        summary = _build_cluster_summary(cluster, catalog_db)

        assert "label" in summary
        assert "description" in summary
        assert "duration" in summary
        assert "key_moments" in summary
        assert "people_present" in summary
        assert "emotional_tone" in summary
        assert "visual_quality_notes" in summary

    def test_duration_from_media(self, catalog_db):
        """Duration is computed from media files."""
        from autopilot.organize.narratives import _build_cluster_summary

        cluster = _seed_cluster_with_full_data(catalog_db)
        summary = _build_cluster_summary(cluster, catalog_db)

        # v1=120s + v2=180s = 300s total
        assert "300" in summary["duration"] or "5" in summary["duration"]

    def test_key_moments_include_transcripts(self, catalog_db):
        """Key moments include transcript highlights."""
        from autopilot.organize.narratives import _build_cluster_summary

        cluster = _seed_cluster_with_full_data(catalog_db)
        summary = _build_cluster_summary(cluster, catalog_db)

        assert "temple" in summary["key_moments"].lower() or "monk" in summary["key_moments"].lower()

    def test_people_present_includes_face_labels(self, catalog_db):
        """People present includes face cluster labels."""
        from autopilot.organize.narratives import _build_cluster_summary

        cluster = _seed_cluster_with_full_data(catalog_db)
        summary = _build_cluster_summary(cluster, catalog_db)

        assert "Person A" in summary["people_present"]

    def test_emotional_tone_from_audio(self, catalog_db):
        """Emotional tone inferred from audio events."""
        from autopilot.organize.narratives import _build_cluster_summary

        cluster = _seed_cluster_with_full_data(catalog_db)
        summary = _build_cluster_summary(cluster, catalog_db)

        # Should mention audio event classes
        assert summary["emotional_tone"] != ""

    def test_visual_quality_from_detection_confidence(self, catalog_db):
        """Visual quality notes derived from detection confidence."""
        from autopilot.organize.narratives import _build_cluster_summary

        cluster = _seed_cluster_with_full_data(catalog_db)
        summary = _build_cluster_summary(cluster, catalog_db)

        assert summary["visual_quality_notes"] != ""

    def test_empty_data_graceful(self, catalog_db):
        """Summary handles cluster with no signals gracefully."""
        from autopilot.organize.narratives import _build_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": json.dumps(["v1"]),
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:00:00",
            "label": None,
            "description": None,
            "gps_center_lat": None,
            "gps_center_lon": None,
        }

        summary = _build_cluster_summary(cluster, catalog_db)
        assert isinstance(summary, dict)
        assert "label" in summary
        assert "duration" in summary

    def test_corrupt_json_resilience(self, catalog_db):
        """Handles corrupt JSON in DB rows without crashing."""
        from autopilot.organize.narratives import _build_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        # Insert corrupt data
        catalog_db.conn.execute(
            "INSERT INTO transcripts (media_id, segments_json, language) VALUES (?, ?, ?)",
            ("v1", "CORRUPT JSON", "en"),
        )
        catalog_db.conn.execute(
            "INSERT INTO detections (media_id, frame_number, detections_json) VALUES (?, ?, ?)",
            ("v1", 0, "BAD"),
        )
        catalog_db.conn.execute(
            "INSERT INTO audio_events (media_id, timestamp_seconds, events_json) VALUES (?, ?, ?)",
            ("v1", 0.0, "INVALID"),
        )

        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": json.dumps(["v1"]),
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:00:00",
            "label": "Test",
            "description": "Test cluster",
            "gps_center_lat": None,
            "gps_center_lon": None,
        }

        # Should not crash
        summary = _build_cluster_summary(cluster, catalog_db)
        assert isinstance(summary, dict)


# -- Step 5: build_master_storyboard tests -------------------------------------


class TestBuildMasterStoryboard:
    """Tests for build_master_storyboard()."""

    def test_multiple_clusters(self, catalog_db):
        """Storyboard contains summaries for multiple clusters."""
        from autopilot.organize.narratives import build_master_storyboard

        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00",
            duration_seconds=120.0,
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4", created_at="2025-01-01T14:00:00",
            duration_seconds=60.0,
        )
        catalog_db.upsert_transcript(
            "v1",
            json.dumps([{"text": "Hello at the temple", "start": 0.0, "end": 2.0}]),
            "en",
        )

        catalog_db.insert_activity_cluster(
            "c1",
            label="Temple visit",
            description="Exploring a temple",
            time_start="2025-01-01T10:00:00",
            time_end="2025-01-01T10:30:00",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_activity_cluster(
            "c2",
            label="Market walk",
            description="Walking through a market",
            time_start="2025-01-01T14:00:00",
            time_end="2025-01-01T14:30:00",
            clip_ids_json=json.dumps(["v2"]),
        )

        storyboard = build_master_storyboard(catalog_db)

        assert isinstance(storyboard, str)
        assert "Temple visit" in storyboard
        assert "Market walk" in storyboard
        assert "c1" in storyboard
        assert "c2" in storyboard

    def test_empty_db(self, catalog_db):
        """Empty database returns empty/minimal storyboard."""
        from autopilot.organize.narratives import build_master_storyboard

        storyboard = build_master_storyboard(catalog_db)
        assert isinstance(storyboard, str)

    def test_cluster_without_label(self, catalog_db):
        """Clusters without labels still appear in storyboard."""
        from autopilot.organize.narratives import build_master_storyboard

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.insert_activity_cluster(
            "c1",
            time_start="2025-01-01T10:00:00",
            time_end="2025-01-01T10:30:00",
            clip_ids_json=json.dumps(["v1"]),
        )

        storyboard = build_master_storyboard(catalog_db)
        assert isinstance(storyboard, str)
        assert "c1" in storyboard


# -- Step 7: _load_and_fill_prompt tests ---------------------------------------


class TestLoadAndFillPrompt:
    """Tests for _load_and_fill_prompt() helper."""

    def test_fills_all_placeholders(self):
        """All 6 creator profile placeholders are replaced."""
        from autopilot.config import AutopilotConfig, CreatorConfig
        from autopilot.organize.narratives import _load_and_fill_prompt

        config = AutopilotConfig(
            input_dir=__import__("pathlib").Path("."),
            output_dir=__import__("pathlib").Path("."),
            creator=CreatorConfig(
                name="Test Creator",
                channel_style="Travel vlog",
                target_audience="Travel enthusiasts",
                default_video_duration_minutes="10-15",
                narration_style="First person conversational",
                music_preference="Ambient lo-fi",
            ),
        )

        prompt = _load_and_fill_prompt(config)

        assert isinstance(prompt, str)
        assert "Test Creator" in prompt
        assert "Travel vlog" in prompt
        assert "Travel enthusiasts" in prompt
        assert "10-15" in prompt
        assert "First person conversational" in prompt
        assert "Ambient lo-fi" in prompt
        # No remaining placeholders
        assert "{creator_name}" not in prompt
        assert "{channel_style}" not in prompt
        assert "{target_audience}" not in prompt
        assert "{default_video_duration}" not in prompt
        assert "{narration_style}" not in prompt
        assert "{music_preference}" not in prompt

    def test_missing_prompt_raises_narrative_error(self):
        """Missing prompt file raises NarrativeError."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import NarrativeError, _load_and_fill_prompt

        config = AutopilotConfig(
            input_dir=__import__("pathlib").Path("."),
            output_dir=__import__("pathlib").Path("."),
        )

        with patch(
            "autopilot.organize.narratives._PROMPT_PATH",
            __import__("pathlib").Path("/nonexistent/prompt.md"),
        ):
            with pytest.raises(NarrativeError, match="[Pp]rompt"):
                _load_and_fill_prompt(config)
