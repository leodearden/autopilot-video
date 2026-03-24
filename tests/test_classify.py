"""Tests for activity classification/labeling (autopilot.organize.classify)."""

from __future__ import annotations

import inspect
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from autopilot.llm import LlmError

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
            "v1",
            "/tmp/v1.mp4",
            created_at="2025-01-01T10:00:00",
            gps_lat=18.788,
            gps_lon=98.985,
        )
        catalog_db.insert_media(
            "v2",
            "/tmp/v2.mp4",
            created_at="2025-01-01T10:30:00",
            gps_lat=18.789,
            gps_lon=98.986,
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
        catalog_db.batch_insert_detections(
            [
                (
                    "v1",
                    0,
                    json.dumps(
                        [
                            {"class": "person", "confidence": 0.95},
                            {"class": "car", "confidence": 0.88},
                        ]
                    ),
                ),
                (
                    "v1",
                    30,
                    json.dumps(
                        [
                            {"class": "person", "confidence": 0.92},
                            {"class": "bicycle", "confidence": 0.75},
                        ]
                    ),
                ),
            ]
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
        assert "detections" in summary
        assert "person" in summary["detections"]

    def test_includes_audio_events(self, catalog_db):
        """Summary includes top audio event classes."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.batch_insert_audio_events(
            [
                (
                    "v1",
                    0.0,
                    json.dumps(
                        [
                            {"class": "Speech", "probability": 0.9},
                            {"class": "Music", "probability": 0.3},
                        ]
                    ),
                ),
            ]
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
        assert "audio_events" in summary
        assert "Speech" in summary["audio_events"]

    def test_includes_gps(self, catalog_db):
        """Summary includes GPS coordinates when available."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media(
            "v1",
            "/tmp/v1.mp4",
            created_at="2025-01-01T10:00:00",
            gps_lat=18.788,
            gps_lon=98.985,
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

    def test_corrupt_clip_ids_json_raises_classify_error(self, catalog_db):
        """Corrupt clip_ids_json raises ClassifyError (cluster is unusable)."""
        from autopilot.organize.classify import ClassifyError, _assemble_cluster_summary

        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": "NOT VALID JSON",
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:00:00",
            "gps_center_lat": None,
            "gps_center_lon": None,
        }

        with pytest.raises(ClassifyError, match="clip_ids"):
            _assemble_cluster_summary(cluster, catalog_db)

    def test_corrupt_segments_json_skips_row(self, catalog_db):
        """Corrupt segments_json in transcript is skipped, not a crash."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        # Insert corrupt transcript directly
        catalog_db.conn.execute(
            "INSERT INTO transcripts (media_id, segments_json, language) VALUES (?, ?, ?)",
            ("v1", "NOT VALID JSON", "en"),
        )

        cluster = {
            "cluster_id": "c1",
            "clip_ids_json": json.dumps(["v1"]),
            "time_start": "2025-01-01T10:00:00",
            "time_end": "2025-01-01T10:00:00",
            "gps_center_lat": None,
            "gps_center_lon": None,
        }

        # Should not crash - corrupt row is skipped
        summary = _assemble_cluster_summary(cluster, catalog_db)
        assert summary["transcripts"] == ""

    def test_corrupt_detections_json_skips_row(self, catalog_db):
        """Corrupt detections_json is skipped, not a crash."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.conn.execute(
            "INSERT INTO detections (media_id, frame_number, detections_json) VALUES (?, ?, ?)",
            ("v1", 0, "CORRUPT"),
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
        assert summary["detections"] == ""

    def test_corrupt_events_json_skips_row(self, catalog_db):
        """Corrupt events_json in audio events is skipped, not a crash."""
        from autopilot.organize.classify import _assemble_cluster_summary

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.conn.execute(
            "INSERT INTO audio_events (media_id, timestamp_seconds, events_json) VALUES (?, ?, ?)",
            ("v1", 0.0, "BAD JSON"),
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
        assert summary["audio_events"] == ""


# -- Step 9: LLM labeling tests -----------------------------------------------


def _make_llm_text(
    label="Morning hike", description="A scenic hike.", split_recommended=False, split_reason=None
):
    """Create a mock invoke_claude text response."""
    return json.dumps(
        {
            "label": label,
            "description": description,
            "split_recommended": split_recommended,
            "split_reason": split_reason,
        }
    )


class TestLLMLabeling:
    """Tests for _call_llm() helper using invoke_claude."""

    def test_calls_invoke_claude(self):
        """_call_llm calls invoke_claude with correct model and params."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import _call_llm

        config = LLMConfig()
        summary = {"time_range": "2025-01-01T10:00:00 to 2025-01-01T11:00:00"}

        with patch("autopilot.organize.classify.invoke_claude", return_value=_make_llm_text()) as mock_invoke:
            _call_llm(summary, config)

        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["model"] == config.utility_model
        assert call_kwargs["max_tokens"] == 1024

    def test_prompt_includes_activity_label_content(self):
        """System prompt contains activity_label.md content."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import _call_llm

        config = LLMConfig()
        summary = {"time_range": "test"}

        with patch("autopilot.organize.classify.invoke_claude", return_value=_make_llm_text()) as mock_invoke:
            _call_llm(summary, config)

        call_kwargs = mock_invoke.call_args[1]
        assert "Activity Classification Specialist" in call_kwargs["system"]

    def test_parses_json_response(self):
        """Correctly parses JSON from invoke_claude text response."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import _call_llm

        config = LLMConfig()
        summary = {"time_range": "test"}

        with patch(
            "autopilot.organize.classify.invoke_claude",
            return_value=_make_llm_text(label="Sunset kayaking", description="A beautiful evening on the river."),
        ):
            result = _call_llm(summary, config)

        assert result["label"] == "Sunset kayaking"
        assert result["description"] == "A beautiful evening on the river."

    def test_parses_json_in_code_block(self):
        """Parses JSON wrapped in ```json code block."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import _call_llm

        config = LLMConfig()
        summary = {"time_range": "test"}

        code_block_response = (
            '```json\n{"label": "Beach day", "description": "Fun at the beach.",'
            ' "split_recommended": false, "split_reason": null}\n```'
        )

        with patch("autopilot.organize.classify.invoke_claude", return_value=code_block_response):
            result = _call_llm(summary, config)

        assert result["label"] == "Beach day"

    def test_llm_error_raises_classify_error(self):
        """LlmError wrapped in ClassifyError."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import ClassifyError, _call_llm

        config = LLMConfig()
        summary = {"time_range": "test"}

        with patch("autopilot.organize.classify.invoke_claude", side_effect=LlmError("CLI timeout")):
            with pytest.raises(ClassifyError, match="API.*failed"):
                _call_llm(summary, config)

    def test_malformed_json_raises_classify_error(self):
        """Malformed JSON response raises ClassifyError."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import ClassifyError, _call_llm

        config = LLMConfig()
        summary = {"time_range": "test"}

        with patch("autopilot.organize.classify.invoke_claude", return_value="This is not valid JSON at all"):
            with pytest.raises(ClassifyError, match="parse"):
                _call_llm(summary, config)

    def test_missing_fields_raises_classify_error(self):
        """Response missing label/description raises ClassifyError."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import ClassifyError, _call_llm

        config = LLMConfig()
        summary = {"time_range": "test"}

        with patch(
            "autopilot.organize.classify.invoke_claude",
            return_value=json.dumps({"only_label": "test"}),
        ):
            with pytest.raises(ClassifyError, match="missing required"):
                _call_llm(summary, config)

    def test_load_prompt_failure_raises_classify_error(self):
        """Missing prompt file raises ClassifyError, not FileNotFoundError."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import ClassifyError, _call_llm

        config = LLMConfig()
        summary = {"time_range": "test"}

        with patch(
            "autopilot.organize.classify._load_prompt",
            side_effect=FileNotFoundError("activity_label.md not found"),
        ):
            with pytest.raises(ClassifyError, match="[Pp]rompt"):
                _call_llm(summary, config)

    def test_llm_error_empty_response_raises_classify_error(self):
        """LlmError from empty response raises ClassifyError."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import ClassifyError, _call_llm

        config = LLMConfig()
        summary = {"time_range": "test"}

        with patch(
            "autopilot.organize.classify.invoke_claude",
            side_effect=LlmError("Empty or missing result in CLI output"),
        ):
            with pytest.raises(ClassifyError, match="API.*failed"):
                _call_llm(summary, config)


# -- Step 11: label_activities end-to-end tests --------------------------------


class TestLabelActivities:
    """Tests for label_activities() end-to-end."""

    def test_labels_clusters(self, catalog_db):
        """Labels unlabeled clusters in the DB."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import label_activities

        config = LLMConfig()

        # Create a cluster in the DB
        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.insert_activity_cluster(
            "c1",
            time_start="2025-01-01T10:00:00",
            time_end="2025-01-01T10:30:00",
            clip_ids_json=json.dumps(["v1"]),
        )

        with patch(
            "autopilot.organize.classify.invoke_claude",
            return_value=_make_llm_text(description="A scenic morning hike."),
        ):
            label_activities(catalog_db, config)

        # Check DB was updated
        clusters = catalog_db.get_activity_clusters()
        assert len(clusters) == 1
        assert clusters[0]["label"] == "Morning hike"
        assert clusters[0]["description"] == "A scenic morning hike."

    def test_skips_already_labeled(self, catalog_db):
        """Skips clusters that already have labels."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import label_activities

        config = LLMConfig()

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.insert_activity_cluster(
            "c1",
            label="Existing label",
            description="Existing description",
            time_start="2025-01-01T10:00:00",
            time_end="2025-01-01T10:30:00",
            clip_ids_json=json.dumps(["v1"]),
        )

        with patch("autopilot.organize.classify.invoke_claude") as mock_invoke:
            label_activities(catalog_db, config)

        # LLM should NOT have been called
        mock_invoke.assert_not_called()

        # Label should remain unchanged
        clusters = catalog_db.get_activity_clusters()
        assert clusters[0]["label"] == "Existing label"

    def test_handles_empty_clusters(self, catalog_db):
        """No clusters in DB -> no errors."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import label_activities

        config = LLMConfig()
        # No clusters to label
        label_activities(catalog_db, config)  # Should not raise

    def test_labels_multiple_clusters(self, catalog_db):
        """Labels multiple clusters, calling invoke_claude for each."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import label_activities

        config = LLMConfig()

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.insert_media("v2", "/tmp/v2.mp4", created_at="2025-01-01T14:00:00")
        catalog_db.insert_activity_cluster(
            "c1",
            time_start="2025-01-01T10:00:00",
            time_end="2025-01-01T10:30:00",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_activity_cluster(
            "c2",
            time_start="2025-01-01T14:00:00",
            time_end="2025-01-01T14:30:00",
            clip_ids_json=json.dumps(["v2"]),
        )

        with patch(
            "autopilot.organize.classify.invoke_claude",
            return_value=_make_llm_text(),
        ) as mock_invoke:
            label_activities(catalog_db, config)

        # Both should be labeled
        clusters = catalog_db.get_activity_clusters()
        assert all(c["label"] == "Morning hike" for c in clusters)
        assert mock_invoke.call_count == 2

    def test_split_recommended_logs_warning(self, catalog_db, caplog):
        """split_recommended=true triggers WARNING log with cluster_id and split_reason."""
        import logging

        from autopilot.config import LLMConfig
        from autopilot.organize.classify import label_activities

        config = LLMConfig()

        catalog_db.insert_media("v1", "/tmp/v1.mp4", created_at="2025-01-01T10:00:00")
        catalog_db.insert_activity_cluster(
            "c-split-test",
            time_start="2025-01-01T10:00:00",
            time_end="2025-01-01T10:30:00",
            clip_ids_json=json.dumps(["v1"]),
        )

        split_response = _make_llm_text(
            label="Mixed activity",
            description="A mix of hiking and dining.",
            split_recommended=True,
            split_reason="Topic shift at 10:15 from hiking to restaurant",
        )

        with caplog.at_level(logging.WARNING, logger="autopilot.organize.classify"):
            with patch("autopilot.organize.classify.invoke_claude", return_value=split_response):
                label_activities(catalog_db, config)

        # Should have a WARNING log mentioning the cluster_id and split reason
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("c-split-test" in msg for msg in warning_msgs), (
            f"Expected warning with cluster_id, got: {warning_msgs}"
        )
        assert any("split" in msg.lower() for msg in warning_msgs), (
            f"Expected warning mentioning split, got: {warning_msgs}"
        )


# -- Step 13: Integration test ------------------------------------------------


class TestIntegration:
    """End-to-end integration: cluster then label."""

    def test_cluster_then_label(self, catalog_db):
        """Full pipeline: insert media, cluster, then label with mocked LLM."""
        import numpy as np

        from autopilot.config import LLMConfig
        from autopilot.organize.classify import label_activities
        from autopilot.organize.cluster import cluster_activities

        config = LLMConfig()

        # Insert media with timestamps, GPS, transcripts, detections, audio events
        catalog_db.insert_media(
            "v1",
            "/tmp/v1.mp4",
            created_at="2025-01-01T10:00:00",
            gps_lat=18.788,
            gps_lon=98.985,
        )
        catalog_db.insert_media(
            "v2",
            "/tmp/v2.mp4",
            created_at="2025-01-01T10:15:00",
            gps_lat=18.789,
            gps_lon=98.986,
        )
        catalog_db.insert_media(
            "v3",
            "/tmp/v3.mp4",
            created_at="2025-01-01T14:00:00",
            gps_lat=19.500,
            gps_lon=99.500,
        )

        # Add transcripts
        catalog_db.upsert_transcript(
            "v1",
            json.dumps([{"text": "Look at the temple!", "start": 0.0, "end": 2.0}]),
            "en",
        )
        catalog_db.upsert_transcript(
            "v3",
            json.dumps([{"text": "This market is amazing.", "start": 0.0, "end": 1.5}]),
            "en",
        )

        # Add detections
        catalog_db.batch_insert_detections(
            [
                ("v1", 0, json.dumps([{"class": "person", "confidence": 0.9}])),
                ("v2", 0, json.dumps([{"class": "temple", "confidence": 0.85}])),
                ("v3", 0, json.dumps([{"class": "food", "confidence": 0.88}])),
            ]
        )

        # Add audio events
        catalog_db.batch_insert_audio_events(
            [
                ("v1", 0.0, json.dumps([{"class": "Speech", "probability": 0.9}])),
                ("v3", 0.0, json.dumps([{"class": "Crowd", "probability": 0.8}])),
            ]
        )

        # Add embeddings
        emb = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).tobytes()
        catalog_db.batch_insert_embeddings(
            [
                ("v1", 0, emb),
                ("v2", 0, emb),
                ("v3", 0, emb),
            ]
        )

        # Phase 1: Cluster
        clusters = cluster_activities(catalog_db)
        assert len(clusters) == 2

        # Phase 2: Label with mocked invoke_claude
        temple_response = _make_llm_text(
            label="Temple visit",
            description="Exploring a beautiful temple.",
        )
        with patch(
            "autopilot.organize.classify.invoke_claude",
            return_value=temple_response,
        ) as mock_invoke:
            label_activities(catalog_db, config)

        # Verify all clusters labeled
        db_clusters = catalog_db.get_activity_clusters()
        assert len(db_clusters) == 2
        for c in db_clusters:
            assert c["label"] == "Temple visit"
            assert c["description"] is not None

        # Verify invoke_claude called twice (once per cluster)
        assert mock_invoke.call_count == 2
