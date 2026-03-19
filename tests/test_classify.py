"""Tests for activity classification/labeling (autopilot.organize.classify)."""

from __future__ import annotations

import inspect
import json
import sys
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


# -- Step 9: LLM labeling tests -----------------------------------------------


def _make_llm_response(label="Morning hike", description="A scenic hike.",
                       split_recommended=False, split_reason=None):
    """Create a mock Anthropic API response."""
    result_json = json.dumps({
        "label": label,
        "description": description,
        "split_recommended": split_recommended,
        "split_reason": split_reason,
    })
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = result_json
    mock_response.content = [mock_content]
    return mock_response


class TestLLMLabeling:
    """Tests for _call_llm() helper."""

    def test_calls_anthropic_api(self):
        """_call_llm calls Anthropic API with correct model."""
        from autopilot.organize.classify import _call_llm
        from autopilot.config import LLMConfig

        config = LLMConfig()
        summary = {"time_range": "2025-01-01T10:00:00 to 2025-01-01T11:00:00"}

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_llm_response()

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = _call_llm(summary, config)

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == config.utility_model

    def test_prompt_includes_activity_label_content(self):
        """System prompt contains activity_label.md content."""
        from autopilot.organize.classify import _call_llm
        from autopilot.config import LLMConfig

        config = LLMConfig()
        summary = {"time_range": "test"}

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_llm_response()

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            _call_llm(summary, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Activity Classification Specialist" in call_kwargs["system"]

    def test_parses_json_response(self):
        """Correctly parses JSON from response text."""
        from autopilot.organize.classify import _call_llm
        from autopilot.config import LLMConfig

        config = LLMConfig()
        summary = {"time_range": "test"}

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_llm_response(
            label="Sunset kayaking",
            description="A beautiful evening on the river.",
        )

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = _call_llm(summary, config)

        assert result["label"] == "Sunset kayaking"
        assert result["description"] == "A beautiful evening on the river."

    def test_parses_json_in_code_block(self):
        """Parses JSON wrapped in ```json code block."""
        from autopilot.organize.classify import _call_llm
        from autopilot.config import LLMConfig

        config = LLMConfig()
        summary = {"time_range": "test"}

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = '```json\n{"label": "Beach day", "description": "Fun at the beach.", "split_recommended": false, "split_reason": null}\n```'
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = _call_llm(summary, config)

        assert result["label"] == "Beach day"

    def test_api_error_raises_classify_error(self):
        """API errors wrapped in ClassifyError."""
        from autopilot.organize.classify import ClassifyError, _call_llm
        from autopilot.config import LLMConfig

        config = LLMConfig()
        summary = {"time_range": "test"}

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API timeout")

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with pytest.raises(ClassifyError, match="API.*failed"):
                _call_llm(summary, config)

    def test_malformed_json_raises_classify_error(self):
        """Malformed JSON response raises ClassifyError."""
        from autopilot.organize.classify import ClassifyError, _call_llm
        from autopilot.config import LLMConfig

        config = LLMConfig()
        summary = {"time_range": "test"}

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "This is not valid JSON at all"
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with pytest.raises(ClassifyError, match="parse"):
                _call_llm(summary, config)

    def test_missing_fields_raises_classify_error(self):
        """Response missing label/description raises ClassifyError."""
        from autopilot.organize.classify import ClassifyError, _call_llm
        from autopilot.config import LLMConfig

        config = LLMConfig()
        summary = {"time_range": "test"}

        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = json.dumps({"only_label": "test"})
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with pytest.raises(ClassifyError, match="missing required"):
                _call_llm(summary, config)
