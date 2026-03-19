"""Tests for narrative organization (autopilot.organize.narratives)."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
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


# -- Step 9: _call_llm / _parse_narratives tests ------------------------------


def _make_narrative_llm_response(narratives_json=None):
    """Create a mock Anthropic API response for narrative proposals."""
    if narratives_json is None:
        narratives_json = json.dumps([{
            "title": "Three Days in Northern Thailand",
            "activity_cluster_ids": ["c1", "c2"],
            "proposed_duration_seconds": 600,
            "arc": {
                "beginning": "Arrival at the temple",
                "middle": "Exploring the grounds",
                "end": "Sunset reflections",
            },
            "emotional_journey": "Curiosity to wonder to peace",
            "target_audience": "Travel enthusiasts",
            "reasoning": "These clusters form a natural arc.",
        }])
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = narratives_json
    mock_response.content = [mock_content]
    return mock_response


class TestCallLLM:
    """Tests for _call_llm() helper."""

    def test_uses_planning_model(self):
        """_call_llm calls Anthropic API with planning_model (not utility_model)."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import _call_llm

        config = AutopilotConfig(
            input_dir=Path("."), output_dir=Path("."),
        )
        storyboard = "# Master Storyboard\n\nCluster c1: Temple visit"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_narrative_llm_response()

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            _call_llm(storyboard, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == config.llm.planning_model

    def test_sends_storyboard_as_user_message(self):
        """Storyboard text is sent as user message content."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import _call_llm

        config = AutopilotConfig(
            input_dir=Path("."), output_dir=Path("."),
        )
        storyboard = "# My unique storyboard content"

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_narrative_llm_response()

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            _call_llm(storyboard, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        user_msg = messages[0]
        assert user_msg["role"] == "user"
        assert "My unique storyboard content" in user_msg["content"]

    def test_system_prompt_from_narrative_planner(self):
        """System prompt contains narrative_planner.md content."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import _call_llm

        config = AutopilotConfig(
            input_dir=Path("."), output_dir=Path("."),
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_narrative_llm_response()

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            _call_llm("storyboard", config)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Narrative Architect" in call_kwargs["system"]

    def test_api_error_raises_narrative_error(self):
        """API errors wrapped in NarrativeError."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import NarrativeError, _call_llm

        config = AutopilotConfig(
            input_dir=Path("."), output_dir=Path("."),
        )

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API timeout")

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with pytest.raises(NarrativeError, match="API.*failed"):
                _call_llm("storyboard", config)

    def test_empty_response_raises_narrative_error(self):
        """Empty response.content raises NarrativeError."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import NarrativeError, _call_llm

        config = AutopilotConfig(
            input_dir=Path("."), output_dir=Path("."),
        )

        mock_response = MagicMock()
        mock_response.content = []

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with pytest.raises(NarrativeError, match="[Ee]mpty"):
                _call_llm("storyboard", config)


class TestParseNarratives:
    """Tests for _parse_narratives() helper."""

    def test_parses_json_array(self):
        """Parses JSON array into Narrative objects."""
        from autopilot.organize.narratives import Narrative, _parse_narratives

        text = json.dumps([{
            "title": "Temple Day",
            "activity_cluster_ids": ["c1"],
            "proposed_duration_seconds": 480,
            "arc": {"beginning": "Arrival", "middle": "Exploring", "end": "Sunset"},
            "emotional_journey": "Wonder to peace",
            "target_audience": "Travelers",
            "reasoning": "Strong visual arc.",
        }])

        narratives = _parse_narratives(text)
        assert len(narratives) == 1
        assert isinstance(narratives[0], Narrative)
        assert narratives[0].title == "Temple Day"
        assert narratives[0].proposed_duration_seconds == 480
        assert narratives[0].activity_cluster_ids == ["c1"]
        assert "beginning" in narratives[0].arc

    def test_parses_json_in_code_block(self):
        """Parses JSON wrapped in ```json code block."""
        from autopilot.organize.narratives import _parse_narratives

        text = '```json\n[{"title": "Beach Day", "activity_cluster_ids": ["c1"], "proposed_duration_seconds": 300, "arc": {"beginning": "A", "middle": "B", "end": "C"}, "emotional_journey": "Joy", "target_audience": "All", "reasoning": "Fun."}]\n```'

        narratives = _parse_narratives(text)
        assert len(narratives) == 1
        assert narratives[0].title == "Beach Day"

    def test_generates_narrative_ids(self):
        """Each parsed narrative gets a unique narrative_id."""
        from autopilot.organize.narratives import _parse_narratives

        text = json.dumps([
            {"title": "A", "activity_cluster_ids": ["c1"], "proposed_duration_seconds": 300, "arc": {}, "emotional_journey": "", "target_audience": "", "reasoning": ""},
            {"title": "B", "activity_cluster_ids": ["c2"], "proposed_duration_seconds": 600, "arc": {}, "emotional_journey": "", "target_audience": "", "reasoning": ""},
        ])

        narratives = _parse_narratives(text)
        assert len(narratives) == 2
        assert narratives[0].narrative_id != ""
        assert narratives[1].narrative_id != ""
        assert narratives[0].narrative_id != narratives[1].narrative_id

    def test_malformed_json_raises_narrative_error(self):
        """Malformed JSON raises NarrativeError."""
        from autopilot.organize.narratives import NarrativeError, _parse_narratives

        with pytest.raises(NarrativeError, match="parse"):
            _parse_narratives("This is not valid JSON")

    def test_missing_required_fields_raises_narrative_error(self):
        """Response missing title raises NarrativeError."""
        from autopilot.organize.narratives import NarrativeError, _parse_narratives

        text = json.dumps([{"only_field": "test"}])
        with pytest.raises(NarrativeError, match="missing"):
            _parse_narratives(text)

    def test_empty_array_returns_empty_list(self):
        """Empty JSON array returns empty list."""
        from autopilot.organize.narratives import _parse_narratives

        narratives = _parse_narratives("[]")
        assert narratives == []


# -- Step 11: propose_narratives end-to-end tests -----------------------------


def _setup_mock_narrative_anthropic(narratives_data=None):
    """Create a mock anthropic module and client for narrative tests."""
    mock_response = _make_narrative_llm_response(
        json.dumps(narratives_data) if narratives_data is not None else None
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    return mock_anthropic, mock_client


class TestProposeNarratives:
    """Tests for propose_narratives() end-to-end."""

    def test_calls_llm_and_returns_narratives(self, catalog_db):
        """propose_narratives calls LLM and returns Narrative list."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import Narrative, propose_narratives

        config = AutopilotConfig(input_dir=Path("."), output_dir=Path("."))
        storyboard = "# Master Storyboard\n\n## Cluster: c1\n- **Label**: Temple visit"

        mock_anthropic, mock_client = _setup_mock_narrative_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = propose_narratives(storyboard, catalog_db, config)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Narrative)
        assert result[0].title == "Three Days in Northern Thailand"
        mock_client.messages.create.assert_called_once()

    def test_stores_narratives_in_db(self, catalog_db):
        """Proposed narratives are stored in the DB with status='proposed'."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import propose_narratives

        config = AutopilotConfig(input_dir=Path("."), output_dir=Path("."))
        storyboard = "# Master Storyboard"

        mock_anthropic, _ = _setup_mock_narrative_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = propose_narratives(storyboard, catalog_db, config)

        # Check DB
        db_narratives = catalog_db.list_narratives(status="proposed")
        assert len(db_narratives) == 1
        assert db_narratives[0]["title"] == "Three Days in Northern Thailand"
        assert db_narratives[0]["narrative_id"] == result[0].narrative_id

    def test_empty_storyboard_handled(self, catalog_db):
        """Empty storyboard still works (LLM may return empty array)."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import propose_narratives

        config = AutopilotConfig(input_dir=Path("."), output_dir=Path("."))

        mock_anthropic, _ = _setup_mock_narrative_anthropic([])
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = propose_narratives("", catalog_db, config)

        assert result == []
        assert catalog_db.list_narratives() == []

    def test_multiple_narratives_stored(self, catalog_db):
        """Multiple narrative proposals are all stored in DB."""
        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import propose_narratives

        config = AutopilotConfig(input_dir=Path("."), output_dir=Path("."))

        two_narratives = [
            {
                "title": "Narrative A",
                "activity_cluster_ids": ["c1"],
                "proposed_duration_seconds": 300,
                "arc": {"beginning": "A", "middle": "B", "end": "C"},
                "emotional_journey": "Joy",
                "target_audience": "All",
                "reasoning": "Good.",
            },
            {
                "title": "Narrative B",
                "activity_cluster_ids": ["c2", "c3"],
                "proposed_duration_seconds": 600,
                "arc": {"beginning": "D", "middle": "E", "end": "F"},
                "emotional_journey": "Wonder",
                "target_audience": "Travelers",
                "reasoning": "Great.",
            },
        ]

        mock_anthropic, _ = _setup_mock_narrative_anthropic(two_narratives)
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = propose_narratives("storyboard", catalog_db, config)

        assert len(result) == 2
        db_narratives = catalog_db.list_narratives()
        assert len(db_narratives) == 2


# -- Step 13: format_for_review tests -----------------------------------------


class TestFormatForReview:
    """Tests for format_for_review()."""

    def test_single_narrative(self):
        """Formats a single narrative with all sections."""
        from autopilot.organize.narratives import Narrative, format_for_review

        n = Narrative(
            narrative_id="n1",
            title="Temple Exploration",
            description="A morning at the temple.",
            proposed_duration_seconds=480,
            activity_cluster_ids=["c1", "c2"],
            arc={"beginning": "Arrival", "middle": "Exploring", "end": "Sunset"},
            emotional_journey="Curiosity to wonder to peace",
            reasoning="Strong visual arc with natural progression.",
        )

        result = format_for_review([n])

        assert isinstance(result, str)
        assert "Temple Exploration" in result
        assert "480" in result or "8" in result  # seconds or minutes
        assert "c1" in result
        assert "c2" in result
        assert "Arrival" in result
        assert "Curiosity" in result
        assert "Strong visual arc" in result

    def test_multiple_narratives_numbered(self):
        """Multiple narratives are numbered."""
        from autopilot.organize.narratives import Narrative, format_for_review

        n1 = Narrative(
            narrative_id="n1", title="Narrative A",
            proposed_duration_seconds=300,
            activity_cluster_ids=["c1"],
            arc={"beginning": "A", "middle": "B", "end": "C"},
            emotional_journey="Joy",
            reasoning="Good.",
        )
        n2 = Narrative(
            narrative_id="n2", title="Narrative B",
            proposed_duration_seconds=600,
            activity_cluster_ids=["c2", "c3"],
            arc={"beginning": "D", "middle": "E", "end": "F"},
            emotional_journey="Wonder",
            reasoning="Great.",
        )

        result = format_for_review([n1, n2])

        assert "1" in result
        assert "2" in result
        assert "Narrative A" in result
        assert "Narrative B" in result

    def test_empty_list(self):
        """Empty list returns appropriate message."""
        from autopilot.organize.narratives import format_for_review

        result = format_for_review([])
        assert isinstance(result, str)
        assert len(result) > 0  # Should have some message, not empty string
