"""Tests for script generation (autopilot.plan.script) and narrative_scripts DB CRUD."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# -- Pre-1: narrative_scripts DB CRUD tests ------------------------------------


class TestDBNarrativeScripts:
    """Tests for narrative_scripts table and CRUD methods in CatalogDB."""

    def test_upsert_and_get_narrative_script(self, catalog_db):
        """Upsert stores a script and get retrieves it."""
        # Need a narrative row first (FK)
        catalog_db.insert_narrative("n1", title="Test Narrative")

        script_data = json.dumps({"scenes": [], "broll_needs": [], "quality_flags": []})
        catalog_db.upsert_narrative_script("n1", script_data)

        result = catalog_db.get_narrative_script("n1")
        assert result is not None
        assert result["narrative_id"] == "n1"
        assert result["script_json"] == script_data
        assert "created_at" in result

    def test_get_nonexistent_returns_none(self, catalog_db):
        """get_narrative_script returns None for non-existent narrative."""
        result = catalog_db.get_narrative_script("nonexistent")
        assert result is None

    def test_upsert_overwrites(self, catalog_db):
        """Second upsert overwrites the first script_json."""
        catalog_db.insert_narrative("n1", title="Test Narrative")

        script_v1 = json.dumps({"scenes": [{"scene_number": 1}]})
        script_v2 = json.dumps({"scenes": [{"scene_number": 1}, {"scene_number": 2}]})

        catalog_db.upsert_narrative_script("n1", script_v1)
        catalog_db.upsert_narrative_script("n1", script_v2)

        result = catalog_db.get_narrative_script("n1")
        assert result is not None
        assert result["script_json"] == script_v2


# -- Step 1: Public API surface tests -----------------------------------------


class TestPublicAPI:
    """Verify ScriptError and public API surface."""

    def test_script_error_importable(self):
        """ScriptError is importable from script module."""
        from autopilot.plan.script import ScriptError

        assert ScriptError is not None

    def test_script_error_is_exception(self):
        """ScriptError is a subclass of Exception with message."""
        from autopilot.plan.script import ScriptError

        assert issubclass(ScriptError, Exception)
        err = ScriptError("test message")
        assert str(err) == "test message"

    def test_build_narrative_storyboard_signature(self):
        """build_narrative_storyboard has narrative_id and db params, returns str."""
        from autopilot.plan.script import build_narrative_storyboard

        sig = inspect.signature(build_narrative_storyboard)
        params = list(sig.parameters.keys())
        assert "narrative_id" in params
        assert "db" in params
        assert sig.return_annotation in (str, "str")

    def test_generate_script_signature(self):
        """generate_script has narrative_id, db, config params, returns dict."""
        from autopilot.plan.script import generate_script

        sig = inspect.signature(generate_script)
        params = list(sig.parameters.keys())
        assert "narrative_id" in params
        assert "db" in params
        assert "config" in params
        assert sig.return_annotation in (dict, "dict")

    def test_all_exports(self):
        """__all__ includes ScriptError, build_narrative_storyboard, generate_script."""
        from autopilot.plan import script

        assert "ScriptError" in script.__all__
        assert "build_narrative_storyboard" in script.__all__
        assert "generate_script" in script.__all__


# -- Step 3: build_narrative_storyboard basic tests ----------------------------


class TestBuildStoryboardBasic:
    """Tests for build_narrative_storyboard: not found and empty clusters."""

    def test_narrative_not_found_raises_script_error(self, catalog_db):
        """Raises ScriptError when narrative_id does not exist in DB."""
        from autopilot.plan.script import ScriptError, build_narrative_storyboard

        with pytest.raises(ScriptError, match="[Nn]arrative.*not found"):
            build_narrative_storyboard("nonexistent", catalog_db)

    def test_empty_cluster_list_returns_minimal_storyboard(self, catalog_db):
        """Narrative with no activity clusters returns minimal storyboard text."""
        from autopilot.plan.script import build_narrative_storyboard

        catalog_db.insert_narrative(
            "n1",
            title="Empty Narrative",
            activity_cluster_ids_json=json.dumps([]),
        )

        result = build_narrative_storyboard("n1", catalog_db)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention the narrative title or indicate it's empty
        assert "Empty Narrative" in result or "no" in result.lower()


# -- Step 5: build_narrative_storyboard with full data -------------------------


def _seed_full_storyboard_data(db):
    """Seed DB with complete data for storyboard assembly tests.

    Creates media files, transcripts, shot boundaries, detections, faces,
    audio events, captions, activity clusters, and a narrative.
    """
    # Media files
    db.insert_media(
        "v1", "/tmp/v1.mp4",
        duration_seconds=60.0, fps=30.0,
        created_at="2025-01-01T10:00:00",
    )
    db.insert_media(
        "v2", "/tmp/v2.mp4",
        duration_seconds=90.0, fps=30.0,
        created_at="2025-01-01T10:05:00",
    )

    # Shot boundaries for v1: two shots [0-30] and [30-60] seconds
    db.upsert_boundaries(
        "v1",
        json.dumps([
            {"start_frame": 0, "end_frame": 900, "start_time": 0.0, "end_time": 30.0},
            {"start_frame": 900, "end_frame": 1800, "start_time": 30.0, "end_time": 60.0},
        ]),
        "transnetv2",
    )

    # Shot boundaries for v2: single shot [0-90]
    db.upsert_boundaries(
        "v2",
        json.dumps([
            {"start_frame": 0, "end_frame": 2700, "start_time": 0.0, "end_time": 90.0},
        ]),
        "transnetv2",
    )

    # Transcripts
    db.upsert_transcript(
        "v1",
        json.dumps([
            {"text": "Look at the beautiful temple!", "start": 5.0, "end": 8.0},
            {"text": "The architecture is stunning.", "start": 35.0, "end": 38.0},
        ]),
        "en",
    )
    db.upsert_transcript(
        "v2",
        json.dumps([
            {"text": "The monks are chanting.", "start": 10.0, "end": 13.0},
        ]),
        "en",
    )

    # Captions (visual descriptions)
    db.upsert_caption("v1", 0.0, 30.0, "Wide shot of golden temple at sunrise", "qwen-vl")
    db.upsert_caption("v1", 30.0, 60.0, "Close-up of ornate door carvings", "qwen-vl")
    db.upsert_caption("v2", 0.0, 90.0, "Monks walking in procession", "qwen-vl")

    # YOLO detections
    db.batch_insert_detections([
        ("v1", 0, json.dumps([
            {"class": "person", "confidence": 0.95, "track_id": 1},
            {"class": "building", "confidence": 0.88},
        ])),
        ("v1", 900, json.dumps([
            {"class": "person", "confidence": 0.92, "track_id": 1},
        ])),
        ("v2", 0, json.dumps([
            {"class": "person", "confidence": 0.90, "track_id": 2},
            {"class": "person", "confidence": 0.85, "track_id": 3},
        ])),
    ])

    # Faces with clusters
    emb = np.zeros(512, dtype=np.float32).tobytes()
    db.batch_insert_faces([
        ("v1", 0, 0, json.dumps({"x": 10, "y": 10, "w": 50, "h": 50}), emb, 1),
        ("v2", 0, 0, json.dumps({"x": 20, "y": 20, "w": 60, "h": 60}), emb, 1),
        ("v2", 0, 1, json.dumps({"x": 80, "y": 20, "w": 60, "h": 60}), emb, 2),
    ])
    db.insert_face_cluster(1, label="Alice")
    db.insert_face_cluster(2, label="Bob")

    # Audio events
    db.batch_insert_audio_events([
        ("v1", 5.0, json.dumps([{"class": "Speech", "probability": 0.9}])),
        ("v1", 35.0, json.dumps([{"class": "Speech", "probability": 0.85}])),
        ("v2", 10.0, json.dumps([
            {"class": "Chanting", "probability": 0.88},
            {"class": "Music", "probability": 0.4},
        ])),
    ])

    # Activity cluster
    db.insert_activity_cluster(
        "c1",
        label="Temple visit",
        description="Exploring a Buddhist temple",
        time_start="2025-01-01T10:00:00",
        time_end="2025-01-01T10:10:00",
        clip_ids_json=json.dumps(["v1", "v2"]),
    )

    # Narrative referencing the cluster
    db.insert_narrative(
        "n1",
        title="A Morning at the Temple",
        description="Documentary of a peaceful morning temple visit",
        proposed_duration_seconds=600.0,
        activity_cluster_ids_json=json.dumps(["c1"]),
        arc_notes=json.dumps({
            "beginning": "Arrival at dawn",
            "middle": "Exploring the grounds",
            "end": "Quiet reflection",
        }),
        emotional_journey="Curiosity to wonder to peace",
    )


class TestBuildStoryboardFull:
    """Tests for build_narrative_storyboard with fully seeded data."""

    def test_contains_transcript_text(self, catalog_db):
        """Storyboard contains transcript text from clips."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_full_storyboard_data(catalog_db)
        result = build_narrative_storyboard("n1", catalog_db)

        assert "beautiful temple" in result
        assert "architecture is stunning" in result
        assert "monks are chanting" in result

    def test_contains_detected_objects(self, catalog_db):
        """Storyboard mentions detected objects (YOLO classes)."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_full_storyboard_data(catalog_db)
        result = build_narrative_storyboard("n1", catalog_db)

        assert "person" in result.lower()
        assert "building" in result.lower()

    def test_contains_people_labels(self, catalog_db):
        """Storyboard mentions face cluster labels."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_full_storyboard_data(catalog_db)
        result = build_narrative_storyboard("n1", catalog_db)

        assert "Alice" in result
        assert "Bob" in result

    def test_contains_audio_events(self, catalog_db):
        """Storyboard mentions audio event classes."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_full_storyboard_data(catalog_db)
        result = build_narrative_storyboard("n1", catalog_db)

        assert "Speech" in result
        assert "Chanting" in result

    def test_contains_duration_info(self, catalog_db):
        """Storyboard includes shot duration information."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_full_storyboard_data(catalog_db)
        result = build_narrative_storyboard("n1", catalog_db)

        # Should mention duration numbers from the shot boundaries
        assert "30" in result or "60" in result or "90" in result

    def test_contains_timestamps(self, catalog_db):
        """Storyboard includes source timestamps for shots."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_full_storyboard_data(catalog_db)
        result = build_narrative_storyboard("n1", catalog_db)

        # Should contain timecodes or time references
        assert "0.0" in result or "00:00" in result or "0s" in result


# -- Step 7: Resilience tests for build_narrative_storyboard -------------------


class TestBuildStoryboardResilience:
    """Tests for error handling in build_narrative_storyboard."""

    def test_corrupt_shot_boundaries_handled(self, catalog_db):
        """Corrupt shot boundaries JSON is handled gracefully."""
        from autopilot.plan.script import build_narrative_storyboard

        catalog_db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0)
        catalog_db.conn.execute(
            "INSERT INTO shot_boundaries (media_id, boundaries_json, method) "
            "VALUES (?, ?, ?)",
            ("v1", "NOT_VALID_JSON", "transnetv2"),
        )
        catalog_db.insert_activity_cluster(
            "c1", label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1", title="Test",
            activity_cluster_ids_json=json.dumps(["c1"]),
        )

        # Should not crash; clip treated as single shot
        result = build_narrative_storyboard("n1", catalog_db)
        assert isinstance(result, str)
        assert "Shot 1" in result

    def test_corrupt_transcript_handled(self, catalog_db):
        """Corrupt transcript JSON is skipped without crashing."""
        from autopilot.plan.script import build_narrative_storyboard

        catalog_db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0)
        catalog_db.conn.execute(
            "INSERT INTO transcripts (media_id, segments_json, language) "
            "VALUES (?, ?, ?)",
            ("v1", "CORRUPT", "en"),
        )
        catalog_db.insert_activity_cluster(
            "c1", label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1", title="Test",
            activity_cluster_ids_json=json.dumps(["c1"]),
        )

        result = build_narrative_storyboard("n1", catalog_db)
        assert isinstance(result, str)
        # Should still have the shot, just no transcript
        assert "Shot 1" in result

    def test_corrupt_detections_handled(self, catalog_db):
        """Corrupt detection JSON rows are skipped."""
        from autopilot.plan.script import build_narrative_storyboard

        catalog_db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0)
        catalog_db.conn.execute(
            "INSERT INTO detections (media_id, frame_number, detections_json) "
            "VALUES (?, ?, ?)",
            ("v1", 0, "BAD_JSON"),
        )
        catalog_db.insert_activity_cluster(
            "c1", label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1", title="Test",
            activity_cluster_ids_json=json.dumps(["c1"]),
        )

        result = build_narrative_storyboard("n1", catalog_db)
        assert isinstance(result, str)
        assert "Shot 1" in result

    def test_clip_without_shot_boundaries_treated_as_single_shot(self, catalog_db):
        """Clips with no shot boundary data are treated as a single shot."""
        from autopilot.plan.script import build_narrative_storyboard

        catalog_db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=45.0, fps=30.0)
        # No shot boundaries inserted
        catalog_db.insert_activity_cluster(
            "c1", label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1", title="Test",
            activity_cluster_ids_json=json.dumps(["c1"]),
        )

        result = build_narrative_storyboard("n1", catalog_db)
        assert isinstance(result, str)
        assert "Shot 1" in result
        # Duration should be full clip duration
        assert "45.0" in result

    def test_corrupt_audio_events_handled(self, catalog_db):
        """Corrupt audio events JSON is skipped."""
        from autopilot.plan.script import build_narrative_storyboard

        catalog_db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0)
        catalog_db.conn.execute(
            "INSERT INTO audio_events (media_id, timestamp_seconds, events_json) "
            "VALUES (?, ?, ?)",
            ("v1", 5.0, "INVALID"),
        )
        catalog_db.insert_activity_cluster(
            "c1", label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1", title="Test",
            activity_cluster_ids_json=json.dumps(["c1"]),
        )

        result = build_narrative_storyboard("n1", catalog_db)
        assert isinstance(result, str)
        assert "Shot 1" in result


# -- Helpers for generate_script tests -----------------------------------------


_SAMPLE_SCRIPT_JSON = {
    "scenes": [
        {
            "scene_number": 1,
            "description": "Opening shot of the temple",
            "estimated_duration_seconds": 8,
            "source_clips": [
                {"clip_id": "v1", "in_timecode": "00:00:00.000", "out_timecode": "00:00:08.000"}
            ],
            "voiceover_text": "The morning light paints the ancient walls...",
            "titles": [],
            "music_mood": "ambient, contemplative",
        },
    ],
    "broll_needs": [
        {
            "description": "Wide aerial of temple at sunrise",
            "duration_seconds": 5,
            "placement_after_scene": 1,
        },
    ],
    "quality_flags": [
        {
            "scene_number": 1,
            "issue": "Slight camera shake",
            "severity": "low",
            "suggestion": "Apply stabilization",
        },
    ],
}


def _make_script_llm_response(script_json=None):
    """Create a mock Anthropic API response for script generation."""
    if script_json is None:
        script_json = json.dumps(_SAMPLE_SCRIPT_JSON)
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = script_json
    mock_response.content = [mock_content]
    return mock_response


def _setup_mock_script_anthropic(script_data=None):
    """Create mock anthropic module and client for script tests."""
    mock_response = _make_script_llm_response(
        json.dumps(script_data) if script_data is not None else None
    )
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    return mock_anthropic, mock_client


def _seed_minimal_narrative(db):
    """Seed DB with minimal data for generate_script tests."""
    db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0)
    db.insert_activity_cluster(
        "c1", label="Test Activity",
        clip_ids_json=json.dumps(["v1"]),
    )
    db.insert_narrative(
        "n1",
        title="Test Narrative",
        description="A test narrative for scripting",
        activity_cluster_ids_json=json.dumps(["c1"]),
    )


# -- Step 9: generate_script LLM call tests -----------------------------------


class TestGenerateScriptLLM:
    """Tests for generate_script LLM interaction."""

    def test_uses_planning_model(self, catalog_db):
        """generate_script calls Anthropic API with planning_model."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_script_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_script("n1", catalog_db, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == config.planning_model

    def test_uses_script_writer_prompt(self, catalog_db):
        """System prompt is loaded from script_writer.md."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_script_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_script("n1", catalog_db, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        # script_writer.md starts with "# Professional Video Script Writer"
        assert "Script Writer" in call_kwargs["system"]

    def test_storyboard_and_description_in_user_message(self, catalog_db):
        """User message contains both storyboard and narrative description."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_script_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_script("n1", catalog_db, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        # Should contain storyboard content
        assert "L-Storyboard" in user_content or "Test Activity" in user_content
        # Should contain narrative description
        assert "test narrative" in user_content.lower()

    def test_narrative_not_found_raises_script_error(self, catalog_db):
        """generate_script raises ScriptError for missing narrative."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import ScriptError, generate_script

        config = LLMConfig()

        with pytest.raises(ScriptError, match="[Nn]arrative.*not found"):
            generate_script("nonexistent", catalog_db, config)


# -- Step 11: generate_script response parsing tests --------------------------


class TestGenerateScriptParsing:
    """Tests for JSON response parsing in generate_script."""

    def test_valid_json_returns_dict(self, catalog_db):
        """Valid JSON response returns dict with scenes/broll_needs/quality_flags."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        mock_anthropic, _ = _setup_mock_script_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = generate_script("n1", catalog_db, config)

        assert isinstance(result, dict)
        assert "scenes" in result
        assert "broll_needs" in result
        assert "quality_flags" in result
        assert len(result["scenes"]) == 1
        assert result["scenes"][0]["scene_number"] == 1

    def test_json_in_code_block_extracted(self, catalog_db):
        """JSON wrapped in ```json code block is correctly extracted."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        wrapped = f"Here's the script:\n```json\n{json.dumps(_SAMPLE_SCRIPT_JSON)}\n```"
        mock_anthropic, _ = _setup_mock_script_anthropic()
        # Override the response text
        mock_client = mock_anthropic.Anthropic.return_value
        mock_client.messages.create.return_value.content[0].text = wrapped

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = generate_script("n1", catalog_db, config)

        assert isinstance(result, dict)
        assert "scenes" in result

    def test_malformed_json_raises_script_error(self, catalog_db):
        """Malformed JSON in response raises ScriptError."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import ScriptError, generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_script_anthropic()
        mock_client.messages.create.return_value.content[0].text = "NOT VALID JSON {"

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with pytest.raises(ScriptError, match="parse"):
                generate_script("n1", catalog_db, config)

    def test_empty_llm_response_raises_script_error(self, catalog_db):
        """Empty LLM response (no content) raises ScriptError."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import ScriptError, generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_script_anthropic()
        mock_client.messages.create.return_value.content = []

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with pytest.raises(ScriptError, match="[Ee]mpty"):
                generate_script("n1", catalog_db, config)


# -- Step 13: DB persistence and error wrapping tests -------------------------


class TestGenerateScriptPersistence:
    """Tests for generate_script DB storage and status updates."""

    def test_script_stored_in_narrative_scripts(self, catalog_db):
        """Script JSON is stored in narrative_scripts table."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        mock_anthropic, _ = _setup_mock_script_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_script("n1", catalog_db, config)

        stored = catalog_db.get_narrative_script("n1")
        assert stored is not None
        parsed = json.loads(stored["script_json"])
        assert "scenes" in parsed
        assert parsed["scenes"][0]["scene_number"] == 1

    def test_narrative_status_updated_to_scripted(self, catalog_db):
        """Narrative status is updated to 'scripted' after script generation."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        mock_anthropic, _ = _setup_mock_script_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_script("n1", catalog_db, config)

        narrative = catalog_db.get_narrative("n1")
        assert narrative is not None
        assert narrative["status"] == "scripted"

    def test_api_error_wrapped_as_script_error(self, catalog_db):
        """API errors are wrapped as ScriptError with chained cause."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import ScriptError, generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_script_anthropic()
        mock_client.messages.create.side_effect = RuntimeError("Connection timeout")

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with pytest.raises(ScriptError, match="API.*failed") as exc_info:
                generate_script("n1", catalog_db, config)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)
