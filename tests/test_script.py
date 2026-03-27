"""Tests for script generation (autopilot.plan.script) and narrative_scripts DB CRUD."""

from __future__ import annotations

import inspect
import json
from unittest.mock import patch

import numpy as np
import pytest

from autopilot.llm import LlmError

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
        "v1",
        "/tmp/v1.mp4",
        duration_seconds=60.0,
        fps=30.0,
        created_at="2025-01-01T10:00:00",
    )
    db.insert_media(
        "v2",
        "/tmp/v2.mp4",
        duration_seconds=90.0,
        fps=30.0,
        created_at="2025-01-01T10:05:00",
    )

    # Shot boundaries for v1: two shots [0-30] and [30-60] seconds
    db.upsert_boundaries(
        "v1",
        json.dumps(
            [
                {"start_frame": 0, "end_frame": 900, "start_time": 0.0, "end_time": 30.0},
                {"start_frame": 900, "end_frame": 1800, "start_time": 30.0, "end_time": 60.0},
            ]
        ),
        "transnetv2",
    )

    # Shot boundaries for v2: single shot [0-90]
    db.upsert_boundaries(
        "v2",
        json.dumps(
            [
                {"start_frame": 0, "end_frame": 2700, "start_time": 0.0, "end_time": 90.0},
            ]
        ),
        "transnetv2",
    )

    # Transcripts
    db.upsert_transcript(
        "v1",
        json.dumps(
            [
                {"text": "Look at the beautiful temple!", "start": 5.0, "end": 8.0},
                {"text": "The architecture is stunning.", "start": 35.0, "end": 38.0},
            ]
        ),
        "en",
    )
    db.upsert_transcript(
        "v2",
        json.dumps(
            [
                {"text": "The monks are chanting.", "start": 10.0, "end": 13.0},
            ]
        ),
        "en",
    )

    # Captions (visual descriptions)
    db.upsert_caption("v1", 0.0, 30.0, "Wide shot of golden temple at sunrise", "qwen-vl")
    db.upsert_caption("v1", 30.0, 60.0, "Close-up of ornate door carvings", "qwen-vl")
    db.upsert_caption("v2", 0.0, 90.0, "Monks walking in procession", "qwen-vl")

    # YOLO detections
    db.batch_insert_detections(
        [
            (
                "v1",
                0,
                json.dumps(
                    [
                        {"class": "person", "confidence": 0.95, "track_id": 1},
                        {"class": "building", "confidence": 0.88},
                    ]
                ),
            ),
            (
                "v1",
                900,
                json.dumps(
                    [
                        {"class": "person", "confidence": 0.92, "track_id": 1},
                    ]
                ),
            ),
            (
                "v2",
                0,
                json.dumps(
                    [
                        {"class": "person", "confidence": 0.90, "track_id": 2},
                        {"class": "person", "confidence": 0.85, "track_id": 3},
                    ]
                ),
            ),
        ]
    )

    # Faces with clusters
    emb = np.zeros(512, dtype=np.float32).tobytes()
    db.batch_insert_faces(
        [
            ("v1", 0, 0, json.dumps({"x": 10, "y": 10, "w": 50, "h": 50}), emb, 1),
            ("v2", 0, 0, json.dumps({"x": 20, "y": 20, "w": 60, "h": 60}), emb, 1),
            ("v2", 0, 1, json.dumps({"x": 80, "y": 20, "w": 60, "h": 60}), emb, 2),
        ]
    )
    db.insert_face_cluster(1, label="Alice")
    db.insert_face_cluster(2, label="Bob")

    # Audio events
    db.batch_insert_audio_events(
        [
            ("v1", 5.0, json.dumps([{"class": "Speech", "probability": 0.9}])),
            ("v1", 35.0, json.dumps([{"class": "Speech", "probability": 0.85}])),
            (
                "v2",
                10.0,
                json.dumps(
                    [
                        {"class": "Chanting", "probability": 0.88},
                        {"class": "Music", "probability": 0.4},
                    ]
                ),
            ),
        ]
    )

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
        arc_notes=json.dumps(
            {
                "beginning": "Arrival at dawn",
                "middle": "Exploring the grounds",
                "end": "Quiet reflection",
            }
        ),
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
            "INSERT INTO shot_boundaries (media_id, boundaries_json, method) VALUES (?, ?, ?)",
            ("v1", "NOT_VALID_JSON", "transnetv2"),
        )
        catalog_db.insert_activity_cluster(
            "c1",
            label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1",
            title="Test",
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
            "INSERT INTO transcripts (media_id, segments_json, language) VALUES (?, ?, ?)",
            ("v1", "CORRUPT", "en"),
        )
        catalog_db.insert_activity_cluster(
            "c1",
            label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1",
            title="Test",
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
            "INSERT INTO detections (media_id, frame_number, detections_json) VALUES (?, ?, ?)",
            ("v1", 0, "BAD_JSON"),
        )
        catalog_db.insert_activity_cluster(
            "c1",
            label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1",
            title="Test",
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
            "c1",
            label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1",
            title="Test",
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
            "INSERT INTO audio_events (media_id, timestamp_seconds, events_json) VALUES (?, ?, ?)",
            ("v1", 5.0, "INVALID"),
        )
        catalog_db.insert_activity_cluster(
            "c1",
            label="Test",
            clip_ids_json=json.dumps(["v1"]),
        )
        catalog_db.insert_narrative(
            "n1",
            title="Test",
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


def _make_script_llm_text(script_data=None):
    """Create a mock invoke_claude text response for script generation."""
    if script_data is not None:
        return json.dumps(script_data)
    return json.dumps(_SAMPLE_SCRIPT_JSON)


def _seed_minimal_narrative(db):
    """Seed DB with minimal data for generate_script tests."""
    db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0)
    db.insert_activity_cluster(
        "c1",
        label="Test Activity",
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
    """Tests for generate_script LLM interaction via invoke_claude."""

    def test_uses_planning_model(self, catalog_db):
        """generate_script calls invoke_claude with planning_model."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        with patch(
            "autopilot.plan.script.invoke_claude",
            return_value=_make_script_llm_text(),
        ) as mock_invoke:
            generate_script("n1", catalog_db, config)

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["model"] == config.planning_model

    def test_uses_script_writer_prompt(self, catalog_db):
        """System prompt is loaded from script_writer.md."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        with patch(
            "autopilot.plan.script.invoke_claude",
            return_value=_make_script_llm_text(),
        ) as mock_invoke:
            generate_script("n1", catalog_db, config)

        call_kwargs = mock_invoke.call_args[1]
        # script_writer.md starts with "# Professional Video Script Writer"
        assert "Script Writer" in call_kwargs["system"]

    def test_storyboard_and_description_in_user_message(self, catalog_db):
        """User message contains both storyboard and narrative description."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        with patch(
            "autopilot.plan.script.invoke_claude",
            return_value=_make_script_llm_text(),
        ) as mock_invoke:
            generate_script("n1", catalog_db, config)

        call_kwargs = mock_invoke.call_args[1]
        user_content = call_kwargs["prompt"]
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

        with patch("autopilot.plan.script.invoke_claude", return_value=_make_script_llm_text()):
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

        with patch("autopilot.plan.script.invoke_claude", return_value=wrapped):
            result = generate_script("n1", catalog_db, config)

        assert isinstance(result, dict)
        assert "scenes" in result

    def test_malformed_json_raises_script_error(self, catalog_db):
        """Malformed JSON in response raises ScriptError."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import ScriptError, generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        with patch("autopilot.plan.script.invoke_claude", return_value="NOT VALID JSON {"):
            with pytest.raises(ScriptError, match="parse"):
                generate_script("n1", catalog_db, config)

    def test_empty_llm_response_raises_script_error(self, catalog_db):
        """Empty/failed LLM response raises ScriptError."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import ScriptError, generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        with patch("autopilot.plan.script.invoke_claude", side_effect=LlmError("Empty response")):
            with pytest.raises(ScriptError, match="LLM.*failed"):
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

        with patch("autopilot.plan.script.invoke_claude", return_value=_make_script_llm_text()):
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

        with patch("autopilot.plan.script.invoke_claude", return_value=_make_script_llm_text()):
            generate_script("n1", catalog_db, config)

        narrative = catalog_db.get_narrative("n1")
        assert narrative is not None
        assert narrative["status"] == "scripted"

    def test_api_error_wrapped_as_script_error(self, catalog_db):
        """LLM errors are wrapped as ScriptError with chained cause."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import ScriptError, generate_script

        config = LLMConfig()
        _seed_minimal_narrative(catalog_db)

        with patch(
            "autopilot.plan.script.invoke_claude",
            side_effect=LlmError("Connection timeout"),
        ):
            with pytest.raises(ScriptError, match="LLM.*failed") as exc_info:
                generate_script("n1", catalog_db, config)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, LlmError)


# -- Step 15: Integration test -------------------------------------------------


class TestIntegration:
    """Full pipeline: seed DB -> storyboard -> generate_script -> verify DB."""

    def test_full_pipeline(self, catalog_db):
        """End-to-end: seeded DB -> build storyboard -> generate script -> verify."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import build_narrative_storyboard, generate_script

        # --- Seed full DB data ---
        _seed_full_storyboard_data(catalog_db)

        # --- Phase 1: Build storyboard ---
        storyboard = build_narrative_storyboard("n1", catalog_db)

        assert isinstance(storyboard, str)
        assert "L-Storyboard" in storyboard
        assert "Temple visit" in storyboard
        assert "Alice" in storyboard
        assert "Bob" in storyboard
        assert "beautiful temple" in storyboard
        assert "Speech" in storyboard
        assert "Chanting" in storyboard

        # Verify shot segmentation (v1 has 2 shots, v2 has 1)
        assert "Shot 1" in storyboard
        assert "Shot 2" in storyboard
        assert "Shot 3" in storyboard

        # --- Phase 2: Generate script (mocked LLM) ---
        config = LLMConfig()

        script_response = {
            "scenes": [
                {
                    "scene_number": 1,
                    "description": "Opening at the temple at dawn",
                    "estimated_duration_seconds": 10,
                    "source_clips": [
                        {
                            "clip_id": "v1",
                            "in_timecode": "00:00:00.000",
                            "out_timecode": "00:00:10.000",
                        }
                    ],
                    "voiceover_text": "Dawn breaks over ancient walls...",
                    "titles": [
                        {
                            "text": "Chiang Mai, Thailand",
                            "style": "lower_third",
                            "display_at_seconds": 2.0,
                            "duration_seconds": 4.0,
                        }
                    ],
                    "music_mood": "ambient, contemplative",
                },
                {
                    "scene_number": 2,
                    "description": "Monks walking in morning procession",
                    "estimated_duration_seconds": 15,
                    "source_clips": [
                        {
                            "clip_id": "v2",
                            "in_timecode": "00:00:00.000",
                            "out_timecode": "00:00:15.000",
                        }
                    ],
                    "voiceover_text": None,
                    "titles": [],
                    "music_mood": "sacred, reverent",
                },
            ],
            "broll_needs": [
                {
                    "description": "Wide aerial of temple complex",
                    "duration_seconds": 5,
                    "placement_after_scene": 1,
                },
            ],
            "quality_flags": [
                {
                    "scene_number": 2,
                    "issue": "Slight underexposure in morning light",
                    "severity": "low",
                    "suggestion": "Brighten shadows in color grade",
                },
            ],
        }

        with patch(
            "autopilot.plan.script.invoke_claude",
            return_value=_make_script_llm_text(script_response),
        ) as mock_invoke:
            result = generate_script("n1", catalog_db, config)

        # --- Verify return value ---
        assert isinstance(result, dict)
        assert len(result["scenes"]) == 2
        assert result["scenes"][0]["description"] == "Opening at the temple at dawn"
        assert result["scenes"][1]["description"] == "Monks walking in morning procession"
        assert len(result["broll_needs"]) == 1
        assert len(result["quality_flags"]) == 1

        # --- Verify DB state ---
        # Script stored
        stored = catalog_db.get_narrative_script("n1")
        assert stored is not None
        stored_data = json.loads(stored["script_json"])
        assert len(stored_data["scenes"]) == 2

        # Narrative status updated
        narrative = catalog_db.get_narrative("n1")
        assert narrative["status"] == "scripted"

        # --- Verify LLM was called correctly ---
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["model"] == config.planning_model
        # Storyboard content was in the prompt
        assert "Temple visit" in call_kwargs["prompt"]
        # System prompt was script_writer.md
        assert "Script Writer" in call_kwargs["system"]


# -- Step 17: NULL field resilience tests --------------------------------------


def _seed_null_resilience_base(db):
    """Seed minimal DB data for NULL resilience tests."""
    db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0)
    db.upsert_boundaries(
        "v1",
        json.dumps(
            [
                {"start_frame": 0, "end_frame": 1800, "start_time": 0.0, "end_time": 60.0},
            ]
        ),
        "transnetv2",
    )
    db.insert_activity_cluster(
        "c1",
        label="Test",
        clip_ids_json=json.dumps(["v1"]),
    )
    db.insert_narrative(
        "n1",
        title="Test",
        activity_cluster_ids_json=json.dumps(["c1"]),
    )


class TestNullFieldResilience:
    """Tests for NULL primary key field handling in storyboard assembly.

    SQLite allows NULL in composite primary key columns. These tests verify
    that rows with NULL in key numeric fields are silently skipped rather
    than causing TypeError when float()/int() is called on None.
    """

    def test_captions_with_null_start_or_end_time_skipped(self, catalog_db):
        """Caption rows with NULL start_time or end_time are skipped, valid ones kept."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_null_resilience_base(catalog_db)

        # Insert a valid caption
        catalog_db.upsert_caption("v1", 0.0, 30.0, "Valid caption text", "qwen-vl")
        # Insert captions with NULL start_time and end_time via raw SQL
        catalog_db.conn.execute(
            "INSERT INTO captions (media_id, start_time, end_time, caption, model_name) "
            "VALUES (?, NULL, ?, ?, ?)",
            ("v1", 30.0, "Null start caption", "qwen-vl"),
        )
        catalog_db.conn.execute(
            "INSERT INTO captions (media_id, start_time, end_time, caption, model_name) "
            "VALUES (?, ?, NULL, ?, ?)",
            ("v1", 31.0, "Null end caption", "qwen-vl"),
        )

        # Should NOT raise TypeError; valid caption appears in output
        result = build_narrative_storyboard("n1", catalog_db)
        assert "Valid caption text" in result
        assert "Null start caption" not in result
        assert "Null end caption" not in result

    def test_detections_with_null_frame_number_skipped(self, catalog_db):
        """Detection rows with NULL frame_number are skipped, valid ones kept."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_null_resilience_base(catalog_db)

        # Insert valid detection
        catalog_db.batch_insert_detections(
            [
                ("v1", 0, json.dumps([{"class": "car", "confidence": 0.9}])),
            ]
        )
        # Insert detection with NULL frame_number via raw SQL
        catalog_db.conn.execute(
            "INSERT INTO detections (media_id, frame_number, detections_json) VALUES (?, NULL, ?)",
            ("v1", json.dumps([{"class": "ghost_object", "confidence": 0.5}])),
        )

        result = build_narrative_storyboard("n1", catalog_db)
        assert "car" in result
        assert "ghost_object" not in result

    def test_faces_with_null_frame_number_skipped(self, catalog_db):
        """Face rows with NULL frame_number are skipped, valid ones kept."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_null_resilience_base(catalog_db)

        emb = np.zeros(512, dtype=np.float32).tobytes()
        # Insert valid face
        catalog_db.batch_insert_faces(
            [
                ("v1", 0, 0, json.dumps({"x": 10, "y": 10, "w": 50, "h": 50}), emb, 1),
            ]
        )
        catalog_db.insert_face_cluster(1, label="ValidPerson")
        # Insert face with NULL frame_number via raw SQL
        catalog_db.conn.execute(
            "INSERT INTO faces (media_id, frame_number, face_index,"
            " bbox_json, embedding, cluster_id) "
            "VALUES (?, NULL, ?, ?, ?, ?)",
            ("v1", 0, json.dumps({"x": 0, "y": 0, "w": 1, "h": 1}), emb, 2),
        )
        catalog_db.insert_face_cluster(2, label="GhostPerson")

        result = build_narrative_storyboard("n1", catalog_db)
        assert "ValidPerson" in result
        assert "GhostPerson" not in result

    def test_audio_events_with_null_timestamp_skipped(self, catalog_db):
        """Audio event rows with NULL timestamp_seconds are skipped, valid ones kept."""
        from autopilot.plan.script import build_narrative_storyboard

        _seed_null_resilience_base(catalog_db)

        # Insert valid audio event
        catalog_db.batch_insert_audio_events(
            [
                ("v1", 5.0, json.dumps([{"class": "Speech", "probability": 0.9}])),
            ]
        )
        # Insert audio event with NULL timestamp_seconds via raw SQL
        catalog_db.conn.execute(
            "INSERT INTO audio_events (media_id, timestamp_seconds, events_json) "
            "VALUES (?, NULL, ?)",
            ("v1", json.dumps([{"class": "GhostSound", "probability": 0.8}])),
        )

        result = build_narrative_storyboard("n1", catalog_db)
        assert "Speech" in result
        assert "GhostSound" not in result
