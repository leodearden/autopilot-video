"""Tests for EDL generation (autopilot.plan.edl)."""

from __future__ import annotations

import inspect
import json
from unittest.mock import patch

import pytest

from autopilot.llm import LlmError

# -- Step 13: Public API surface tests ----------------------------------------


class TestEdlPublicAPI:
    """Verify EdlError, generate_edl, and TOOL_DEFINITIONS surface."""

    def test_edl_error_importable(self):
        """EdlError is importable from edl module."""
        from autopilot.plan.edl import EdlError

        assert EdlError is not None

    def test_edl_error_is_exception(self):
        """EdlError is a subclass of Exception with message."""
        from autopilot.plan.edl import EdlError

        assert issubclass(EdlError, Exception)
        err = EdlError("test message")
        assert str(err) == "test message"

    def test_generate_edl_signature(self):
        """generate_edl has narrative_id, db, config params, returns dict."""
        from autopilot.plan.edl import generate_edl

        sig = inspect.signature(generate_edl)
        params = list(sig.parameters.keys())
        assert "narrative_id" in params
        assert "db" in params
        assert "config" in params
        assert sig.return_annotation in (dict, "dict")

    def test_all_exports(self):
        """__all__ includes EdlError, generate_edl, TOOL_DEFINITIONS."""
        from autopilot.plan import edl

        assert "EdlError" in edl.__all__
        assert "generate_edl" in edl.__all__
        assert "TOOL_DEFINITIONS" in edl.__all__


# -- Helpers for generate_edl tests -------------------------------------------


def _make_edl_dict(overrides=None):
    """Create a valid EDL dict as returned by invoke_claude structured output."""
    edl = {
        "clips": [
            {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            },
        ],
        "transitions": [],
        "crop_modes": [],
        "titles": [],
        "audio_settings": [],
        "music": [],
        "voiceovers": [],
        "broll_requests": [],
    }
    if overrides:
        edl.update(overrides)
    return edl


# EDL with overlapping clips for validation failure tests
_BAD_EDL_DICT = {
    "clips": [
        {
            "clip_id": "v1",
            "in_timecode": "00:00:00.000",
            "out_timecode": "00:00:10.000",
            "track": 1,
        },
        {
            "clip_id": "v1",
            "in_timecode": "00:00:05.000",
            "out_timecode": "00:00:15.000",
            "track": 1,
        },
    ],
    "transitions": [],
    "crop_modes": [],
    "titles": [],
    "audio_settings": [],
    "music": [],
    "voiceovers": [],
    "broll_requests": [],
}


def _seed_edl_narrative(db):
    """Seed DB with minimal data for generate_edl tests."""
    db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0)
    db.insert_activity_cluster(
        "c1",
        label="Test Activity",
        clip_ids_json=json.dumps(["v1"]),
    )
    db.insert_narrative(
        "n1",
        title="Test Narrative",
        description="A test narrative for EDL generation",
        proposed_duration_seconds=10.0,
        activity_cluster_ids_json=json.dumps(["c1"]),
    )
    # Also need a script (from scripting stage)
    db.upsert_narrative_script(
        "n1",
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_number": 1,
                        "description": "Opening shot",
                        "estimated_duration_seconds": 10,
                        "source_clips": [
                            {
                                "clip_id": "v1",
                                "in_timecode": "00:00:00.000",
                                "out_timecode": "00:00:10.000",
                            }
                        ],
                        "voiceover_text": "Welcome to the video",
                        "titles": [],
                        "music_mood": "ambient",
                    },
                ],
                "broll_needs": [],
                "quality_flags": [],
            }
        ),
    )


# -- Step 15: LLM interaction basic tests (via invoke_claude) -----------------


class TestGenerateEdlLLM:
    """Tests for generate_edl LLM interaction via invoke_claude structured output."""

    def test_uses_planning_model(self, catalog_db):
        """generate_edl calls invoke_claude with config.planning_model."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        with patch(
            "autopilot.plan.edl.invoke_claude", return_value=_make_edl_dict(),
        ) as mock_invoke:
            generate_edl("n1", catalog_db, config)

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["model"] == config.planning_model

    def test_passes_edit_planner_prompt(self, catalog_db):
        """System prompt is loaded from edit_planner.md."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        with patch(
            "autopilot.plan.edl.invoke_claude", return_value=_make_edl_dict(),
        ) as mock_invoke:
            generate_edl("n1", catalog_db, config)

        call_kwargs = mock_invoke.call_args[1]
        assert "EDL" in call_kwargs["system"] or "edit" in call_kwargs["system"].lower()

    def test_passes_json_schema(self, catalog_db):
        """invoke_claude is called with json_schema parameter for structured output."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import EDL_SCHEMA, generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        with patch(
            "autopilot.plan.edl.invoke_claude", return_value=_make_edl_dict(),
        ) as mock_invoke:
            generate_edl("n1", catalog_db, config)

        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["json_schema"] == EDL_SCHEMA

    def test_edl_schema_has_all_8_arrays(self):
        """EDL_SCHEMA defines all 8 EDL array fields."""
        from autopilot.plan.edl import EDL_SCHEMA

        assert isinstance(EDL_SCHEMA, dict)
        props = EDL_SCHEMA.get("properties", {})
        expected_keys = {
            "clips", "transitions", "crop_modes", "titles",
            "audio_settings", "music", "voiceovers", "broll_requests",
        }
        assert expected_keys.issubset(set(props.keys()))

    def test_prompt_includes_script_and_storyboard(self, catalog_db):
        """Prompt contains script and storyboard data."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        with patch(
            "autopilot.plan.edl.invoke_claude", return_value=_make_edl_dict(),
        ) as mock_invoke:
            generate_edl("n1", catalog_db, config)

        call_kwargs = mock_invoke.call_args[1]
        prompt = call_kwargs["prompt"]
        # Should contain script data
        assert "scene" in prompt.lower() or "opening" in prompt.lower()

    def test_narrative_not_found_raises_edl_error(self, catalog_db):
        """generate_edl raises EdlError for missing narrative."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import EdlError, generate_edl

        config = LLMConfig()

        with pytest.raises(EdlError, match="[Nn]arrative.*not found"):
            generate_edl("nonexistent", catalog_db, config)


# -- Step 17: Structured output EDL dict tests --------------------------------


class TestStructuredOutput:
    """Tests for EDL dict returned directly from invoke_claude structured output."""

    def test_clips_from_structured_output(self, catalog_db):
        """Clips array from structured output is used directly in EDL."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        edl_dict = _make_edl_dict()
        with patch("autopilot.plan.edl.invoke_claude", return_value=edl_dict):
            edl = generate_edl("n1", catalog_db, config)

        assert len(edl["clips"]) == 1
        assert edl["clips"][0]["clip_id"] == "v1"

    def test_transitions_from_structured_output(self, catalog_db):
        """Transitions from structured output are used directly."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        edl_dict = _make_edl_dict({
            "transitions": [{"type": "crossfade", "duration": 1.0, "position": "00:00:10.000"}],
        })
        with patch("autopilot.plan.edl.invoke_claude", return_value=edl_dict):
            edl = generate_edl("n1", catalog_db, config)

        assert len(edl["transitions"]) == 1
        assert edl["transitions"][0]["type"] == "crossfade"

    def test_audio_settings_from_structured_output(self, catalog_db):
        """Audio settings from structured output are used directly."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        edl_dict = _make_edl_dict({
            "audio_settings": [{"clip_id": "v1", "level_db": -6.0}],
        })
        with patch("autopilot.plan.edl.invoke_claude", return_value=edl_dict):
            edl = generate_edl("n1", catalog_db, config)

        assert len(edl["audio_settings"]) == 1
        assert edl["audio_settings"][0]["level_db"] == -6.0

    def test_all_8_edl_fields_from_structured_output(self, catalog_db):
        """All 8 EDL fields are populated from structured output dict."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        full_edl = {
            "clips": [
                {
                    "clip_id": "v1", "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000", "track": 1,
                },
            ],
            "transitions": [{"type": "cut", "duration": 0, "position": "00:00:10.000"}],
            "crop_modes": [{"clip_id": "v1", "mode": "center"}],
            "titles": [
                {
                    "text": "Title", "style": "lower_third",
                    "position": "00:00:02.000", "duration": 3.0,
                },
            ],
            "audio_settings": [{"clip_id": "v1", "level_db": -6.0}],
            "music": [{"mood": "ambient", "duration": 10.0, "start_time": "00:00:00.000"}],
            "voiceovers": [{"text": "Welcome", "start_time": "00:00:00.000", "duration": 5.0}],
            "broll_requests": [
                {"description": "Aerial shot", "duration": 3.0, "start_time": "00:00:05.000"},
            ],
        }
        with patch("autopilot.plan.edl.invoke_claude", return_value=full_edl):
            edl = generate_edl("n1", catalog_db, config)

        assert len(edl["clips"]) == 1
        assert len(edl["transitions"]) == 1
        assert len(edl["crop_modes"]) == 1
        assert len(edl["titles"]) == 1
        assert len(edl["audio_settings"]) == 1
        assert len(edl["music"]) == 1
        assert len(edl["voiceovers"]) == 1
        assert len(edl["broll_requests"]) == 1


# -- Step 19: DB persistence and status update tests --------------------------


class TestEdlPersistence:
    """Tests for generate_edl DB storage and status updates."""

    def test_edl_stored_in_edit_plans(self, catalog_db):
        """EDL JSON is stored in edit_plans table via upsert_edit_plan."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        with patch("autopilot.plan.edl.invoke_claude", return_value=_make_edl_dict()):
            generate_edl("n1", catalog_db, config)

        stored = catalog_db.get_edit_plan("n1")
        assert stored is not None
        edl_data = json.loads(stored["edl_json"])
        assert "clips" in edl_data
        assert len(edl_data["clips"]) >= 1

    def test_narrative_status_updated_to_planned(self, catalog_db):
        """Narrative status is updated to 'planned' after EDL generation."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        with patch("autopilot.plan.edl.invoke_claude", return_value=_make_edl_dict()):
            generate_edl("n1", catalog_db, config)

        narrative = catalog_db.get_narrative("n1")
        assert narrative is not None
        assert narrative["status"] == "planned"

    def test_validation_result_stored_in_edit_plans(self, catalog_db):
        """Validation result JSON is stored in edit_plans.validation_json."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        with patch("autopilot.plan.edl.invoke_claude", return_value=_make_edl_dict()):
            generate_edl("n1", catalog_db, config)

        stored = catalog_db.get_edit_plan("n1")
        assert stored is not None
        assert stored["validation_json"] is not None
        val_data = json.loads(stored["validation_json"])
        assert "passed" in val_data


# -- Step 21: Validation retry loop tests -------------------------------------


class TestEdlRetryLoop:
    """Tests for generate_edl validation retry loop with invoke_claude."""

    def test_validation_passes_first_try_no_retry(self, catalog_db):
        """Validation passes on first try — invoke_claude called only once."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        with patch(
            "autopilot.plan.edl.invoke_claude", return_value=_make_edl_dict(),
        ) as mock_invoke:
            generate_edl("n1", catalog_db, config)

        assert mock_invoke.call_count == 1

    def test_validation_fails_then_passes_on_retry(self, catalog_db):
        """Validation fails then passes — invoke_claude called twice."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        # First call: bad EDL (overlapping clips), second: good EDL
        with patch(
            "autopilot.plan.edl.invoke_claude",
            side_effect=[_BAD_EDL_DICT, _make_edl_dict()],
        ) as mock_invoke:
            edl = generate_edl("n1", catalog_db, config)

        assert mock_invoke.call_count == 2
        assert len(edl["clips"]) == 1  # good response

    def test_validation_fails_max_retries_raises_edl_error(self, catalog_db):
        """Validation fails after max retries — EdlError raised."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import EdlError, generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        # All calls return bad EDL
        with patch("autopilot.plan.edl.invoke_claude", return_value=_BAD_EDL_DICT) as mock_invoke:
            with pytest.raises(EdlError, match="[Vv]alidation"):
                generate_edl("n1", catalog_db, config)

        # Initial + 3 retries = 4 calls max
        assert mock_invoke.call_count <= 4

    def test_retry_prompt_includes_validation_errors(self, catalog_db):
        """On retry, the prompt includes previous validation errors."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        # First: bad, second: good
        with patch(
            "autopilot.plan.edl.invoke_claude",
            side_effect=[_BAD_EDL_DICT, _make_edl_dict()],
        ) as mock_invoke:
            generate_edl("n1", catalog_db, config)

        # Second call's prompt should include error feedback
        second_call_kwargs = mock_invoke.call_args_list[1][1]
        prompt = second_call_kwargs["prompt"]
        assert "error" in prompt.lower() or "overlap" in prompt.lower()

    def test_llm_error_wrapped_as_edl_error(self, catalog_db):
        """LlmError from invoke_claude is wrapped as EdlError."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import EdlError, generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        with patch("autopilot.plan.edl.invoke_claude", side_effect=LlmError("CLI timeout")):
            with pytest.raises(EdlError, match="LLM.*failed"):
                generate_edl("n1", catalog_db, config)


# -- Step 23: Integration test ------------------------------------------------


class TestEdlIntegration:
    """Full pipeline: seeded DB -> generate_edl with mocked invoke_claude -> verify."""

    def test_full_pipeline(self, catalog_db):
        """End-to-end: seed DB -> generate_edl -> verify EDL, DB state, validation."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        # --- Seed full DB data ---
        catalog_db.insert_media(
            "v1",
            "/tmp/v1.mp4",
            duration_seconds=60.0,
            fps=30.0,
            created_at="2025-01-01T10:00:00",
        )
        catalog_db.insert_media(
            "v2",
            "/tmp/v2.mp4",
            duration_seconds=90.0,
            fps=30.0,
            created_at="2025-01-01T10:05:00",
        )

        # Shot boundaries
        catalog_db.upsert_boundaries(
            "v1",
            json.dumps(
                [
                    {"start_frame": 0, "end_frame": 900, "start_time": 0.0, "end_time": 30.0},
                ]
            ),
            "transnetv2",
        )

        # Activity cluster + narrative
        catalog_db.insert_activity_cluster(
            "c1",
            label="Temple visit",
            clip_ids_json=json.dumps(["v1", "v2"]),
        )
        catalog_db.insert_narrative(
            "n1",
            title="Morning Temple",
            description="A peaceful morning visit",
            proposed_duration_seconds=25.0,
            activity_cluster_ids_json=json.dumps(["c1"]),
        )

        # Script (from prior stage)
        catalog_db.upsert_narrative_script(
            "n1",
            json.dumps(
                {
                    "scenes": [
                        {
                            "scene_number": 1,
                            "description": "Opening at temple",
                            "estimated_duration_seconds": 15,
                            "source_clips": [
                                {
                                    "clip_id": "v1",
                                    "in_timecode": "00:00:00.000",
                                    "out_timecode": "00:00:15.000",
                                },
                            ],
                            "voiceover_text": "Dawn breaks...",
                            "titles": [{"text": "Thailand", "style": "lower_third"}],
                            "music_mood": "ambient",
                        },
                        {
                            "scene_number": 2,
                            "description": "Monks walking",
                            "estimated_duration_seconds": 10,
                            "source_clips": [
                                {
                                    "clip_id": "v2",
                                    "in_timecode": "00:00:00.000",
                                    "out_timecode": "00:00:10.000",
                                },
                            ],
                            "voiceover_text": None,
                            "titles": [],
                            "music_mood": "sacred",
                        },
                    ],
                    "broll_needs": [],
                    "quality_flags": [],
                }
            ),
        )

        # --- Mock invoke_claude to return valid structured EDL ---
        full_edl = {
            "clips": [
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:15.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:15.000",
                    "out_timecode": "00:00:25.000",
                    "track": 1,
                },
            ],
            "transitions": [
                {
                    "type": "crossfade",
                    "duration": 1.0,
                    "position": "00:00:15.000",
                },
            ],
            "crop_modes": [],
            "titles": [
                {
                    "text": "Thailand",
                    "style": "lower_third",
                    "position": "00:00:02.000",
                    "duration": 4.0,
                },
            ],
            "audio_settings": [
                {
                    "clip_id": "v1",
                    "level_db": -6.0,
                    "fade_in": 0.5,
                },
                {
                    "clip_id": "v2",
                    "level_db": -12.0,
                },
            ],
            "music": [
                {
                    "mood": "ambient contemplative",
                    "duration": 25.0,
                    "start_time": "00:00:00.000",
                },
            ],
            "voiceovers": [
                {
                    "text": "Dawn breaks over ancient walls...",
                    "start_time": "00:00:00.000",
                    "duration": 8.0,
                },
            ],
            "broll_requests": [],
        }

        config = LLMConfig()

        with patch("autopilot.plan.edl.invoke_claude", return_value=full_edl) as mock_invoke:
            edl = generate_edl("n1", catalog_db, config)

        # --- Verify EDL structure ---
        assert isinstance(edl, dict)
        assert len(edl["clips"]) == 2
        assert edl["clips"][0]["clip_id"] == "v1"
        assert edl["clips"][1]["clip_id"] == "v2"
        assert len(edl["transitions"]) == 1
        assert len(edl["audio_settings"]) == 2
        assert len(edl["voiceovers"]) == 1
        assert len(edl["titles"]) == 1
        assert len(edl["music"]) == 1
        assert edl["target_duration_seconds"] == 25.0

        # --- Verify DB state ---
        # Edit plan stored
        stored = catalog_db.get_edit_plan("n1")
        assert stored is not None
        edl_data = json.loads(stored["edl_json"])
        assert len(edl_data["clips"]) == 2

        # Validation stored
        val_data = json.loads(stored["validation_json"])
        assert val_data["passed"] is True
        assert len(val_data["errors"]) == 0

        # Narrative status updated
        narrative = catalog_db.get_narrative("n1")
        assert narrative["status"] == "planned"

        # --- Verify invoke_claude called correctly ---
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["model"] == config.planning_model
        assert "json_schema" in call_kwargs
        # System prompt is edit_planner
        assert "EDL" in call_kwargs["system"]
        # Prompt contains script and storyboard
        assert "Temple" in call_kwargs["prompt"] or "temple" in call_kwargs["prompt"]
