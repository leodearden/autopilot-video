"""Tests for EDL generation (autopilot.plan.edl)."""

from __future__ import annotations

import inspect
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

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


def _make_tool_use_block(name: str, input_data: dict) -> MagicMock:
    """Create a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data
    return block


def _make_tool_use_response(tool_calls: list[tuple[str, dict]]) -> MagicMock:
    """Create a mock Anthropic response with tool_use content blocks."""
    blocks = [_make_tool_use_block(name, data) for name, data in tool_calls]
    mock_response = MagicMock()
    mock_response.content = blocks
    mock_response.stop_reason = "end_turn"
    return mock_response


def _setup_mock_edl_anthropic(tool_calls: list[tuple[str, dict]] | None = None):
    """Create mock anthropic module and client for EDL tests."""
    if tool_calls is None:
        tool_calls = [
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }),
        ]
    mock_response = _make_tool_use_response(tool_calls)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value = mock_client
    return mock_anthropic, mock_client


def _seed_edl_narrative(db):
    """Seed DB with minimal data for generate_edl tests."""
    db.insert_media("v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0)
    db.insert_activity_cluster(
        "c1", label="Test Activity",
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
    db.upsert_narrative_script("n1", json.dumps({
        "scenes": [
            {
                "scene_number": 1,
                "description": "Opening shot",
                "estimated_duration_seconds": 10,
                "source_clips": [
                    {"clip_id": "v1", "in_timecode": "00:00:00.000", "out_timecode": "00:00:10.000"}
                ],
                "voiceover_text": "Welcome to the video",
                "titles": [],
                "music_mood": "ambient",
            },
        ],
        "broll_needs": [],
        "quality_flags": [],
    }))


# -- Step 15: LLM interaction basic tests ------------------------------------


class TestGenerateEdlLLM:
    """Tests for generate_edl LLM interaction."""

    def test_uses_planning_model(self, catalog_db):
        """generate_edl calls Anthropic API with config.planning_model."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_edl_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_edl("n1", catalog_db, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == config.planning_model

    def test_passes_edit_planner_prompt(self, catalog_db):
        """System prompt is loaded from edit_planner.md."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_edl_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_edl("n1", catalog_db, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "EDL" in call_kwargs["system"] or "edit" in call_kwargs["system"].lower()

    def test_passes_tool_definitions(self, catalog_db):
        """Tool definitions are passed in the tools parameter."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_edl_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_edl("n1", catalog_db, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "select_clip" in tool_names

    def test_user_message_includes_script_and_storyboard(self, catalog_db):
        """User message contains script and storyboard data."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_edl_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_edl("n1", catalog_db, config)

        call_kwargs = mock_client.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        # Should contain script data
        assert "scene" in user_content.lower() or "opening" in user_content.lower()

    def test_narrative_not_found_raises_edl_error(self, catalog_db):
        """generate_edl raises EdlError for missing narrative."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import EdlError, generate_edl

        config = LLMConfig()

        with pytest.raises(EdlError, match="[Nn]arrative.*not found"):
            generate_edl("nonexistent", catalog_db, config)


# -- Step 17: Tool-use response collection tests ------------------------------


class TestToolUseCollection:
    """Tests for tool_use content block collection into EDL structure."""

    def test_select_clip_collected_into_clips(self, catalog_db):
        """select_clip tool calls are collected into clips array."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        tool_calls = [
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }),
        ]
        mock_anthropic, _ = _setup_mock_edl_anthropic(tool_calls)
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            edl = generate_edl("n1", catalog_db, config)

        assert len(edl["clips"]) == 1
        assert edl["clips"][0]["clip_id"] == "v1"

    def test_add_transition_collected_into_transitions(self, catalog_db):
        """add_transition tool calls are collected into transitions array."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        tool_calls = [
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }),
            ("add_transition", {
                "type": "crossfade",
                "duration": 1.0,
                "position": "00:00:10.000",
            }),
        ]
        mock_anthropic, _ = _setup_mock_edl_anthropic(tool_calls)
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            edl = generate_edl("n1", catalog_db, config)

        assert len(edl["transitions"]) == 1
        assert edl["transitions"][0]["type"] == "crossfade"

    def test_set_audio_collected_into_audio_settings(self, catalog_db):
        """set_audio tool calls are collected into audio_settings array."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        tool_calls = [
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }),
            ("set_audio", {
                "clip_id": "v1",
                "level_db": -6.0,
            }),
        ]
        mock_anthropic, _ = _setup_mock_edl_anthropic(tool_calls)
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            edl = generate_edl("n1", catalog_db, config)

        assert len(edl["audio_settings"]) == 1
        assert edl["audio_settings"][0]["level_db"] == -6.0

    def test_all_8_tool_types_categorized(self, catalog_db):
        """All 8 tool types are properly categorized in EDL dict."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        tool_calls = [
            ("select_clip", {
                "clip_id": "v1", "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000", "track": 1,
            }),
            ("add_transition", {
                "type": "cut", "duration": 0, "position": "00:00:10.000",
            }),
            ("set_crop_mode", {
                "clip_id": "v1", "mode": "center",
            }),
            ("add_title", {
                "text": "Title", "style": "lower_third",
                "position": "00:00:02.000", "duration": 3.0,
            }),
            ("set_audio", {
                "clip_id": "v1", "level_db": -6.0,
            }),
            ("add_music", {
                "mood": "ambient", "duration": 10.0, "start_time": "00:00:00.000",
            }),
            ("add_voiceover", {
                "text": "Welcome", "start_time": "00:00:00.000", "duration": 5.0,
            }),
            ("request_broll", {
                "description": "Aerial shot", "duration": 3.0,
                "start_time": "00:00:05.000",
            }),
        ]
        mock_anthropic, _ = _setup_mock_edl_anthropic(tool_calls)
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
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

        mock_anthropic, _ = _setup_mock_edl_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
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

        mock_anthropic, _ = _setup_mock_edl_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
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

        mock_anthropic, _ = _setup_mock_edl_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_edl("n1", catalog_db, config)

        stored = catalog_db.get_edit_plan("n1")
        assert stored is not None
        assert stored["validation_json"] is not None
        val_data = json.loads(stored["validation_json"])
        assert "passed" in val_data


# -- Step 21: Validation retry loop tests -------------------------------------


class TestEdlRetryLoop:
    """Tests for generate_edl validation retry loop."""

    def test_validation_passes_first_try_no_retry(self, catalog_db):
        """Validation passes on first try — LLM called only once."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        mock_anthropic, mock_client = _setup_mock_edl_anthropic()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_edl("n1", catalog_db, config)

        assert mock_client.messages.create.call_count == 1

    def test_validation_fails_then_passes_on_retry(self, catalog_db):
        """Validation fails then passes on retry — LLM called twice."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        # First response: overlapping clips (invalid)
        bad_response = _make_tool_use_response([
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }),
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:05.000",
                "out_timecode": "00:00:15.000",
                "track": 1,
            }),
        ])
        # Second response: valid (non-overlapping)
        good_response = _make_tool_use_response([
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }),
        ])

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [bad_response, good_response]
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            edl = generate_edl("n1", catalog_db, config)

        assert mock_client.messages.create.call_count == 2
        assert len(edl["clips"]) == 1  # good response

    def test_validation_fails_3_times_raises_edl_error(self, catalog_db):
        """Validation fails 3 times — EdlError raised with validation errors."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import EdlError, generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        # All responses: overlapping clips (invalid)
        bad_response = _make_tool_use_response([
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }),
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:05.000",
                "out_timecode": "00:00:15.000",
                "track": 1,
            }),
        ])

        mock_client = MagicMock()
        mock_client.messages.create.return_value = bad_response
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            with pytest.raises(EdlError, match="[Vv]alidation"):
                generate_edl("n1", catalog_db, config)

        # Initial + 3 retries = 4 calls max
        assert mock_client.messages.create.call_count <= 4

    def test_retry_message_includes_validation_errors(self, catalog_db):
        """Retry message includes previous validation errors for context."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        config = LLMConfig()
        _seed_edl_narrative(catalog_db)

        # First: bad, second: good
        bad_response = _make_tool_use_response([
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }),
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:05.000",
                "out_timecode": "00:00:15.000",
                "track": 1,
            }),
        ])
        good_response = _make_tool_use_response([
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            }),
        ])

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [bad_response, good_response]
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            generate_edl("n1", catalog_db, config)

        # The second call's messages should include error feedback
        second_call = mock_client.messages.create.call_args_list[1]
        messages = second_call[1]["messages"]
        # Should have more than just the initial user message
        assert len(messages) >= 2
        # Check that validation errors are mentioned in the feedback
        feedback_text = str(messages)
        assert "overlap" in feedback_text.lower() or "error" in feedback_text.lower()


# -- Step 23: Integration test ------------------------------------------------


class TestEdlIntegration:
    """Full pipeline: seeded DB -> generate_edl with mocked LLM -> verify."""

    def test_full_pipeline(self, catalog_db):
        """End-to-end: seed DB -> generate_edl -> verify EDL, DB state, validation."""
        from autopilot.config import LLMConfig
        from autopilot.plan.edl import generate_edl

        # --- Seed full DB data ---
        catalog_db.insert_media(
            "v1", "/tmp/v1.mp4", duration_seconds=60.0, fps=30.0,
            created_at="2025-01-01T10:00:00",
        )
        catalog_db.insert_media(
            "v2", "/tmp/v2.mp4", duration_seconds=90.0, fps=30.0,
            created_at="2025-01-01T10:05:00",
        )

        # Shot boundaries
        catalog_db.upsert_boundaries(
            "v1",
            json.dumps([
                {"start_frame": 0, "end_frame": 900,
                 "start_time": 0.0, "end_time": 30.0},
            ]),
            "transnetv2",
        )

        # Activity cluster + narrative
        catalog_db.insert_activity_cluster(
            "c1", label="Temple visit",
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
        catalog_db.upsert_narrative_script("n1", json.dumps({
            "scenes": [
                {
                    "scene_number": 1,
                    "description": "Opening at temple",
                    "estimated_duration_seconds": 15,
                    "source_clips": [
                        {"clip_id": "v1", "in_timecode": "00:00:00.000",
                         "out_timecode": "00:00:15.000"},
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
                        {"clip_id": "v2", "in_timecode": "00:00:00.000",
                         "out_timecode": "00:00:10.000"},
                    ],
                    "voiceover_text": None,
                    "titles": [],
                    "music_mood": "sacred",
                },
            ],
            "broll_needs": [],
            "quality_flags": [],
        }))

        # --- Mock LLM to return valid tool-use response ---
        tool_calls = [
            ("select_clip", {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:15.000",
                "track": 1,
            }),
            ("select_clip", {
                "clip_id": "v2",
                "in_timecode": "00:00:15.000",
                "out_timecode": "00:00:25.000",
                "track": 1,
            }),
            ("add_transition", {
                "type": "crossfade",
                "duration": 1.0,
                "position": "00:00:15.000",
            }),
            ("set_audio", {
                "clip_id": "v1",
                "level_db": -6.0,
                "fade_in": 0.5,
            }),
            ("set_audio", {
                "clip_id": "v2",
                "level_db": -12.0,
            }),
            ("add_voiceover", {
                "text": "Dawn breaks over ancient walls...",
                "start_time": "00:00:00.000",
                "duration": 8.0,
            }),
            ("add_title", {
                "text": "Thailand",
                "style": "lower_third",
                "position": "00:00:02.000",
                "duration": 4.0,
            }),
            ("add_music", {
                "mood": "ambient contemplative",
                "duration": 25.0,
                "start_time": "00:00:00.000",
            }),
        ]

        config = LLMConfig()
        mock_anthropic, mock_client = _setup_mock_edl_anthropic(tool_calls)

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
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

        # --- Verify LLM called correctly ---
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == config.planning_model
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 8
        # System prompt is edit_planner
        assert "EDL" in call_kwargs["system"]
        # User message contains script and storyboard
        user_msg = call_kwargs["messages"][0]["content"]
        assert "Temple" in user_msg or "temple" in user_msg
