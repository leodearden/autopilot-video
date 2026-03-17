"""Tests for autopilot/prompts/edit_planner.md prompt template."""

from pathlib import Path

from tests.conftest import extract_json_blocks

EXPECTED_TOOL_NAMES = {
    "select_clip",
    "add_transition",
    "set_crop_mode",
    "add_title",
    "set_audio",
    "add_music",
    "add_voiceover",
    "request_broll",
}


class TestEditPlannerPromptExists:
    """Verify the edit planner prompt file exists and has substantial content."""

    def test_file_exists(self, prompts_dir: Path):
        path = prompts_dir / "edit_planner.md"
        assert path.exists(), f"Expected prompt file at {path}"

    def test_file_is_non_empty(self, prompts_dir: Path):
        path = prompts_dir / "edit_planner.md"
        content = path.read_text()
        assert len(content) > 500, (
            f"edit_planner.md should have substantial content (>500 chars given "
            f"tool schemas), got {len(content)} chars"
        )


class TestEditPlannerToolDefinitions:
    """Verify tool definitions are valid and complete."""

    def _get_tools(self, prompts_dir: Path) -> list:
        content = (prompts_dir / "edit_planner.md").read_text()
        return extract_json_blocks(content)

    def test_exactly_8_tool_definitions(self, prompts_dir: Path):
        tools = self._get_tools(prompts_dir)
        assert len(tools) == 8, (
            f"Expected exactly 8 tool definitions, found {len(tools)}"
        )

    def test_each_tool_has_required_keys(self, prompts_dir: Path):
        tools = self._get_tools(prompts_dir)
        for tool in tools:
            assert "name" in tool, f"Tool missing 'name' key: {tool}"
            assert "description" in tool, (
                f"Tool '{tool.get('name', '?')}' missing 'description' key"
            )
            assert "input_schema" in tool, (
                f"Tool '{tool.get('name', '?')}' missing 'input_schema' key"
            )

    def test_tool_names_match_expected_set(self, prompts_dir: Path):
        tools = self._get_tools(prompts_dir)
        tool_names = {t["name"] for t in tools}
        assert tool_names == EXPECTED_TOOL_NAMES, (
            f"Tool names mismatch.\n"
            f"Expected: {EXPECTED_TOOL_NAMES}\n"
            f"Got: {tool_names}\n"
            f"Missing: {EXPECTED_TOOL_NAMES - tool_names}\n"
            f"Extra: {tool_names - EXPECTED_TOOL_NAMES}"
        )

    def test_each_input_schema_is_object_type(self, prompts_dir: Path):
        tools = self._get_tools(prompts_dir)
        for tool in tools:
            schema = tool["input_schema"]
            assert schema.get("type") == "object", (
                f"Tool '{tool['name']}' input_schema type should be 'object', "
                f"got '{schema.get('type')}'"
            )
            assert "properties" in schema and isinstance(schema["properties"], dict), (
                f"Tool '{tool['name']}' input_schema missing 'properties' dict"
            )


class TestEditPlannerToolParameters:
    """Verify key tool parameters are present."""

    def _get_tool(self, prompts_dir: Path, name: str) -> dict:
        content = (prompts_dir / "edit_planner.md").read_text()
        tools = extract_json_blocks(content)
        for t in tools:
            if t.get("name") == name:
                return t
        raise AssertionError(f"Tool '{name}' not found")

    def test_select_clip_has_required_properties(self, prompts_dir: Path):
        tool = self._get_tool(prompts_dir, "select_clip")
        props = tool["input_schema"]["properties"]
        for required in ["clip_id", "in_timecode", "out_timecode", "track"]:
            assert required in props, (
                f"select_clip missing required property '{required}'"
            )

    def test_set_crop_mode_has_required_properties(self, prompts_dir: Path):
        tool = self._get_tool(prompts_dir, "set_crop_mode")
        props = tool["input_schema"]["properties"]
        for required in ["clip_id", "mode"]:
            assert required in props, (
                f"set_crop_mode missing required property '{required}'"
            )

    def test_add_title_has_text_property(self, prompts_dir: Path):
        tool = self._get_tool(prompts_dir, "add_title")
        props = tool["input_schema"]["properties"]
        assert "text" in props, "add_title missing required property 'text'"

    def test_set_audio_has_required_properties(self, prompts_dir: Path):
        tool = self._get_tool(prompts_dir, "set_audio")
        props = tool["input_schema"]["properties"]
        for required in ["clip_id", "level_db"]:
            assert required in props, (
                f"set_audio missing required property '{required}'"
            )


class TestEditPlannerInstructions:
    """Verify the prompt contains instructions about timeline construction and validation."""

    def _read(self, prompts_dir: Path) -> str:
        return (prompts_dir / "edit_planner.md").read_text().lower()

    def test_references_overlap_validation(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "overlap" in content or "overlapping" in content, (
            "Prompt must reference overlap/overlapping for timeline validation"
        )

    def test_references_duration_validation(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "duration" in content, (
            "Prompt must reference duration for timeline validation"
        )

    def test_references_audio_level_validation(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "audio level" in content or "broadcast" in content, (
            "Prompt must reference audio level/broadcast-safe validation"
        )
