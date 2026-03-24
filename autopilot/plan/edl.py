"""EDL generation: Claude CLI structured-output edit decision list construction.

Provides generate_edl() for LLM-powered EDL construction using structured
JSON output via the Claude CLI, and TOOL_DEFINITIONS parsed from the
edit_planner prompt.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autopilot.llm import LlmError, invoke_claude

if TYPE_CHECKING:
    from autopilot.config import LLMConfig
    from autopilot.db import CatalogDB

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "edit_planner.md"

__all__ = [
    "EDL_SCHEMA",
    "EdlError",
    "TOOL_DEFINITIONS",
    "generate_edl",
]


class EdlError(Exception):
    """Raised for all EDL generation failures."""


def _parse_tool_definitions(prompt_path: Path) -> list[dict]:
    """Parse tool definitions from JSON blocks in edit_planner.md.

    Returns:
        List of tool definition dicts for the Anthropic API tools= parameter.
    """
    try:
        text = prompt_path.read_text()
    except OSError as e:
        logger.warning("Could not load edit_planner prompt: %s", e)
        return []

    pattern = r"```json\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    tools: list[dict] = []
    for match in matches:
        try:
            parsed = json.loads(match.strip())
            if isinstance(parsed, dict) and "name" in parsed:
                tools.append(parsed)
        except json.JSONDecodeError:
            continue
    return tools


TOOL_DEFINITIONS: list[dict] = _parse_tool_definitions(_PROMPT_PATH)


def _build_edl_schema(tools: list[dict]) -> dict[str, Any]:
    """Derive JSON schema for EDL structured output from tool definitions.

    Maps each tool's input_schema to the items schema for its corresponding
    EDL array (e.g. select_clip.input_schema → clips[].items).
    """
    _TOOL_TO_KEY = {
        "select_clip": "clips",
        "add_transition": "transitions",
        "set_crop_mode": "crop_modes",
        "add_title": "titles",
        "set_audio": "audio_settings",
        "add_music": "music",
        "add_voiceover": "voiceovers",
        "request_broll": "broll_requests",
    }

    properties: dict[str, Any] = {}
    for tool in tools:
        key = _TOOL_TO_KEY.get(tool.get("name", ""))
        if key is None:
            continue
        item_schema = tool.get("input_schema", {"type": "object"})
        properties[key] = {"type": "array", "items": item_schema}

    # Ensure all 8 keys exist even if some tools weren't parsed
    for key in _TOOL_TO_KEY.values():
        if key not in properties:
            properties[key] = {"type": "array", "items": {"type": "object"}}

    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
    }


EDL_SCHEMA: dict[str, Any] = _build_edl_schema(TOOL_DEFINITIONS)


def _build_user_message(narrative: dict, script_data: dict, storyboard: str) -> str:
    """Build user message with script and storyboard for the LLM."""
    title = str(narrative.get("title") or "")
    description = str(narrative.get("description") or "")
    duration = narrative.get("proposed_duration_seconds", 0)

    script_json = json.dumps(script_data, indent=2)

    return (
        f"## Narrative\n\n"
        f"**Title**: {title}\n"
        f"**Description**: {description}\n"
        f"**Target duration**: {duration} seconds\n\n"
        f"## Script\n\n```json\n{script_json}\n```\n\n"
        f"## Storyboard\n\n{storyboard}"
    )


def generate_edl(narrative_id: str, db: CatalogDB, config: LLMConfig) -> dict:
    """Generate an EDL for a narrative using Claude CLI structured output.

    Loads the script and storyboard, sends them with the edit_planner
    prompt to the LLM with a JSON schema, receives a structured EDL dict
    directly, validates, and stores in DB.

    Args:
        narrative_id: ID of the narrative to plan edits for.
        db: Catalog database instance.
        config: LLM configuration with planning_model.

    Returns:
        EDL dict with clips, transitions, audio_settings, etc.

    Raises:
        EdlError: If narrative/script not found, LLM fails, or
            validation fails after retries.
    """
    from autopilot.plan.script import build_narrative_storyboard

    # Load narrative
    narrative = db.get_narrative(narrative_id)
    if narrative is None:
        raise EdlError(f"Narrative not found: {narrative_id}")

    # Load script
    script_row = db.get_narrative_script(narrative_id)
    if script_row is None:
        raise EdlError(f"Script not found for narrative: {narrative_id}")

    try:
        script_data = json.loads(str(script_row["script_json"]))
    except (json.JSONDecodeError, KeyError) as e:
        raise EdlError(f"Corrupt script data: {e}") from e

    # Build storyboard
    try:
        storyboard = build_narrative_storyboard(narrative_id, db)
    except Exception as e:
        raise EdlError(f"Failed to build storyboard: {e}") from e

    # Load prompt
    try:
        system_prompt = _PROMPT_PATH.read_text()
    except OSError as e:
        raise EdlError(f"Failed to load edit_planner prompt: {e}") from e

    # Build user message
    user_message = _build_user_message(narrative, script_data, storyboard)

    from autopilot.plan.validator import validate_edl as _validate_edl

    target_duration = narrative.get("proposed_duration_seconds", 0)
    max_retries = 3
    prompt = user_message

    for attempt in range(1 + max_retries):
        # Call LLM with structured output
        try:
            edl = invoke_claude(
                prompt=prompt,
                system=system_prompt,
                model=config.planning_model,
                max_tokens=16384,
                json_schema=EDL_SCHEMA,
            )
        except LlmError as e:
            raise EdlError(f"LLM API call failed: {e}") from e

        assert isinstance(edl, dict)  # type guard: json_schema returns dict
        edl["target_duration_seconds"] = target_duration

        # Validate
        validation = _validate_edl(edl, db)

        if validation.passed:
            # Store and return
            validation_dict = {
                "passed": validation.passed,
                "errors": validation.errors,
                "warnings": validation.warnings,
            }
            with db:
                db.upsert_edit_plan(
                    narrative_id,
                    json.dumps(edl),
                    validation_json=json.dumps(validation_dict),
                )
                db.update_narrative_status(narrative_id, "planned")
            return edl

        # Validation failed — append error feedback to prompt for retry
        if attempt < max_retries:
            error_list = "\n".join(f"- {e}" for e in validation.errors)
            warning_list = "\n".join(f"- {w}" for w in validation.warnings)
            feedback = (
                "\n\nThe EDL you generated has validation errors. "
                "Please fix these issues and regenerate the EDL:\n\n"
                f"**Errors:**\n{error_list}"
            )
            if warning_list:
                feedback += f"\n\n**Warnings:**\n{warning_list}"

            prompt = user_message + feedback
            logger.info(
                "EDL validation failed (attempt %d/%d): %s",
                attempt + 1,
                1 + max_retries,
                validation.errors,
            )

    # All retries exhausted
    all_errors = "; ".join(validation.errors)
    raise EdlError(f"Validation failed after {1 + max_retries} attempts: {all_errors}")
