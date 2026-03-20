"""EDL generation: Claude Opus tool-use edit decision list construction.

Provides generate_edl() for LLM-powered EDL construction using tool-use
function calling, and TOOL_DEFINITIONS parsed from the edit_planner prompt.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.config import LLMConfig
    from autopilot.db import CatalogDB

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "edit_planner.md"

__all__ = [
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


def _collect_tool_calls(content_blocks: list) -> dict:
    """Collect tool_use content blocks into an EDL structure.

    Args:
        content_blocks: List of content blocks from the API response.

    Returns:
        EDL dict with categorized tool calls.
    """
    edl: dict = {
        "clips": [],
        "transitions": [],
        "crop_modes": [],
        "titles": [],
        "audio_settings": [],
        "music": [],
        "voiceovers": [],
        "broll_requests": [],
    }

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

    for block in content_blocks:
        if getattr(block, "type", None) != "tool_use":
            continue
        key = _TOOL_TO_KEY.get(block.name)
        if key is not None:
            edl[key].append(block.input)

    return edl


def _call_llm(
    user_message: str,
    system_prompt: str,
    config: LLMConfig,
    tools: list[dict],
) -> list:
    """Call Claude Opus with tool-use for EDL generation.

    Returns:
        List of content blocks from the response.

    Raises:
        EdlError: If the API call fails or response is empty.
    """
    import anthropic  # type: ignore[reportMissingImports]

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config.planning_model,
            max_tokens=16384,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            tools=tools,
        )
    except Exception as e:
        raise EdlError(f"LLM API call failed: {e}") from e

    if not response.content:
        raise EdlError("Empty response from LLM")

    return response.content


def generate_edl(narrative_id: str, db: CatalogDB, config: LLMConfig) -> dict:
    """Generate an EDL for a narrative using Claude Opus tool-use.

    Loads the script and storyboard, sends them with the edit_planner
    prompt to the LLM with tool definitions, collects tool_use response
    blocks into an EDL structure, validates, and stores in DB.

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
    storyboard = build_narrative_storyboard(narrative_id, db)

    # Load prompt
    try:
        system_prompt = _PROMPT_PATH.read_text()
    except OSError as e:
        raise EdlError(f"Failed to load edit_planner prompt: {e}") from e

    # Build user message
    user_message = _build_user_message(narrative, script_data, storyboard)

    # Call LLM
    content_blocks = _call_llm(user_message, system_prompt, config, TOOL_DEFINITIONS)

    # Collect tool calls into EDL
    edl = _collect_tool_calls(content_blocks)

    # Add target duration
    edl["target_duration_seconds"] = narrative.get(
        "proposed_duration_seconds", 0,
    )

    return edl
