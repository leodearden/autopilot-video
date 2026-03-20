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
    raise EdlError("Not implemented")
