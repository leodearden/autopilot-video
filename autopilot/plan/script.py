"""Script generation: narrative storyboard assembly and scene-by-scene scripting.

Provides build_narrative_storyboard() for assembling structured text from
a narrative's activity clusters with per-shot data, and generate_script()
for LLM-powered scene-by-scene script creation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.config import LLMConfig
    from autopilot.db import CatalogDB

__all__ = [
    "ScriptError",
    "build_narrative_storyboard",
    "generate_script",
]


class ScriptError(Exception):
    """Raised for all script generation failures."""


def build_narrative_storyboard(narrative_id: str, db: CatalogDB) -> str:
    """Build a structured text storyboard for a narrative's activity clusters.

    Assembles per-shot data from the narrative's activity clusters including
    transcripts, visual descriptions, YOLO detections, faces, and audio events.

    Args:
        narrative_id: ID of the narrative to build the storyboard for.
        db: Catalog database instance.

    Returns:
        Structured text storyboard suitable for LLM consumption.

    Raises:
        ScriptError: If the narrative is not found.
    """
    raise NotImplementedError


def generate_script(narrative_id: str, db: CatalogDB, config: LLMConfig) -> dict:
    """Generate a scene-by-scene script for a narrative using Claude Opus.

    Builds the narrative storyboard, sends it with the narrative description
    to the LLM, parses the JSON response, stores it in the DB, and updates
    the narrative status to 'scripted'.

    Args:
        narrative_id: ID of the narrative to script.
        db: Catalog database instance.
        config: LLM configuration with planning_model.

    Returns:
        Parsed script dict with scenes, broll_needs, and quality_flags.

    Raises:
        ScriptError: If narrative not found, LLM call fails, or response is malformed.
    """
    raise NotImplementedError
