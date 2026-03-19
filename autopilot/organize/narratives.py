"""Narrative organization: storyboard construction and narrative proposals.

Provides build_master_storyboard() for assembling structured text from
activity clusters, and propose_narratives() for LLM-powered narrative
planning with format_for_review() for human checkpoint display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.config import AutopilotConfig
    from autopilot.db import CatalogDB

__all__ = [
    "Narrative",
    "NarrativeError",
    "build_master_storyboard",
    "format_for_review",
    "propose_narratives",
]

logger = logging.getLogger(__name__)


class NarrativeError(Exception):
    """Raised for all narrative organization failures."""


@dataclass
class Narrative:
    """A proposed video narrative."""

    narrative_id: str = ""
    title: str = ""
    description: str = ""
    proposed_duration_seconds: float = 0.0
    activity_cluster_ids: list[str] = field(default_factory=list)
    arc: dict[str, str] = field(default_factory=dict)
    emotional_journey: str = ""
    reasoning: str = ""
    status: str = "proposed"


def build_master_storyboard(db: CatalogDB) -> str:
    """Build a structured text storyboard from all activity clusters.

    Iterates over activity clusters in the database, enriches each with
    signal data (transcripts, detections, audio events, faces), and
    formats everything into structured text suitable for LLM consumption.

    Args:
        db: Catalog database instance.

    Returns:
        Structured text storyboard.
    """
    raise NotImplementedError


def propose_narratives(
    storyboard: str,
    db: CatalogDB,
    config: AutopilotConfig,
) -> list[Narrative]:
    """Propose video narratives using LLM analysis of the storyboard.

    Sends the master storyboard to Claude Opus for narrative planning,
    parses the response into Narrative objects, and stores them in the DB.

    Args:
        storyboard: Structured text storyboard from build_master_storyboard.
        db: Catalog database instance for storing proposals.
        config: Full autopilot config with creator profile and LLM settings.

    Returns:
        List of proposed Narrative objects.

    Raises:
        NarrativeError: If LLM call fails or response is malformed.
    """
    raise NotImplementedError


def format_for_review(narratives: list[Narrative]) -> str:
    """Format proposed narratives for human review.

    Produces a human-readable summary of each narrative with numbered
    entries showing title, duration, included clusters, arc, emotional
    journey, and reasoning.

    Args:
        narratives: List of Narrative objects to format.

    Returns:
        Formatted review text.
    """
    raise NotImplementedError
