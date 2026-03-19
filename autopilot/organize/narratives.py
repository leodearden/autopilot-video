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
