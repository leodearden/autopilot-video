"""Render routing — selects fast vs slow path per clip and assembles output.

Routes each EDL clip to either the FFmpeg fast path (static crops) or
the MoviePy slow path (dynamic per-frame crops, PiP overlays), then
concatenates all rendered segments into the final output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.config import OutputConfig
    from autopilot.db import CatalogDB

__all__ = ["RoutingError", "route_and_render"]

logger = logging.getLogger(__name__)


class RoutingError(Exception):
    """Raised for all render routing and assembly failures."""


def route_and_render(
    narrative_id: str,
    db: CatalogDB,
    config: OutputConfig,
) -> Path:
    """Route EDL clips to appropriate renderers and assemble final output.

    Args:
        narrative_id: ID of the narrative to render.
        db: Catalog database handle for loading EDL, media, and crop data.
        config: Output encoding configuration.

    Returns:
        Path to the final rendered output file.

    Raises:
        RoutingError: If routing or rendering fails.
    """
    raise NotImplementedError("route_and_render not yet implemented")
