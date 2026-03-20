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


# Crop modes that use static (constant) crop coordinates — FFmpeg fast path
_FAST_CROP_MODES = {"center", "manual_offset", "stabilize_only"}

# Crop modes that produce dynamic per-frame crops — MoviePy slow path
_SLOW_CROP_MODES = {"auto_subject"}


class RoutingError(Exception):
    """Raised for all render routing and assembly failures."""


def _classify_clip(clip: dict, crop_modes: dict) -> str:
    """Classify a clip as 'fast' (FFmpeg) or 'slow' (MoviePy).

    Args:
        clip: EDL clip entry dict with clip_id and optional overlay field.
        crop_modes: Mapping of clip_id -> crop mode string from EDL.

    Returns:
        'fast' or 'slow'.
    """
    # Complex overlays force slow path
    overlay = clip.get("overlay")
    if overlay in ("pip", "split_screen"):
        return "slow"

    # Check crop mode
    clip_id = clip.get("clip_id", "")
    mode = crop_modes.get(clip_id, "center")  # default to center (fast)

    if mode in _SLOW_CROP_MODES:
        return "slow"

    return "fast"


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
