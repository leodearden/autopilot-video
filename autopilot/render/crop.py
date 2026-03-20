"""Auto-crop viewport computation for intelligent reframing.

Computes per-frame crop coordinates to reframe source footage into
target aspect ratios (e.g. 16:9 or 9:16). Supports rule-of-thirds
framing, EMA smoothing, detection gap handling, and boundary clamping.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np  # type: ignore[reportMissingImports]

if TYPE_CHECKING:
    from autopilot.config import CameraConfig
    from autopilot.db import CatalogDB

__all__ = ["CropError", "compute_crop_path"]

logger = logging.getLogger(__name__)


class CropError(Exception):
    """Raised for all auto-crop computation failures."""


def _compute_crop_dimensions(
    source_w: int, source_h: int, target_aspect: str
) -> tuple[int, int]:
    """Compute crop window dimensions from source resolution and target aspect.

    Maximizes the crop window to use as much of the source as possible while
    matching the target aspect ratio exactly.

    Args:
        source_w: Source frame width in pixels.
        source_h: Source frame height in pixels.
        target_aspect: Aspect ratio string like '16:9' or '9:16'.

    Returns:
        (crop_w, crop_h) in pixels.

    Raises:
        CropError: If the aspect string cannot be parsed.
    """
    try:
        parts = target_aspect.split(":")
        if len(parts) != 2:
            raise ValueError("need exactly two parts")
        aspect_w = int(parts[0])
        aspect_h = int(parts[1])
        if aspect_w <= 0 or aspect_h <= 0:
            raise ValueError("aspect components must be positive")
    except (ValueError, TypeError) as e:
        raise CropError(
            f"Invalid target aspect ratio: {target_aspect!r}"
        ) from e

    # Try fitting width first (maximize width to source_w)
    crop_w = source_w
    crop_h = source_w * aspect_h // aspect_w

    if crop_h > source_h:
        # Width-first doesn't fit; fit by height instead
        crop_h = source_h
        crop_w = source_h * aspect_w // aspect_h

    return (crop_w, crop_h)


def compute_crop_path(
    media_id: str,
    target_aspect: str,
    db: CatalogDB,
    config: CameraConfig,
    edl_entry: dict,
) -> np.ndarray:
    """Compute per-frame crop coordinates for a media file.

    Args:
        media_id: ID of the source media file.
        target_aspect: Target aspect ratio string (e.g. '16:9', '9:16').
        db: Catalog database handle for reading detections and storing results.
        config: Camera configuration with source resolution and smoothing params.
        edl_entry: EDL entry dict with mode, subject_track_id, timecodes, etc.

    Returns:
        Array of shape (N, 2) with per-frame (crop_x, crop_y) top-left coordinates.

    Raises:
        CropError: For any crop computation failure.
    """
    raise NotImplementedError("compute_crop_path not yet implemented")
