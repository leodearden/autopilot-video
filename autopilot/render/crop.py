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
