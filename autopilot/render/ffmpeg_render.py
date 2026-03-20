"""FFmpeg fast-path rendering — GPU-accelerated simple clip rendering.

Builds FFmpeg command lines with filter_complex for NVENC encoding,
loudnorm audio normalization, and xfade transitions. Used for clips
with static crops that don't require per-frame processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from autopilot.config import OutputConfig

__all__ = ["RenderError", "render_simple"]

logger = logging.getLogger(__name__)


class RenderError(Exception):
    """Raised for all FFmpeg fast-path rendering failures."""


def render_simple(
    edl_entry: dict,
    crop_path: np.ndarray | None,
    output_path: Path,
    config: OutputConfig,
) -> Path:
    """Render a clip via FFmpeg fast path (GPU-accelerated).

    Args:
        edl_entry: EDL clip entry with source_path, in_timecode, out_timecode,
            transition info, and audio settings.
        crop_path: Optional ndarray of shape (N, 2) with per-frame crop
            coordinates. Must be static (all rows identical) for fast path.
        output_path: Path for the rendered output file.
        config: Output encoding configuration.

    Returns:
        The output_path on success.

    Raises:
        RenderError: If rendering fails.
    """
    raise NotImplementedError("render_simple not yet implemented")
