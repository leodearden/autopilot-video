"""MoviePy complex-path rendering — frame-by-frame compositions.

Uses MoviePy 2.x APIs for per-frame crop application and complex
compositions like picture-in-picture overlays. Used for clips that
require dynamic per-frame processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from autopilot.config import OutputConfig

__all__ = ["ComplexRenderError", "render_complex"]

logger = logging.getLogger(__name__)


class ComplexRenderError(Exception):
    """Raised for all MoviePy complex-path rendering failures."""


def render_complex(
    edl_entry: dict,
    crop_path: np.ndarray | None,
    output_path: Path,
    config: OutputConfig,
) -> Path:
    """Render a clip via MoviePy complex path (frame-by-frame).

    Args:
        edl_entry: EDL clip entry with source_path, in_timecode, out_timecode,
            and overlay/composition settings.
        crop_path: ndarray of shape (N, 2) with per-frame crop (x, y)
            top-left coordinates for dynamic cropping.
        output_path: Path for the rendered output file.
        config: Output encoding configuration.

    Returns:
        The output_path on success.

    Raises:
        ComplexRenderError: If rendering fails.
    """
    raise NotImplementedError("render_complex not yet implemented")
