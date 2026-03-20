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

from autopilot.plan.validator import timecode_to_seconds as _timecode_to_seconds

try:
    from moviepy import VideoFileClip  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    VideoFileClip = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from autopilot.config import OutputConfig

__all__ = ["ComplexRenderError", "render_complex"]

logger = logging.getLogger(__name__)


class ComplexRenderError(Exception):
    """Raised for all MoviePy complex-path rendering failures."""


def _apply_dynamic_crop(
    clip: object,
    crop_path: np.ndarray,
    crop_w: int,
    crop_h: int,
    fps: float,
) -> object:
    """Apply per-frame crop to a clip using crop_path coordinates.

    Args:
        clip: Source MoviePy clip.
        crop_path: Array of shape (N, 2) with (crop_x, crop_y) per frame.
        crop_w: Crop window width.
        crop_h: Crop window height.
        fps: Frame rate for frame index calculation.

    Returns:
        Cropped clip.
    """
    n_frames = crop_path.shape[0]

    def make_frame(get_frame, t):  # type: ignore[no-untyped-def]
        """Generate a cropped frame at time t."""
        frame = get_frame(t)
        frame_idx = int(t * fps)
        # Clamp to crop_path length
        frame_idx = min(frame_idx, n_frames - 1)
        crop_x = int(crop_path[frame_idx, 0])
        crop_y = int(crop_path[frame_idx, 1])
        # Clamp to frame bounds
        src_h, src_w = frame.shape[:2]
        crop_x = min(crop_x, max(0, src_w - crop_w))
        crop_y = min(crop_y, max(0, src_h - crop_h))
        return frame[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

    return clip.transform(make_frame)


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
    source_path = edl_entry["source_path"]
    in_tc = edl_entry.get("in_timecode", "00:00:00.000")
    out_tc = edl_entry.get("out_timecode")

    try:
        # Open source video
        clip = VideoFileClip(source_path)

        # Subclip by timecodes
        in_sec = _timecode_to_seconds(in_tc)
        out_sec = _timecode_to_seconds(out_tc) if out_tc else clip.duration
        clip = clip.subclipped(in_sec, out_sec)

        # Apply dynamic per-frame crop if crop_path provided
        if crop_path is not None:
            if crop_path.shape[0] == 0:
                raise ComplexRenderError(
                    f"Empty crop_path for clip {edl_entry.get('clip_id', '?')}"
                )
            crop_w, crop_h = config.resolution
            fps = clip.fps or 30
            clip = _apply_dynamic_crop(clip, crop_path, crop_w, crop_h, fps)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write output
        clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            audio_bitrate=config.audio_bitrate,
        )

    except ComplexRenderError:
        raise
    except Exception as e:
        raise ComplexRenderError(
            f"MoviePy render failed for clip {edl_entry.get('clip_id', '?')}: {e}"
        ) from e

    return output_path
