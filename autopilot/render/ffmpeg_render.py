"""FFmpeg fast-path rendering — GPU-accelerated simple clip rendering.

Builds FFmpeg command lines with filter_complex for NVENC encoding,
loudnorm audio normalization, and xfade transitions. Used for clips
with static crops that don't require per-frame processing.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from autopilot.plan.validator import timecode_to_seconds as _timecode_to_seconds

if TYPE_CHECKING:
    from autopilot.config import OutputConfig

__all__ = ["RenderError", "render_simple"]

logger = logging.getLogger(__name__)


class RenderError(Exception):
    """Raised for all FFmpeg fast-path rendering failures."""


def _is_static_crop(crop_path: np.ndarray) -> bool:
    """Return True if all rows in crop_path are identical (static crop)."""
    if crop_path.shape[0] <= 1:
        return True
    return bool(np.all(crop_path == crop_path[0]))


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
    source_path = edl_entry["source_path"]
    in_tc = edl_entry.get("in_timecode", "00:00:00.000")
    out_tc = edl_entry.get("out_timecode")

    # Build command
    cmd: list[str] = ["ffmpeg", "-y"]

    # Seek to start
    cmd.extend(["-ss", in_tc])

    # Input file
    cmd.extend(["-i", source_path])

    # End time
    if out_tc:
        cmd.extend(["-to", out_tc])

    # Build video filter chain
    vf_parts: list[str] = []

    # Static crop if crop_path provided and static
    if crop_path is not None and _is_static_crop(crop_path):
        crop_x = int(crop_path[0, 0])
        crop_y = int(crop_path[0, 1])
        crop_w, crop_h = config.resolution
        vf_parts.append(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")

    # Fade in/out for transitions
    transition = edl_entry.get("transition")
    if transition and isinstance(transition, dict):
        duration = transition.get("duration", 0.5)
        vf_parts.append(f"fade=t=in:st=0:duration={duration}")
        # Compute clip duration for fade-out placement
        in_sec = _timecode_to_seconds(in_tc)
        if out_tc:
            out_sec = _timecode_to_seconds(out_tc)
            clip_dur = out_sec - in_sec
            fade_out_start = max(0.0, clip_dur - duration)
            vf_parts.append(
                f"fade=t=out:st={fade_out_start}:duration={duration}"
            )

    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])

    # Video codec — NVENC
    cmd.extend(["-c:v", "h264_nvenc"])

    # Quality
    cmd.extend(["-crf", str(config.quality_crf)])

    # Audio: loudnorm + AAC
    cmd.extend([
        "-af", f"loudnorm=I={config.target_loudness_lufs}:TP=-1.5:LRA=11",
        "-c:a", "aac",
        "-b:a", config.audio_bitrate,
    ])

    # Output
    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RenderError(
            f"FFmpeg render failed for clip {edl_entry.get('clip_id', '?')}: {e}"
        ) from e

    return output_path
