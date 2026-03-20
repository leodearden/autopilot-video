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


def _select_subject_track(
    all_detections: list[list[dict]], edl_entry: dict
) -> int:
    """Select the subject track to follow for auto-crop.

    If edl_entry specifies an integer subject_track_id, return it directly.
    Otherwise, auto-select the track with the highest cumulative bbox area
    across all frames.

    Args:
        all_detections: Per-frame list of detection dicts, each with
            'track_id', 'bbox_xywh' [cx, cy, w, h], etc.
        edl_entry: EDL entry dict, may contain 'subject_track_id'.

    Returns:
        Integer track ID.

    Raises:
        CropError: If no detections are available for auto-selection.
    """
    track_id = edl_entry.get("subject_track_id")
    if isinstance(track_id, int):
        return track_id

    # Auto-select: find track with largest cumulative bbox area
    track_areas: dict[int, float] = {}
    for frame_dets in all_detections:
        for det in frame_dets:
            tid = det.get("track_id")
            if tid is None:
                continue
            bbox = det.get("bbox_xywh", [0, 0, 0, 0])
            area = bbox[2] * bbox[3]  # width * height
            track_areas[tid] = track_areas.get(tid, 0.0) + area

    if not track_areas:
        raise CropError("No detections with track IDs available for subject selection")

    return max(track_areas, key=track_areas.get)  # type: ignore[arg-type]


def _compute_raw_center(
    subject_bbox: list[float],
    crop_w: int,
    crop_h: int,
    thirds_horizontal: str = "right",
) -> tuple[float, float]:
    """Compute raw crop center using rule-of-thirds framing for a single subject.

    Places the subject horizontally at the 1/3 or 2/3 position of the crop
    window, and vertically positions the eye line (top 1/3 of bbox) at the
    upper 1/3 of the crop.

    Args:
        subject_bbox: [cx, cy, w, h] in source pixel coordinates.
        crop_w: Crop window width.
        crop_h: Crop window height.
        thirds_horizontal: 'left' to place subject at 1/3, 'right' for 2/3.

    Returns:
        (crop_center_x, crop_center_y) in source pixel coordinates.
    """
    subj_cx, subj_cy, _, bbox_h = subject_bbox

    # Horizontal: place subject at target fraction of crop width
    if thirds_horizontal == "left":
        h_frac = 1.0 / 3.0
    else:
        h_frac = 2.0 / 3.0

    # crop_center_x such that subject is at h_frac from left of crop:
    # subject_cx = crop_center_x - crop_w/2 + h_frac * crop_w
    # => crop_center_x = subject_cx - (h_frac - 0.5) * crop_w
    crop_center_x = subj_cx - (h_frac - 0.5) * crop_w

    # Vertical: place eye line at 1/3 from top of crop
    # Eye line is at top 1/3 of bbox: eye_y = subj_cy - bbox_h/3
    eye_y = subj_cy - bbox_h / 3.0
    # eye_y = crop_center_y - crop_h/2 + crop_h/3
    # => crop_center_y = eye_y + crop_h/2 - crop_h/3 = eye_y + crop_h/6
    crop_center_y = eye_y + crop_h / 6.0

    return (crop_center_x, crop_center_y)


def _compute_multi_subject_center(
    subject_bboxes: list[list[float]],
    crop_w: int,
    crop_h: int,
) -> tuple[float, float]:
    """Compute crop center for multiple subjects by centering on their bounding box.

    Args:
        subject_bboxes: List of [cx, cy, w, h] bboxes in source pixel coordinates.
        crop_w: Crop window width (unused, reserved for future aspect-aware logic).
        crop_h: Crop window height (unused, reserved for future aspect-aware logic).

    Returns:
        (crop_center_x, crop_center_y) centered on the bounding box of all subjects.
    """
    if not subject_bboxes:
        return (crop_w / 2.0, crop_h / 2.0)

    # Compute the bounding box containing all subject centers
    centers_x = [bbox[0] for bbox in subject_bboxes]
    centers_y = [bbox[1] for bbox in subject_bboxes]

    group_cx = (min(centers_x) + max(centers_x)) / 2.0
    group_cy = (min(centers_y) + max(centers_y)) / 2.0

    return (group_cx, group_cy)


def _build_raw_path(
    all_detections: list[list[dict]],
    track_id: int,
    crop_w: int,
    crop_h: int,
    thirds_horizontal: str = "right",
) -> np.ndarray:
    """Build per-frame raw crop center path from detections.

    For each frame, finds the specified track and computes a rule-of-thirds
    crop center. Frames where the track is absent get NaN markers.

    Args:
        all_detections: Per-frame list of detection dicts.
        track_id: Track ID to follow.
        crop_w: Crop window width.
        crop_h: Crop window height.
        thirds_horizontal: 'left' or 'right' for rule-of-thirds side.

    Returns:
        Array of shape (N, 2) with (crop_center_x, crop_center_y) per frame.
        NaN for frames where the track is not detected.
    """
    n_frames = len(all_detections)
    path = np.full((n_frames, 2), np.nan, dtype=np.float64)

    for i, frame_dets in enumerate(all_detections):
        # Find the target track in this frame
        subject_bbox = None
        for det in frame_dets:
            if det.get("track_id") == track_id:
                subject_bbox = det["bbox_xywh"]
                break

        if subject_bbox is not None:
            cx, cy = _compute_raw_center(subject_bbox, crop_w, crop_h, thirds_horizontal)
            path[i, 0] = cx
            path[i, 1] = cy

    return path


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
