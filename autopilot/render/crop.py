"""Auto-crop viewport computation for intelligent reframing.

Computes per-frame crop coordinates to reframe source footage into
target aspect ratios (e.g. 16:9 or 9:16). Supports rule-of-thirds
framing, EMA smoothing, detection gap handling, and boundary clamping.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, cast

import numpy as np  # type: ignore[reportMissingImports]

from autopilot.plan.validator import timecode_to_seconds

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
                subject_bbox = det.get("bbox_xywh")
                break

        if subject_bbox is not None:
            cx, cy = _compute_raw_center(subject_bbox, crop_w, crop_h, thirds_horizontal)
            path[i, 0] = cx
            path[i, 1] = cy

    return path


def _smooth_path(raw_path: np.ndarray, fps: float, tau: float) -> np.ndarray:
    """Apply EMA smoothing to a crop center path.

    Uses exponential moving average with alpha = 1 - exp(-1/(tau*fps)).
    NaN entries are skipped (the smoothed value holds from the last valid frame).

    Args:
        raw_path: Array of shape (N, 2) with per-frame crop centers.
        fps: Video frame rate in frames per second.
        tau: Time constant in seconds (higher = more smoothing).

    Returns:
        Smoothed array of shape (N, 2).
    """
    alpha = 1.0 - np.exp(-1.0 / (tau * fps))
    n = raw_path.shape[0]
    smoothed = np.empty_like(raw_path)

    # Initialize with first valid frame
    initialized = False
    prev = np.zeros(2)
    for i in range(n):
        if np.any(np.isnan(raw_path[i])):
            # Hold previous value
            smoothed[i] = prev
        elif not initialized:
            prev = raw_path[i].copy()
            smoothed[i] = prev
            initialized = True
        else:
            prev = prev + alpha * (raw_path[i] - prev)
            smoothed[i] = prev

    return smoothed


def _handle_detection_gaps(
    raw_path: np.ndarray,
    fps: float,
    source_w: int,
    source_h: int,
    hold_seconds: float = 2.0,
    drift_seconds: float = 1.0,
) -> np.ndarray:
    """Replace NaN gaps in the raw path with hold/drift/center behavior.

    Strategy for each NaN frame within a gap:
    - First hold_seconds of gap: hold last valid position.
    - Next drift_seconds: linear interpolation from held position to frame center.
    - Beyond that: stay at frame center.

    Args:
        raw_path: Array of shape (N, 2) with NaN for missing frames.
        fps: Video frame rate.
        source_w: Source frame width.
        source_h: Source frame height.
        hold_seconds: How long to hold last position before drifting.
        drift_seconds: How long the drift phase lasts.

    Returns:
        Array of shape (N, 2) with all NaN values replaced.
    """
    result = raw_path.copy()
    n = result.shape[0]
    center = np.array([source_w / 2.0, source_h / 2.0])
    hold_frames = int(hold_seconds * fps)
    drift_frames = int(drift_seconds * fps)

    last_valid = center.copy()
    gap_start = -1

    for i in range(n):
        if np.any(np.isnan(result[i])):
            if gap_start < 0:
                gap_start = i
            frames_into_gap = i - gap_start

            if frames_into_gap < hold_frames:
                # Hold phase
                result[i] = last_valid
            elif frames_into_gap < hold_frames + drift_frames:
                # Drift phase: linear interpolation toward center
                t = (frames_into_gap - hold_frames) / max(drift_frames, 1)
                result[i] = last_valid + t * (center - last_valid)
            else:
                # Past drift: at center
                result[i] = center
        else:
            last_valid = result[i].copy()
            gap_start = -1

    return result


def _clamp_to_bounds(
    path: np.ndarray,
    source_w: int,
    source_h: int,
    crop_w: int,
    crop_h: int,
) -> np.ndarray:
    """Convert crop centers to clamped top-left (crop_x, crop_y) coordinates.

    Converts center coordinates to top-left corner, then clamps so the entire
    crop window stays within [0, source_w] x [0, source_h].

    Args:
        path: Array of shape (N, 2) with (center_x, center_y) per frame.
        source_w: Source frame width.
        source_h: Source frame height.
        crop_w: Crop window width.
        crop_h: Crop window height.

    Returns:
        Array of shape (N, 2) with (crop_x, crop_y) top-left coordinates.
    """
    result = np.empty_like(path)

    # Convert center to top-left
    result[:, 0] = path[:, 0] - crop_w / 2.0
    result[:, 1] = path[:, 1] - crop_h / 2.0

    # Clamp to bounds
    max_x = max(0.0, float(source_w - crop_w))
    max_y = max(0.0, float(source_h - crop_h))

    result[:, 0] = np.clip(result[:, 0], 0.0, max_x)
    result[:, 1] = np.clip(result[:, 1], 0.0, max_y)

    return result


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
    # -- Common setup ----------------------------------------------------------
    source_w, source_h = config.source_resolution
    crop_w, crop_h = _compute_crop_dimensions(source_w, source_h, target_aspect)

    # Determine frame range from timecodes
    media = db.get_media(media_id)
    if media is None:
        raise CropError(f"Media not found: {media_id!r}")

    fps = float(cast(float, media["fps"])) if media.get("fps") else 30.0
    in_tc = edl_entry.get("in_timecode", "00:00:00.000")
    out_tc = edl_entry.get("out_timecode")
    start_sec = timecode_to_seconds(in_tc)
    if out_tc:
        end_sec = timecode_to_seconds(out_tc)
    else:
        end_sec = float(cast(float, media.get("duration_seconds") or 0.0))

    num_frames = max(1, int(math.ceil((end_sec - start_sec) * fps)))

    mode = edl_entry.get("mode", "center")

    # -- Mode dispatch --------------------------------------------------------
    if mode == "center":
        center_x = (source_w - crop_w) / 2.0
        center_y = (source_h - crop_h) / 2.0
        path = np.full((num_frames, 2), [center_x, center_y])
        return path

    if mode == "manual_offset":
        offset_x = float(edl_entry.get("offset_x", 0))
        offset_y = float(edl_entry.get("offset_y", 0))
        raw_x = (source_w - crop_w) / 2.0 + offset_x
        raw_y = (source_h - crop_h) / 2.0 + offset_y
        # Clamp to bounds
        max_x = max(0.0, float(source_w - crop_w))
        max_y = max(0.0, float(source_h - crop_h))
        clamped_x = float(np.clip(raw_x, 0.0, max_x))
        clamped_y = float(np.clip(raw_y, 0.0, max_y))
        path = np.full((num_frames, 2), [clamped_x, clamped_y])
        return path

    if mode == "stabilize_only":
        logger.warning(
            "stabilize_only mode is a placeholder; falling back to center crop "
            "(gyro-based stabilization not yet implemented)"
        )
        center_x = (source_w - crop_w) / 2.0
        center_y = (source_h - crop_h) / 2.0
        path = np.full((num_frames, 2), [center_x, center_y])
        return path

    if mode == "auto_subject":
        # Load detections for frame range
        frame_start = int(start_sec * fps)
        frame_end = frame_start + num_frames - 1
        det_rows = db.get_detections_for_range(media_id, frame_start, frame_end)

        # Build per-frame detection lists
        det_by_frame: dict[int, list[dict]] = {}
        for row in det_rows:
            fn = int(cast(int, row["frame_number"]))
            try:
                det_by_frame[fn] = json.loads(cast(str, row["detections_json"]))
            except (json.JSONDecodeError, TypeError) as e:
                raise CropError(
                    f"Malformed detections_json for media={media_id!r} "
                    f"frame={fn}: {e}"
                ) from e

        all_detections = [
            det_by_frame.get(frame_start + i, []) for i in range(num_frames)
        ]

        # Select subject track
        track_id = _select_subject_track(all_detections, edl_entry)

        # Build raw path
        raw_path = _build_raw_path(all_detections, track_id, crop_w, crop_h)

        # Handle detection gaps
        filled_path = _handle_detection_gaps(
            raw_path, fps, source_w, source_h,
            hold_seconds=2.0, drift_seconds=1.0,
        )

        # Smooth
        tau = edl_entry.get("smoothing_tau", config.crop_smoothing_tau)
        if not tau or tau <= 0:
            raise CropError(f"smoothing_tau must be positive, got {tau!r}")
        smoothed_path = _smooth_path(filled_path, fps, tau)

        # Clamp to bounds (converts centers to top-left)
        result = _clamp_to_bounds(smoothed_path, source_w, source_h, crop_w, crop_h)

        # Store in DB
        path_data = result.astype(np.float64).tobytes()
        db.upsert_crop_path(
            media_id, target_aspect, track_id,
            smoothing_tau=tau, path_data=path_data,
        )

        return result

    raise CropError(f"Unknown crop mode: {mode!r}")
