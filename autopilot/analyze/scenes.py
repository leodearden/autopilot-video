"""Shot boundary detection using TransNetV2 (GPU) and PySceneDetect (CPU fallback).

Provides detect_shots() for detecting shot/scene boundaries in video files.
TransNetV2 is tried first as the primary GPU-based method, with PySceneDetect
as a CPU-based fallback if TransNetV2 fails.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.db import CatalogDB

__all__ = ["ShotDetectionError", "detect_shots"]

logger = logging.getLogger(__name__)

# TransNetV2 expects frames downsampled to this resolution
TRANSNETV2_INPUT_WIDTH = 48
TRANSNETV2_INPUT_HEIGHT = 27

# PySceneDetect AdaptiveDetector threshold
PYSCENEDETECT_THRESHOLD = 27.0


class ShotDetectionError(Exception):
    """Raised for all shot boundary detection failures."""


def _pyscenedetect_to_boundaries(scenes: list) -> list[dict]:
    """Convert PySceneDetect scene list to boundary dicts.

    Args:
        scenes: List of (start_timecode, end_timecode) tuples from
            scenedetect.detect().

    Returns:
        List of dicts with start_frame, end_frame, transition_type keys.
    """
    result = []
    for start_tc, end_tc in scenes:
        result.append(
            {
                "start_frame": start_tc.get_frames(),
                "end_frame": end_tc.get_frames() - 1,
                "transition_type": "cut",
            }
        )
    return result


def _transnetv2_to_boundaries(scenes: np.ndarray) -> list[dict]:
    """Convert TransNetV2 scene pairs to boundary dicts.

    Args:
        scenes: (N, 2) array of [start_frame, end_frame] pairs from
            TransNetV2 predictions_to_scenes.

    Returns:
        List of dicts with start_frame, end_frame, transition_type keys.
    """
    if len(scenes) == 0:
        return []
    result = []
    for row in scenes:
        result.append(
            {
                "start_frame": int(row[0]),
                "end_frame": int(row[1]),
                "transition_type": "cut",
            }
        )
    return result


def _read_and_downsample_frames(
    video_path: Path, cv2_module: object
) -> tuple[np.ndarray, float, int]:
    """Read all frames from a video and downsample to TransNetV2 input size.

    Args:
        video_path: Path to the video file.
        cv2_module: The cv2 module (passed to enable deferred import testing).

    Returns:
        Tuple of (frames array shape (N, 27, 48, 3), fps, total_frames).

    Raises:
        ShotDetectionError: If the video cannot be opened.
    """
    cv2 = cv2_module
    cap = cv2.VideoCapture(str(video_path))  # type: ignore[union-attr]
    if not cap.isOpened():
        cap.release()
        raise ShotDetectionError(f"Failed to open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))  # type: ignore[union-attr]
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # type: ignore[union-attr]

        frames = []
        for _ in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            resized = cv2.resize(  # type: ignore[union-attr]
                frame,
                (TRANSNETV2_INPUT_WIDTH, TRANSNETV2_INPUT_HEIGHT),
                interpolation=cv2.INTER_AREA,  # type: ignore[union-attr]
            )
            frames.append(resized)

        if frames:
            arr = np.stack(frames, axis=0)
        else:
            arr = np.empty(
                (0, TRANSNETV2_INPUT_HEIGHT, TRANSNETV2_INPUT_WIDTH, 3),
                dtype=np.uint8,
            )
        return arr, fps, total_frames
    finally:
        cap.release()


def _run_pyscenedetect(media_id: str, video_path: Path, db: CatalogDB) -> None:
    """Run PySceneDetect adaptive detection and store boundaries.

    Args:
        media_id: Unique identifier for the media file.
        video_path: Path to the video file.
        db: Catalog database for storing boundaries.
    """
    import scenedetect  # type: ignore[import-untyped]
    from scenedetect.detectors import AdaptiveDetector  # type: ignore[import-untyped]

    scenes = scenedetect.detect(
        str(video_path),
        AdaptiveDetector(adaptive_threshold=PYSCENEDETECT_THRESHOLD),
    )
    boundaries = _pyscenedetect_to_boundaries(scenes)

    with db:
        db.upsert_boundaries(media_id, json.dumps(boundaries), "pyscenedetect")


def detect_shots(
    media_id: str,
    video_path: Path,
    db: CatalogDB,
    scheduler: GPUScheduler,
) -> None:
    """Detect shot boundaries in a video file.

    Tries TransNetV2 (GPU) first, falls back to PySceneDetect (CPU)
    on failure. Results are stored in the catalog database.

    Args:
        media_id: Unique identifier for the media file.
        video_path: Path to the video file.
        db: Catalog database for storing boundaries.
        scheduler: GPU scheduler for model loading.

    Raises:
        ShotDetectionError: If both detection methods fail.
    """
    # Idempotency: skip if boundaries already exist for this media
    existing = db.get_boundaries(media_id)
    if existing:
        logger.info("Boundaries already exist for %s, skipping", media_id)
        return

    # Validate video path before heavy imports
    if not video_path.exists():
        raise ShotDetectionError(f"Video file not found: {video_path}")

    logger.info("Starting shot detection for %s", media_id)

    # Try TransNetV2 (GPU primary)
    try:
        import cv2  # type: ignore[import-untyped]

        frames, fps, total_frames = _read_and_downsample_frames(video_path, cv2)

        with scheduler.model("transnetv2") as model:
            predictions, _ = model.predict_frames(frames)
            scenes = model.predictions_to_scenes(predictions)

        boundaries = _transnetv2_to_boundaries(scenes)

        with db:
            db.upsert_boundaries(media_id, json.dumps(boundaries), "transnetv2")

        logger.info(
            "Completed shot detection for %s: %d boundaries via %s",
            media_id,
            len(boundaries),
            "transnetv2",
        )
        return
    except Exception:
        logger.warning(
            "TransNetV2 failed for %s, falling back to PySceneDetect",
            media_id,
            exc_info=True,
        )

    # Fallback to PySceneDetect (CPU)
    try:
        _run_pyscenedetect(media_id, video_path, db)
        # Read back to log count
        row = db.get_boundaries(media_id, method="pyscenedetect")
        count = len(json.loads(row["boundaries_json"])) if row else 0
        logger.info(
            "Completed shot detection for %s: %d boundaries via %s",
            media_id,
            count,
            "pyscenedetect",
        )
    except Exception as exc:
        raise ShotDetectionError(f"Shot detection failed for {media_id}: {exc}") from exc
