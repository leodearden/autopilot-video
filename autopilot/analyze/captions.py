"""Selective video captioning with Qwen2.5-VL-7B.

Provides caption_clip() for on-demand single-clip captioning and
batch_caption() for sampling/bulk captioning across media files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.config import ModelConfig
    from autopilot.db import CatalogDB

__all__ = ["CaptionError", "caption_clip", "batch_caption"]

logger = logging.getLogger(__name__)


class CaptionError(Exception):
    """Raised for all captioning-related failures."""


def _extract_clip_frames(
    video_path: Path,
    start_time: float,
    end_time: float,
    fps: float,
    num_frames: int = 8,
) -> list[Image.Image]:
    """Extract evenly-spaced frames from a video clip segment.

    Args:
        video_path: Path to the video file.
        start_time: Clip start time in seconds.
        end_time: Clip end time in seconds.
        fps: Video frame rate.
        num_frames: Number of frames to extract.

    Returns:
        List of PIL Image objects (RGB).
    """
    from PIL import Image as PILImage

    if end_time <= start_time:
        return []

    import cv2  # type: ignore[reportMissingImports]

    cap = cv2.VideoCapture(str(video_path))
    try:
        # Compute evenly-spaced timestamps in [start_time, end_time]
        if num_frames == 1:
            timestamps = [(start_time + end_time) / 2.0]
        else:
            step = (end_time - start_time) / (num_frames - 1)
            timestamps = [start_time + i * step for i in range(num_frames)]

        frames: list[Image.Image] = []
        for ts in timestamps:
            frame_idx = int(ts * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            # Convert BGR (cv2) to RGB (PIL)
            rgb_frame = frame[:, :, ::-1].copy()
            frames.append(PILImage.fromarray(rgb_frame))

        return frames
    finally:
        cap.release()


_CAPTION_PROMPT = (
    "Describe the scene, activities, and notable elements in this video clip."
)


def caption_clip(
    media_id: str,
    video_path: Path,
    start_time: float,
    end_time: float,
    db: CatalogDB,
    scheduler: GPUScheduler,
    config: ModelConfig,
) -> str:
    """Generate a caption for a single video clip segment.

    Args:
        media_id: Unique identifier for the media file.
        video_path: Path to the video file.
        start_time: Clip start time in seconds.
        end_time: Clip end time in seconds.
        db: Catalog database for storing/retrieving captions.
        scheduler: GPU scheduler for model loading.
        config: Model configuration.

    Returns:
        Generated caption string.
    """
    # Idempotency: return existing caption if available
    existing = db.get_caption(media_id, start_time, end_time)
    if existing is not None:
        logger.info("Caption already exists for %s [%.1f-%.1f], skipping", media_id, start_time, end_time)
        return str(existing["caption"])

    # Get media metadata for fps
    media = db.get_media(media_id)
    fps = float(media["fps"]) if media and media.get("fps") else 30.0

    # Extract frames from clip segment
    frames = _extract_clip_frames(video_path, start_time, end_time, fps)

    # Load model via scheduler and run inference
    with scheduler.model(config.caption_model) as model_bundle:
        model = model_bundle["model"]
        processor = model_bundle["processor"]

        # Build conversation for the VLM
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": _CAPTION_PROMPT})

        messages = [{"role": "user", "content": content}]

        # Process inputs and generate
        inputs = processor(messages, return_tensors="pt")
        output_ids = model.generate(**inputs, max_new_tokens=256)
        caption = processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]

    # Store in database
    db.upsert_caption(media_id, start_time, end_time, caption, config.caption_model)

    return caption


def batch_caption(
    media_ids: list[str],
    db: CatalogDB,
    scheduler: GPUScheduler,
    config: ModelConfig,
    sample_rate: float = 0.1,
) -> None:
    """Caption clips across multiple media files with sampling.

    Args:
        media_ids: List of media file identifiers.
        db: Catalog database.
        scheduler: GPU scheduler for model loading.
        config: Model configuration.
        sample_rate: Fraction of clips to caption (0.0 to 1.0).
    """
    raise NotImplementedError
