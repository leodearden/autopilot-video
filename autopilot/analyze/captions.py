"""Selective video captioning with Qwen2.5-VL-7B.

Provides caption_clip() for on-demand single-clip captioning and
batch_caption() for sampling/bulk captioning across media files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL import Image

    from autopilot.analyze.gpu_scheduler import GPUScheduler, ModelSpec
    from autopilot.config import ModelConfig
    from autopilot.db import CatalogDB

__all__ = ["CaptionError", "caption_clip", "batch_caption", "register_caption_model"]

logger = logging.getLogger(__name__)


class CaptionError(Exception):
    """Raised for all captioning-related failures."""


_CAPTION_VRAM_BYTES = 14 * 1024**3  # ~14 GB FP16 for 7B model


def _make_caption_model_spec(config: ModelConfig) -> ModelSpec:
    """Create a ModelSpec for the caption model.

    Tries vLLM first for batched inference; falls back to transformers.
    """
    from autopilot.analyze.gpu_scheduler import ModelSpec

    model_name = config.caption_model

    def _load_fn() -> dict[str, Any]:
        try:
            import vllm  # noqa: F401

            logger.info("Loading %s via vLLM", model_name)
            llm = vllm.LLM(model=model_name, dtype="float16")
            return {"backend": "vllm", "model": llm, "processor": None}
        except ImportError:
            logger.warning("vLLM unavailable, falling back to transformers for %s", model_name)
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto"
            )
            return {"backend": "transformers", "model": model, "processor": processor}

    def _unload_fn(model_bundle: Any) -> None:
        del model_bundle
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    return ModelSpec(
        load_fn=_load_fn,
        unload_fn=_unload_fn,
        vram_bytes=_CAPTION_VRAM_BYTES,
    )


def register_caption_model(scheduler: GPUScheduler, config: ModelConfig) -> None:
    """Register the caption model with the GPU scheduler."""
    spec = _make_caption_model_spec(config)
    scheduler.register(config.caption_model, spec)


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


def _run_transformers_inference(
    model_bundle: dict[str, Any],
    frames: list[Image.Image],
) -> str:
    """Run inference via HuggingFace transformers backend.

    Args:
        model_bundle: Dict with 'model' and 'processor' keys.
        frames: List of PIL Images from the clip.

    Returns:
        Generated caption string.
    """
    model = model_bundle["model"]
    processor = model_bundle["processor"]

    content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": _CAPTION_PROMPT})
    messages = [{"role": "user", "content": content}]

    inputs = processor(messages, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=256)
    # Slice off input prompt tokens before decoding (standard HF pattern)
    output_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    return str(processor.batch_decode(output_ids, skip_special_tokens=True)[0])


def _run_vllm_inference(
    model_bundle: dict[str, Any],
    frames: list[Image.Image],
) -> str:
    """Run inference via vLLM backend.

    Args:
        model_bundle: Dict with 'model' key (vLLM LLM instance).
        frames: List of PIL Images from the clip.

    Returns:
        Generated caption string.
    """
    from vllm import SamplingParams  # type: ignore[import-untyped]

    llm = model_bundle["model"]

    # Build prompt with image placeholders for vLLM
    image_tokens = "".join("<image>" for _ in frames)
    prompt = f"{image_tokens}\n{_CAPTION_PROMPT}"

    sampling_params = SamplingParams(max_tokens=256)
    outputs = llm.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": frames}}],
        sampling_params=sampling_params,
    )
    return str(outputs[0].outputs[0].text)


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
    # Validate time range inputs (before any DB/IO)
    if start_time < 0:
        raise CaptionError(f"Invalid start_time: {start_time} (must be >= 0)")
    if start_time > end_time:
        raise CaptionError(
            f"Invalid time range: start_time={start_time} > end_time={end_time}"
        )

    # Idempotency: return existing caption if available
    existing = db.get_caption(media_id, start_time, end_time)
    if existing is not None:
        logger.info(
            "Caption already exists for %s [%.1f-%.1f], skipping",
            media_id, start_time, end_time,
        )
        return str(existing["caption"])

    logger.info("Captioning %s [%.1f-%.1f]", media_id, start_time, end_time)

    # Validate video path (after idempotency check, before IO)
    if not video_path.exists():
        raise CaptionError(f"Video file not found: {video_path}")

    # Validate media exists before any GPU work
    media = db.get_media(media_id)
    if media is None:
        raise CaptionError(f"Media not found: {media_id}")
    fps = float(media.get("fps") or 30.0)  # type: ignore[arg-type]

    # Extract frames from clip segment
    frames = _extract_clip_frames(video_path, start_time, end_time, fps)
    if not frames:
        raise CaptionError(f"No frames extracted from {video_path} [{start_time}-{end_time}]")

    # Load model via scheduler and run inference
    with scheduler.model(config.caption_model) as model_bundle:
        backend = model_bundle.get("backend", "transformers")
        if backend == "vllm":
            caption = _run_vllm_inference(model_bundle, frames)
        else:
            caption = _run_transformers_inference(model_bundle, frames)

    # Store in database (with db: ensures commit on success / rollback on failure)
    with db:
        db.upsert_caption(media_id, start_time, end_time, caption, config.caption_model)

    logger.info(
        "Caption complete for %s [%.1f-%.1f]: %d chars",
        media_id, start_time, end_time, len(caption),
    )

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
    import random

    if not media_ids:
        return

    # Build clip segments for each media file
    clips: list[tuple[str, Path, float, float]] = []
    for mid in media_ids:
        media = db.get_media(mid)
        if media is None:
            logger.warning("Media %s not found in DB, skipping", mid)
            continue
        file_path = Path(str(media["file_path"]))
        dur_val = media.get("duration_seconds") or 0.0
        duration = float(dur_val)  # type: ignore[arg-type]
        if duration <= 0:
            logger.warning("Media %s has no duration, skipping", mid)
            continue
        # Use full video as single clip (shot boundaries can be added later)
        clips.append((mid, file_path, 0.0, duration))

    # Apply sampling
    if sample_rate < 1.0:
        k = max(1, int(len(clips) * sample_rate))
        clips = random.sample(clips, min(k, len(clips)))

    # Caption each selected clip
    completed = 0
    total = len(clips)
    for mid, file_path, start, end in clips:
        try:
            caption_clip(mid, file_path, start, end, db, scheduler, config)
            completed += 1
        except CaptionError as e:
            logger.error("Failed to caption %s [%.1f-%.1f]: %s", mid, start, end, e)

    logger.info("%d/%d clips captioned", completed, total)
