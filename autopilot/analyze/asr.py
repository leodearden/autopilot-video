"""ASR transcription using WhisperX with forced alignment and diarization.

Provides transcribe_media() for running WhisperX-based speech-to-text
on audio files with optional wav2vec2 forced alignment and pyannote
speaker diarization.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.config import ModelConfig
    from autopilot.db import CatalogDB

__all__ = ["TranscriptionError", "transcribe_media"]

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised for all transcription pipeline failures."""


def _normalize_segments(segments: list[dict]) -> list[dict]:
    """Normalize WhisperX segments to PRD schema.

    Strips internal WhisperX fields and ensures each segment has only:
    start, end, text, speaker, words (with word, start, end, score).
    Coerces numpy scalars to plain Python floats for JSON serialization.

    Args:
        segments: Raw WhisperX segment dicts.

    Returns:
        List of normalized segment dicts matching PRD schema.
    """
    result = []
    for seg in segments:
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": str(w.get("word", "")),
                "start": float(w["start"]) if w.get("start") is not None else None,
                "end": float(w["end"]) if w.get("end") is not None else None,
                "score": float(w.get("score", 0.0)),
            })
        result.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": str(seg.get("text", "")),
            "speaker": seg.get("speaker"),
            "words": words,
        })
    return result


def transcribe_media(
    media_id: str,
    audio_path: Path,
    db: CatalogDB,
    scheduler: GPUScheduler,
    config: ModelConfig,
    *,
    batch_size: int = 24,
    hf_token: str | None = None,
) -> None:
    """Run WhisperX transcription with optional alignment and diarization.

    Args:
        media_id: Unique identifier for the media file.
        audio_path: Path to the audio file.
        db: Catalog database for storing transcripts.
        scheduler: GPU scheduler for model loading.
        config: Model configuration (whisper_size).
        batch_size: Number of audio segments per inference batch.
        hf_token: HuggingFace token for pyannote diarization.
    """
    # Idempotency: skip if transcript already exists for this media
    existing = db.get_transcript(media_id)
    if existing is not None:
        logger.info("Transcript already exists for %s, skipping", media_id)
        return
