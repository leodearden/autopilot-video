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

    # Validate audio path before importing whisperx
    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    # Update media status
    with db:
        db.update_media_status(media_id, "analyzing")

    import whisperx  # type: ignore[import-untyped]

    device_str = f"cuda:{scheduler.device}"
    logger.info(
        "Starting transcription for %s with model %s",
        media_id,
        config.whisper_size,
    )

    try:
        # Stage 1: Load audio and transcribe
        audio = whisperx.load_audio(str(audio_path))
        with scheduler.model(config.whisper_size) as model:
            result = model.transcribe(audio, batch_size=batch_size)
        language = result.get("language", "en")

        # Stage 2: Forced alignment for word-level timestamps (non-fatal)
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=language, device=device_str
            )
            result = whisperx.align(
                result["segments"],
                align_model,
                metadata,
                audio,
                device_str,
                return_char_alignments=False,
            )
            del align_model, metadata
        except Exception:
            logger.warning(
                "Forced alignment failed for %s, continuing without"
                " word timestamps",
                media_id,
                exc_info=True,
            )

        # Stage 3: Speaker diarization (non-fatal, requires HF token)
        token = hf_token or os.environ.get("HF_TOKEN")
        if token:
            try:
                diarize_pipeline = whisperx.DiarizationPipeline(
                    use_auth_token=token, device=device_str
                )
                diarize_segments = diarize_pipeline(audio)
                result = whisperx.assign_word_speakers(
                    diarize_segments, result
                )
            except Exception:
                logger.warning(
                    "Diarization failed for %s, continuing without"
                    " speaker labels",
                    media_id,
                    exc_info=True,
                )
        else:
            logger.info(
                "No HuggingFace token provided, skipping diarization"
                " for %s",
                media_id,
            )

        # Stage 4: Normalize and store transcript
        segments = _normalize_segments(result.get("segments", []))
        segments_json = json.dumps(segments)
        with db:
            db.upsert_transcript(media_id, segments_json, language)

        logger.info(
            "Completed transcription for %s: %d segments, language=%s",
            media_id,
            len(segments),
            language,
        )
    except TranscriptionError:
        raise
    except Exception as exc:
        raise TranscriptionError(
            f"Transcription failed for {media_id}: {exc}"
        ) from exc
