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
    pass
