"""Voiceover generation via TTS engines (Kokoro or ElevenLabs)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.config import ModelConfig

__all__ = [
    "VoiceoverError",
    "generate_voiceover",
]

logger = logging.getLogger(__name__)

# ElevenLabs defaults
_ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
_ELEVENLABS_DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"  # Rachel


class VoiceoverError(Exception):
    """Raised for any voiceover generation error."""


def generate_voiceover(text: str, output_path: Path, config: ModelConfig) -> Path:
    """Generate a voiceover audio file from text using the configured TTS engine.

    Args:
        text: The narration text to be spoken.
        output_path: Path where the generated .wav file will be written.
        config: ModelConfig with tts_engine selection (kokoro or elevenlabs).

    Returns:
        Path to the generated audio file.

    Raises:
        VoiceoverError: If generation fails or engine is unknown.
    """
    engine = config.tts_engine

    if engine == "kokoro":
        return _generate_kokoro(text, output_path)
    elif engine == "elevenlabs":
        return _generate_elevenlabs(text, output_path)
    else:
        raise VoiceoverError(
            f"Unknown TTS engine: {engine!r}. Supported: kokoro, elevenlabs"
        )


def _generate_kokoro(text: str, output_path: Path) -> Path:
    """Generate voiceover using Kokoro TTS.

    Uses the kokoro library to synthesize speech locally.

    Args:
        text: Narration text.
        output_path: Output .wav file path.

    Returns:
        Path to the generated file.

    Raises:
        VoiceoverError: If Kokoro generation fails.
    """
    try:
        import kokoro
        import soundfile as sf
    except ImportError as e:
        raise VoiceoverError(
            f"Kokoro TTS dependencies not installed: {e}"
        ) from e

    try:
        logger.info("Generating voiceover with Kokoro TTS (%d chars)", len(text))
        pipeline = kokoro.KPipeline(lang_code="a")

        # Collect audio chunks from the pipeline
        audio_chunks = []
        for _graphemes, _phonemes, audio_chunk in pipeline(text):
            if audio_chunk is not None:
                audio_chunks.append(audio_chunk)

        if not audio_chunks:
            raise VoiceoverError("Kokoro produced no audio output")

        # Concatenate all chunks
        import numpy as np
        combined = np.concatenate(audio_chunks)

        # Write to file at 24kHz sample rate (Kokoro default)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), combined, 24000)

        logger.info("Kokoro voiceover written to %s", output_path)
        return output_path

    except VoiceoverError:
        raise
    except Exception as e:
        raise VoiceoverError(f"Kokoro TTS generation failed: {e}") from e


def _generate_elevenlabs(text: str, output_path: Path) -> Path:
    """Generate voiceover using ElevenLabs API.

    Args:
        text: Narration text.
        output_path: Output audio file path.

    Returns:
        Path to the generated file.

    Raises:
        VoiceoverError: If API call fails or key is missing.
    """
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise VoiceoverError(
            "ELEVENLABS_API_KEY environment variable not set"
        )

    try:
        import requests as _requests
    except ImportError as e:
        raise VoiceoverError(
            f"requests library not installed: {e}"
        ) from e

    try:
        logger.info("Generating voiceover with ElevenLabs API (%d chars)", len(text))

        url = f"{_ELEVENLABS_API_URL}/{_ELEVENLABS_DEFAULT_VOICE}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
            },
        }

        response = _requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)

        logger.info("ElevenLabs voiceover written to %s", output_path)
        return output_path

    except VoiceoverError:
        raise
    except Exception as e:
        raise VoiceoverError(f"ElevenLabs TTS generation failed: {e}") from e
