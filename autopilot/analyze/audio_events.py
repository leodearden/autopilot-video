"""Audio event classification using PANNs CNN14.

Provides classify_audio_events() for per-second top-5 audio event
classification against AudioSet's 527 classes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np  # pyright: ignore[reportMissingImports]

if TYPE_CHECKING:
    from autopilot.analyze.gpu_scheduler import GPUScheduler
    from autopilot.db import CatalogDB

__all__ = ["AudioEventError", "classify_audio_events"]

logger = logging.getLogger(__name__)


class AudioEventError(Exception):
    """Raised for all audio event classification failures."""


def _window_audio(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float = 1.0,
) -> list[np.ndarray]:
    """Split a 1D waveform into non-overlapping windows.

    Args:
        audio: 1D float waveform array.
        sample_rate: Audio sample rate in Hz.
        window_seconds: Duration of each window in seconds.

    Returns:
        List of 1D arrays, each of length sample_rate * window_seconds.
        The final window is zero-padded if shorter than the window size.
    """
    if len(audio) == 0:
        return []
    window_size = int(sample_rate * window_seconds)
    num_windows = -(-len(audio) // window_size)  # ceil division
    # Pad to exact multiple of window_size
    pad_length = num_windows * window_size - len(audio)
    if pad_length > 0:
        audio = np.pad(audio, (0, pad_length))
    return [audio[i * window_size : (i + 1) * window_size] for i in range(num_windows)]


def _extract_top_k(
    probabilities: np.ndarray,
    labels: list[str],
    k: int = 5,
) -> list[dict[str, Any]]:
    """Extract top-k class predictions from a probability vector.

    Args:
        probabilities: 1D array of class probabilities.
        labels: List of class label strings (same length as probabilities).
        k: Number of top predictions to return.

    Returns:
        List of dicts with 'class' (str) and 'probability' (float),
        sorted by probability descending.
    """
    k = min(k, len(labels))
    top_indices = np.argsort(probabilities)[::-1][:k]
    return [{"class": str(labels[i]), "probability": float(probabilities[i])} for i in top_indices]


def classify_audio_events(
    media_id: str,
    audio_path: Path,
    db: CatalogDB,
    scheduler: GPUScheduler,
    *,
    top_k: int = 5,
) -> None:
    """Run PANNs CNN14 audio event classification.

    Args:
        media_id: Unique identifier for the media file.
        audio_path: Path to the audio file.
        db: Catalog database for storing audio events.
        scheduler: GPU scheduler for model loading.
        top_k: Number of top predictions per second window.
    """
    # Idempotency: skip if audio events already exist for this media
    existing = db.has_audio_events(media_id)
    if existing:
        logger.info("Audio events already exist for %s, skipping", media_id)
        return

    # Validate audio path before importing heavy ML libraries
    if not audio_path.exists():
        raise AudioEventError(f"Audio file not found: {audio_path}")

    # Update media status
    with db:
        db.update_media_status(media_id, "analyzing")

    try:
        import librosa  # type: ignore[import-untyped]

        audio, sr = librosa.load(str(audio_path), sr=32000, mono=True)
        windows = _window_audio(audio, 32000)

        if not windows:
            logger.info("No audio data for %s, skipping classification", media_id)
            return

        import panns_inference  # type: ignore[import-untyped]

        labels = panns_inference.config.labels

        logger.info("Starting audio event classification for %s", media_id)

        rows: list[tuple[str, float, str]] = []
        with scheduler.model("panns_cnn14") as model:
            for i, window in enumerate(windows):
                clipwise_output, _ = model.inference(window[None, :])
                events = _extract_top_k(clipwise_output[0], labels, k=top_k)
                rows.append((media_id, float(i), json.dumps(events)))

        with db:
            db.batch_insert_audio_events(rows)

        logger.info(
            "Completed audio event classification for %s: %d seconds",
            media_id,
            len(windows),
        )
    except AudioEventError:
        raise
    except Exception as exc:
        raise AudioEventError(f"Audio event classification failed for {media_id}: {exc}") from exc
