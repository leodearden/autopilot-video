"""Audio event classification using PANNs CNN14.

Provides classify_audio_events() for per-second top-5 audio event
classification against AudioSet's 527 classes.
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
) -> list[dict[str, object]]:
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
    return [
        {"class": str(labels[i]), "probability": float(probabilities[i])}
        for i in top_indices
    ]


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
    pass
