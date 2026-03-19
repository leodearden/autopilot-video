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
