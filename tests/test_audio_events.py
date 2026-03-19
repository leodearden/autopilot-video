"""Tests for audio event classification (autopilot.analyze.audio_events)."""

from __future__ import annotations

import inspect
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# -- Mock helpers --------------------------------------------------------------


def _make_mock_panns(
    labels: list[str],
) -> tuple[MagicMock, MagicMock]:
    """Create mock panns_inference module and config sub-module.

    Args:
        labels: List of AudioSet class labels.

    Returns:
        Tuple of (mock_panns_module, mock_panns_config_module).
        mock_panns_module has AudioTagging whose instances have .inference
        returning (np.random probabilities shape (1, len(labels)), mock_embedding).
        mock_panns_config_module has .labels attribute.
    """
    mock_panns = MagicMock()
    mock_config = MagicMock()
    mock_config.labels = labels

    # AudioTagging() returns a model object with .inference method
    probs = np.random.default_rng(42).random((1, len(labels)), dtype=np.float32)
    mock_embedding = np.zeros((1, 2048), dtype=np.float32)
    mock_model = MagicMock()
    mock_model.inference.return_value = (probs, mock_embedding)
    mock_panns.AudioTagging.return_value = mock_model

    # Wire config as sub-attribute
    mock_panns.config = mock_config

    return mock_panns, mock_config


def _make_mock_librosa(
    duration_seconds: float,
    sr: int = 32000,
) -> MagicMock:
    """Create mock librosa module returning audio of given duration.

    Args:
        duration_seconds: Length of the returned waveform in seconds.
        sr: Sample rate.

    Returns:
        MagicMock with .load returning (np.zeros waveform, sr).
    """
    mock_librosa = MagicMock()
    num_samples = int(duration_seconds * sr)
    waveform = np.zeros(num_samples, dtype=np.float32)
    mock_librosa.load.return_value = (waveform, sr)
    return mock_librosa


def _make_full_pipeline_mocks(
    catalog_db,
    media_id: str,
    duration_seconds: float,
    *,
    labels: list[str] | None = None,
    sr: int = 32000,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Set up all mocks for a complete audio event classification run.

    Inserts a media file into catalog_db, configures mock panns_inference,
    mock librosa, and mock scheduler.

    Args:
        catalog_db: CatalogDB fixture instance.
        media_id: Media ID to use.
        duration_seconds: Audio duration in seconds.
        labels: AudioSet labels (defaults to 527 synthetic labels).
        sr: Sample rate.

    Returns:
        Tuple of (mock_panns, mock_panns_config, mock_librosa, scheduler).
    """
    if labels is None:
        labels = [f"class_{i}" for i in range(527)]

    catalog_db.insert_media(media_id, f"/tmp/{media_id}.wav")

    mock_panns, mock_panns_config = _make_mock_panns(labels)
    mock_librosa = _make_mock_librosa(duration_seconds, sr)

    # Configure scheduler with context manager protocol
    mock_model = mock_panns.AudioTagging.return_value
    scheduler = MagicMock()
    scheduler.device = 0
    scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
    scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

    return mock_panns, mock_panns_config, mock_librosa, scheduler


# -- Test classes --------------------------------------------------------------


class TestPublicAPI:
    """Verify public API surface and type signatures."""

    def test_exports_importable(self):
        """AudioEventError and classify_audio_events are importable."""
        from autopilot.analyze.audio_events import (
            AudioEventError,
            classify_audio_events,
        )

        assert AudioEventError is not None
        assert classify_audio_events is not None

    def test_audio_event_error_is_exception(self):
        """AudioEventError is a subclass of Exception with message."""
        from autopilot.analyze.audio_events import AudioEventError

        assert issubclass(AudioEventError, Exception)
        err = AudioEventError("test message")
        assert str(err) == "test message"

    def test_classify_audio_events_signature(self):
        """classify_audio_events has correct parameter signature."""
        from autopilot.analyze.audio_events import classify_audio_events

        sig = inspect.signature(classify_audio_events)
        params = list(sig.parameters.keys())

        # Positional parameters
        assert "media_id" in params
        assert "audio_path" in params
        assert "db" in params
        assert "scheduler" in params

        # Keyword-only parameter with default
        top_k_param = sig.parameters["top_k"]
        assert top_k_param.kind == inspect.Parameter.KEYWORD_ONLY
        assert top_k_param.default == 5

        # Return annotation (string 'None' due to __future__ annotations)
        assert sig.return_annotation in (None, "None")
