"""Tests for ASR transcription (autopilot.analyze.asr)."""

from __future__ import annotations

import inspect
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# -- Mock helpers --------------------------------------------------------------


def _make_mock_whisperx() -> MagicMock:
    """Create a MagicMock module mimicking whisperx.

    Returns a mock with attributes: load_model, load_audio,
    load_align_model, align, DiarizationPipeline, assign_word_speakers.
    """
    mock_wx = MagicMock()
    mock_wx.load_model = MagicMock()
    mock_wx.load_audio = MagicMock(return_value="mock_audio_array")
    mock_wx.load_align_model = MagicMock(
        return_value=(MagicMock(name="align_model"), {"metadata": True})
    )
    mock_wx.align = MagicMock()
    mock_wx.DiarizationPipeline = MagicMock()
    mock_wx.assign_word_speakers = MagicMock()
    return mock_wx


def _make_whisperx_transcribe_result(
    segments: list[dict],
    language: str = "en",
) -> dict:
    """Build a dict mimicking whisperx model.transcribe() output.

    Args:
        segments: List of segment dicts with at least 'start', 'end', 'text'.
        language: Detected language code.

    Returns:
        Dict with 'segments' and 'language' keys.
    """
    return {"segments": segments, "language": language}


def _make_aligned_result(segments: list[dict]) -> dict:
    """Build a post-alignment result with word-level timestamps added.

    Args:
        segments: Aligned segment dicts with 'words' lists.

    Returns:
        Dict with 'segments' and 'word_segments' keys.
    """
    return {"segments": segments, "word_segments": []}


def _make_diarize_segments() -> dict:
    """Return mock diarization output suitable for assign_word_speakers."""
    return {"segments": [{"speaker": "SPEAKER_01"}]}


def _make_full_pipeline_mocks(
    catalog_db,
    media_id: str,
    segments: list[dict],
    language: str = "en",
    hf_token: str | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Set up all mocks for a complete pipeline run.

    Inserts a media file into catalog_db, configures a mock whisperx module
    and scheduler with complete pipeline behavior.

    Args:
        catalog_db: CatalogDB fixture instance.
        media_id: Media ID to use.
        segments: Segments for transcription result.
        language: Language code for transcription result.
        hf_token: Optional HuggingFace token.

    Returns:
        Tuple of (mock_whisperx, mock_scheduler).
    """
    catalog_db.insert_media(media_id, "/tmp/audio.wav")

    mock_wx = _make_mock_whisperx()

    # Configure transcribe result
    transcribe_result = _make_whisperx_transcribe_result(segments, language)
    mock_model = MagicMock()
    mock_model.transcribe.return_value = transcribe_result

    # Configure alignment result (adds word-level timestamps)
    aligned_segments = []
    for seg in segments:
        aligned_seg = dict(seg)
        if "words" not in aligned_seg:
            aligned_seg["words"] = [
                {
                    "word": w,
                    "start": seg.get("start", 0.0),
                    "end": seg.get("end", 0.0),
                    "score": 0.9,
                }
                for w in seg.get("text", "").split()
            ]
        aligned_segments.append(aligned_seg)
    mock_wx.align.return_value = _make_aligned_result(aligned_segments)

    # Configure diarization
    diarize_result = _make_diarize_segments()
    mock_wx.DiarizationPipeline.return_value.return_value = diarize_result

    # Configure assign_word_speakers to add speaker labels
    speaker_segments = []
    for seg in aligned_segments:
        speaker_seg = dict(seg)
        speaker_seg["speaker"] = "SPEAKER_01"
        speaker_segments.append(speaker_seg)
    mock_wx.assign_word_speakers.return_value = {"segments": speaker_segments}

    # Configure scheduler
    scheduler = MagicMock()
    scheduler.device = 0
    scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
    scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

    return mock_wx, scheduler
