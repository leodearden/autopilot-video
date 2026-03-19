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


# -- Test classes --------------------------------------------------------------


class TestPublicAPI:
    """Verify public API surface and type signatures."""

    def test_exports_importable(self):
        """TranscriptionError and transcribe_media are importable."""
        from autopilot.analyze.asr import TranscriptionError, transcribe_media

        assert TranscriptionError is not None
        assert transcribe_media is not None

    def test_transcription_error_is_exception(self):
        """TranscriptionError is a subclass of Exception with message."""
        from autopilot.analyze.asr import TranscriptionError

        assert issubclass(TranscriptionError, Exception)
        err = TranscriptionError("test message")
        assert str(err) == "test message"

    def test_transcribe_media_signature(self):
        """transcribe_media has correct parameter signature."""
        from autopilot.analyze.asr import transcribe_media

        sig = inspect.signature(transcribe_media)
        params = list(sig.parameters.keys())

        # Positional parameters
        assert "media_id" in params
        assert "audio_path" in params
        assert "db" in params
        assert "scheduler" in params
        assert "config" in params

        # Keyword-only parameters with defaults
        batch_size_param = sig.parameters["batch_size"]
        assert batch_size_param.kind == inspect.Parameter.KEYWORD_ONLY
        assert batch_size_param.default == 24

        hf_token_param = sig.parameters["hf_token"]
        assert hf_token_param.kind == inspect.Parameter.KEYWORD_ONLY
        assert hf_token_param.default is None

        # Return annotation (string 'None' due to __future__ annotations)
        assert sig.return_annotation in (None, "None")


class TestNormalizeSegments:
    """Tests for _normalize_segments() private helper."""

    def test_complete_segment(self):
        """Input with all PRD fields produces correct output dict."""
        from autopilot.analyze.asr import _normalize_segments

        segments = [{
            "start": 0.5,
            "end": 2.3,
            "text": "Hello world",
            "speaker": "SPEAKER_01",
            "words": [
                {"word": "Hello", "start": 0.5, "end": 1.0, "score": 0.95},
                {"word": "world", "start": 1.1, "end": 2.3, "score": 0.88},
            ],
        }]
        result = _normalize_segments(segments)
        assert len(result) == 1
        seg = result[0]
        assert seg["start"] == 0.5
        assert seg["end"] == 2.3
        assert seg["text"] == "Hello world"
        assert seg["speaker"] == "SPEAKER_01"
        assert len(seg["words"]) == 2
        assert seg["words"][0] == {
            "word": "Hello", "start": 0.5, "end": 1.0, "score": 0.95,
        }
        assert seg["words"][1] == {
            "word": "world", "start": 1.1, "end": 2.3, "score": 0.88,
        }

    def test_multiple_segments(self):
        """Three segments all formatted correctly."""
        from autopilot.analyze.asr import _normalize_segments

        segments = [
            {"start": 0.0, "end": 1.0, "text": "First"},
            {"start": 1.0, "end": 2.0, "text": "Second"},
            {"start": 2.0, "end": 3.0, "text": "Third"},
        ]
        result = _normalize_segments(segments)
        assert len(result) == 3
        assert [s["text"] for s in result] == ["First", "Second", "Third"]

    def test_empty_segments(self):
        """Empty list input returns empty list."""
        from autopilot.analyze.asr import _normalize_segments

        assert _normalize_segments([]) == []

    def test_missing_speaker_defaults_to_none(self):
        """Segment without 'speaker' key gets speaker=None."""
        from autopilot.analyze.asr import _normalize_segments

        segments = [{"start": 0.0, "end": 1.0, "text": "Test"}]
        result = _normalize_segments(segments)
        assert result[0]["speaker"] is None

    def test_missing_words_defaults_to_empty(self):
        """Segment without 'words' key gets words=[]."""
        from autopilot.analyze.asr import _normalize_segments

        segments = [{"start": 0.0, "end": 1.0, "text": "Test"}]
        result = _normalize_segments(segments)
        assert result[0]["words"] == []

    def test_word_missing_score_defaults_to_zero(self):
        """Word without 'score' gets score=0.0."""
        from autopilot.analyze.asr import _normalize_segments

        segments = [{
            "start": 0.0,
            "end": 1.0,
            "text": "Test",
            "words": [{"word": "Test", "start": 0.0, "end": 1.0}],
        }]
        result = _normalize_segments(segments)
        assert result[0]["words"][0]["score"] == 0.0

    def test_strips_extra_fields(self):
        """WhisperX internal fields are not present in output."""
        from autopilot.analyze.asr import _normalize_segments

        segments = [{
            "start": 0.0,
            "end": 1.0,
            "text": "Test",
            "tokens": [123, 456],
            "avg_logprob": -0.5,
            "temperature": 0.0,
            "compression_ratio": 1.2,
            "no_speech_prob": 0.01,
        }]
        result = _normalize_segments(segments)
        seg = result[0]
        assert "tokens" not in seg
        assert "avg_logprob" not in seg
        assert "temperature" not in seg
        assert "compression_ratio" not in seg
        assert "no_speech_prob" not in seg
        # Only PRD keys present
        assert set(seg.keys()) == {"start", "end", "text", "speaker", "words"}

    def test_json_serializable(self):
        """json.dumps succeeds on output, no numpy types leak."""
        from autopilot.analyze.asr import _normalize_segments

        import numpy as np

        segments = [{
            "start": np.float32(0.5),
            "end": np.float64(2.3),
            "text": "Hello",
            "words": [{
                "word": "Hello",
                "start": np.float32(0.5),
                "end": np.float64(2.3),
                "score": np.float32(0.95),
            }],
        }]
        result = _normalize_segments(segments)
        # Should not raise
        serialized = json.dumps(result)
        assert isinstance(serialized, str)


class TestIdempotency:
    """Tests for transcript idempotency check."""

    def test_skips_when_transcript_exists(self, catalog_db):
        """Skip transcription when transcript already exists."""
        from autopilot.analyze.asr import transcribe_media

        catalog_db.insert_media("vid1", "/tmp/audio.wav")
        catalog_db.upsert_transcript("vid1", '{"segments": []}', "en")

        scheduler = MagicMock()
        config = MagicMock()
        config.whisper_size = "large-v3"

        transcribe_media(
            "vid1",
            Path("/tmp/audio.wav"),
            catalog_db,
            scheduler,
            config,
        )

        # Scheduler should NOT be called for model loading
        scheduler.model.assert_not_called()

    def test_proceeds_when_no_transcript(self, catalog_db):
        """Proceed with transcription when no transcript exists."""
        from autopilot.analyze.asr import transcribe_media

        mock_wx, scheduler = _make_full_pipeline_mocks(
            catalog_db,
            "vid1",
            [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        )

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch.object(Path, "exists", return_value=True):
                transcribe_media(
                    "vid1",
                    Path("/tmp/audio.wav"),
                    catalog_db,
                    scheduler,
                    MagicMock(whisper_size="large-v3"),
                )

        # Scheduler SHOULD be called for model loading
        scheduler.model.assert_called_once()


class TestInputValidation:
    """Tests for audio path validation."""

    def test_raises_on_missing_audio(self, catalog_db):
        """TranscriptionError raised for non-existent audio file."""
        from autopilot.analyze.asr import TranscriptionError, transcribe_media

        catalog_db.insert_media("vid1", "/nonexistent/audio.wav")

        scheduler = MagicMock()
        config = MagicMock()
        config.whisper_size = "large-v3"

        with pytest.raises(TranscriptionError, match="not found"):
            transcribe_media(
                "vid1",
                Path("/nonexistent/audio.wav"),
                catalog_db,
                scheduler,
                config,
            )

    def test_audio_path_validated_before_whisperx(self, catalog_db):
        """Path validation happens before whisperx is touched."""
        from autopilot.analyze.asr import TranscriptionError, transcribe_media

        catalog_db.insert_media("vid1", "/nonexistent/audio.wav")

        scheduler = MagicMock()
        config = MagicMock()
        config.whisper_size = "large-v3"

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(TranscriptionError):
                transcribe_media(
                    "vid1",
                    Path("/tmp/audio.wav"),
                    catalog_db,
                    scheduler,
                    config,
                )

        # Scheduler should NOT be called (failed before whisperx)
        scheduler.model.assert_not_called()
