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
        import numpy as np

        from autopilot.analyze.asr import _normalize_segments

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


class TestStatusUpdate:
    """Tests for media status update."""

    def test_sets_status_analyzing(self, catalog_db):
        """Media status updated to 'analyzing' during transcription."""
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

        media = catalog_db.get_media("vid1")
        assert media is not None
        assert media["status"] == "analyzing"


class TestTranscription:
    """Tests for core transcription pipeline."""

    def test_loads_audio_and_transcribes(self, catalog_db):
        """Verify whisperx.load_audio and model.transcribe are called."""
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

        # Verify load_audio called with string path
        mock_wx.load_audio.assert_called_once_with(str(Path("/tmp/audio.wav")))

        # Verify model.transcribe called with audio and batch_size
        mock_model = scheduler.model.return_value.__enter__.return_value
        mock_model.transcribe.assert_called_once()
        call_args = mock_model.transcribe.call_args
        assert call_args[0][0] == "mock_audio_array"
        assert call_args[1]["batch_size"] == 24

    def test_custom_batch_size(self, catalog_db):
        """Custom batch_size passed through to model.transcribe."""
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
                    batch_size=16,
                )

        mock_model = scheduler.model.return_value.__enter__.return_value
        call_args = mock_model.transcribe.call_args
        assert call_args[1]["batch_size"] == 16

    def test_uses_config_whisper_size(self, catalog_db):
        """scheduler.model called with config.whisper_size."""
        from autopilot.analyze.asr import transcribe_media

        mock_wx, scheduler = _make_full_pipeline_mocks(
            catalog_db,
            "vid1",
            [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        )

        config = MagicMock()
        config.whisper_size = "large-v3-turbo"

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch.object(Path, "exists", return_value=True):
                transcribe_media(
                    "vid1",
                    Path("/tmp/audio.wav"),
                    catalog_db,
                    scheduler,
                    config,
                )

        scheduler.model.assert_called_once_with("large-v3-turbo")


class TestForcedAlignment:
    """Tests for wav2vec2 forced alignment stage."""

    def test_runs_alignment(self, catalog_db):
        """Alignment model loaded and whisperx.align called."""
        from autopilot.analyze.asr import transcribe_media

        mock_wx, scheduler = _make_full_pipeline_mocks(
            catalog_db,
            "vid1",
            [{"start": 0.0, "end": 1.0, "text": "Hello world"}],
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

        # Verify load_align_model called with language and device
        mock_wx.load_align_model.assert_called_once()
        call_kwargs = mock_wx.load_align_model.call_args
        assert call_kwargs[1]["language_code"] == "en"
        assert "cuda" in call_kwargs[1]["device"]

        # Verify align called
        mock_wx.align.assert_called_once()

    def test_alignment_failure_continues(self, catalog_db, caplog):
        """Alignment failure logs warning but transcript still stored."""
        from autopilot.analyze.asr import transcribe_media

        mock_wx, scheduler = _make_full_pipeline_mocks(
            catalog_db,
            "vid1",
            [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        )
        # Make alignment fail
        mock_wx.align.side_effect = RuntimeError("alignment failed")

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch.object(Path, "exists", return_value=True):
                with caplog.at_level(logging.WARNING):
                    transcribe_media(
                        "vid1",
                        Path("/tmp/audio.wav"),
                        catalog_db,
                        scheduler,
                        MagicMock(whisper_size="large-v3"),
                    )

        # Should NOT raise, transcript should be stored
        transcript = catalog_db.get_transcript("vid1")
        assert transcript is not None

        # Warning should be logged
        assert any("align" in r.message.lower() for r in caplog.records
                    if r.levelno >= logging.WARNING)


class TestDiarization:
    """Tests for pyannote speaker diarization stage."""

    def test_runs_diarization_with_explicit_token(self, catalog_db):
        """DiarizationPipeline called with explicit hf_token."""
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
                    hf_token="test-token",
                )

        # Verify DiarizationPipeline called with token and device
        mock_wx.DiarizationPipeline.assert_called_once()
        call_kwargs = mock_wx.DiarizationPipeline.call_args[1]
        assert call_kwargs["use_auth_token"] == "test-token"
        assert "cuda" in call_kwargs["device"]

        # Verify assign_word_speakers called
        mock_wx.assign_word_speakers.assert_called_once()

    def test_uses_env_token_as_fallback(self, catalog_db, monkeypatch):
        """HF_TOKEN env var used when hf_token not passed."""
        from autopilot.analyze.asr import transcribe_media

        monkeypatch.setenv("HF_TOKEN", "env-token")

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

        call_kwargs = mock_wx.DiarizationPipeline.call_args[1]
        assert call_kwargs["use_auth_token"] == "env-token"

    def test_skips_without_token(self, catalog_db, monkeypatch):
        """No diarization when no HF token available."""
        from autopilot.analyze.asr import transcribe_media

        monkeypatch.delenv("HF_TOKEN", raising=False)

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

        # DiarizationPipeline should NOT be called
        mock_wx.DiarizationPipeline.assert_not_called()

    def test_diarization_failure_continues(self, catalog_db, caplog):
        """Diarization failure logs warning but transcript still stored."""
        from autopilot.analyze.asr import transcribe_media

        mock_wx, scheduler = _make_full_pipeline_mocks(
            catalog_db,
            "vid1",
            [{"start": 0.0, "end": 1.0, "text": "Hello"}],
        )
        # Make diarization fail
        mock_wx.DiarizationPipeline.side_effect = RuntimeError("diarization failed")

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch.object(Path, "exists", return_value=True):
                with caplog.at_level(logging.WARNING):
                    transcribe_media(
                        "vid1",
                        Path("/tmp/audio.wav"),
                        catalog_db,
                        scheduler,
                        MagicMock(whisper_size="large-v3"),
                        hf_token="test-token",
                    )

        # Should NOT raise, transcript should be stored
        transcript = catalog_db.get_transcript("vid1")
        assert transcript is not None

        # Warning should be logged
        assert any("diariz" in r.message.lower() for r in caplog.records
                    if r.levelno >= logging.WARNING)


class TestDBStorage:
    """Tests for transcript DB storage."""

    def test_stores_normalized_transcript(self, catalog_db):
        """Stored transcript matches PRD schema."""
        from autopilot.analyze.asr import transcribe_media

        segments = [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "Hello world",
                "speaker": "SPEAKER_01",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.7, "score": 0.95},
                    {"word": "world", "start": 0.8, "end": 1.5, "score": 0.88},
                ],
            },
            {
                "start": 2.0,
                "end": 3.5,
                "text": "How are you",
                "speaker": "SPEAKER_02",
                "words": [
                    {"word": "How", "start": 2.0, "end": 2.3, "score": 0.9},
                    {"word": "are", "start": 2.4, "end": 2.6, "score": 0.85},
                    {"word": "you", "start": 2.7, "end": 3.5, "score": 0.92},
                ],
            },
        ]

        mock_wx, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", segments,
        )
        # Make assign_word_speakers return segments with speakers
        mock_wx.assign_word_speakers.return_value = {"segments": segments}

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch.object(Path, "exists", return_value=True):
                transcribe_media(
                    "vid1",
                    Path("/tmp/audio.wav"),
                    catalog_db,
                    scheduler,
                    MagicMock(whisper_size="large-v3"),
                )

        transcript = catalog_db.get_transcript("vid1")
        assert transcript is not None
        stored = json.loads(str(transcript["segments_json"]))
        assert len(stored) == 2

        # Verify PRD schema fields
        seg0 = stored[0]
        assert seg0["start"] == 0.0
        assert seg0["end"] == 1.5
        assert seg0["text"] == "Hello world"
        assert seg0["speaker"] == "SPEAKER_01"
        assert len(seg0["words"]) == 2
        assert seg0["words"][0] == {
            "word": "Hello", "start": 0.0, "end": 0.7, "score": 0.95,
        }

    def test_language_stored(self, catalog_db):
        """Stored language matches whisperx detected language."""
        from autopilot.analyze.asr import transcribe_media

        mock_wx, scheduler = _make_full_pipeline_mocks(
            catalog_db,
            "vid1",
            [{"start": 0.0, "end": 1.0, "text": "Bonjour"}],
            language="fr",
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

        transcript = catalog_db.get_transcript("vid1")
        assert transcript is not None
        assert transcript["language"] == "fr"

    def test_empty_segments_stored(self, catalog_db):
        """Empty segments list stored and retrievable."""
        from autopilot.analyze.asr import transcribe_media

        mock_wx, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", [],
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

        transcript = catalog_db.get_transcript("vid1")
        assert transcript is not None
        stored = json.loads(str(transcript["segments_json"]))
        assert stored == []


class TestErrorHandling:
    """Tests for error handling in core pipeline."""

    def test_transcription_failure_raises(self, catalog_db):
        """RuntimeError in model.transcribe raises TranscriptionError."""
        from autopilot.analyze.asr import TranscriptionError, transcribe_media

        catalog_db.insert_media("vid1", "/tmp/audio.wav")

        mock_wx = _make_mock_whisperx()
        mock_model = MagicMock()
        mock_model.transcribe.side_effect = RuntimeError("CUDA OOM")

        scheduler = MagicMock()
        scheduler.device = 0
        scheduler.model.return_value.__enter__ = MagicMock(
            return_value=mock_model
        )
        scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(TranscriptionError, match="vid1"):
                    transcribe_media(
                        "vid1",
                        Path("/tmp/audio.wav"),
                        catalog_db,
                        scheduler,
                        MagicMock(whisper_size="large-v3"),
                    )

    def test_audio_load_failure_raises(self, catalog_db):
        """FileNotFoundError in load_audio raises TranscriptionError."""
        from autopilot.analyze.asr import TranscriptionError, transcribe_media

        catalog_db.insert_media("vid1", "/tmp/audio.wav")

        mock_wx = _make_mock_whisperx()
        mock_wx.load_audio.side_effect = FileNotFoundError("no such file")

        scheduler = MagicMock()
        scheduler.device = 0

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(TranscriptionError):
                    transcribe_media(
                        "vid1",
                        Path("/tmp/audio.wav"),
                        catalog_db,
                        scheduler,
                        MagicMock(whisper_size="large-v3"),
                    )

    def test_failure_does_not_store_transcript(self, catalog_db):
        """No partial transcript stored after TranscriptionError."""
        from autopilot.analyze.asr import TranscriptionError, transcribe_media

        catalog_db.insert_media("vid1", "/tmp/audio.wav")

        mock_wx = _make_mock_whisperx()
        mock_wx.load_audio.side_effect = RuntimeError("load failed")

        scheduler = MagicMock()
        scheduler.device = 0

        with patch.dict(sys.modules, {"whisperx": mock_wx}):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(TranscriptionError):
                    transcribe_media(
                        "vid1",
                        Path("/tmp/audio.wav"),
                        catalog_db,
                        scheduler,
                        MagicMock(whisper_size="large-v3"),
                    )

        # No transcript should be stored
        assert catalog_db.get_transcript("vid1") is None
