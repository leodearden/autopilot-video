"""Tests for voiceover generation (autopilot.source.voiceover)."""

from __future__ import annotations

import inspect
import sys
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_kokoro() -> MagicMock:
    """Create a MagicMock module mimicking the kokoro TTS library."""
    mock_kokoro = MagicMock()
    mock_kokoro.KPipeline = MagicMock()
    return mock_kokoro


def _make_mock_soundfile() -> MagicMock:
    """Create a MagicMock module mimicking soundfile."""
    mock_sf = MagicMock()
    mock_sf.write = MagicMock()
    return mock_sf


def _make_model_config(tts_engine: str = "kokoro") -> MagicMock:
    """Create a mock ModelConfig with specified tts_engine."""
    config = MagicMock()
    config.tts_engine = tts_engine
    return config


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------


class TestVoiceoverPublicAPI:
    """Verify VoiceoverError, generate_voiceover surface."""

    def test_voiceover_error_importable(self):
        """VoiceoverError is importable from voiceover module."""
        from autopilot.source.voiceover import VoiceoverError

        assert VoiceoverError is not None

    def test_voiceover_error_is_exception(self):
        """VoiceoverError is a subclass of Exception."""
        from autopilot.source.voiceover import VoiceoverError

        assert issubclass(VoiceoverError, Exception)
        err = VoiceoverError("test error")
        assert str(err) == "test error"

    def test_generate_voiceover_signature(self):
        """generate_voiceover has text, output_path, config params, returns Path."""
        from autopilot.source.voiceover import generate_voiceover

        sig = inspect.signature(generate_voiceover)
        params = list(sig.parameters.keys())
        assert "text" in params
        assert "output_path" in params
        assert "config" in params

    def test_all_exports(self):
        """__all__ includes VoiceoverError and generate_voiceover."""
        from autopilot.source import voiceover

        assert "VoiceoverError" in voiceover.__all__
        assert "generate_voiceover" in voiceover.__all__


# ---------------------------------------------------------------------------
# Kokoro engine tests
# ---------------------------------------------------------------------------


class TestKokoroEngine:
    """Tests for the Kokoro TTS engine path."""

    def test_kokoro_creates_output_file(self, tmp_path):
        """Kokoro engine writes a .wav file to output_path."""
        mock_kokoro = _make_mock_kokoro()
        mock_sf = _make_mock_soundfile()

        # Mock pipeline that yields (graphemes, phonemes, audio_chunk)
        import numpy as np

        audio_chunk = np.zeros(24000, dtype=np.float32)
        pipeline_instance = MagicMock()
        pipeline_instance.return_value = iter([("hello", "h@loU", audio_chunk)])
        mock_kokoro.KPipeline.return_value = pipeline_instance

        output_path = tmp_path / "voice.wav"
        config = _make_model_config("kokoro")

        with patch.dict(sys.modules, {"kokoro": mock_kokoro, "soundfile": mock_sf}):
            if "autopilot.source.voiceover" in sys.modules:
                del sys.modules["autopilot.source.voiceover"]
            from autopilot.source.voiceover import generate_voiceover

            result = generate_voiceover("Hello world", output_path, config)

        assert result == output_path
        mock_sf.write.assert_called_once()

    def test_kokoro_pipeline_called_with_text(self, tmp_path):
        """Kokoro pipeline is invoked with the provided text."""
        mock_kokoro = _make_mock_kokoro()
        mock_sf = _make_mock_soundfile()

        import numpy as np

        audio_chunk = np.zeros(24000, dtype=np.float32)
        pipeline_instance = MagicMock()
        pipeline_instance.return_value = iter([("hello", "h@loU", audio_chunk)])
        mock_kokoro.KPipeline.return_value = pipeline_instance

        output_path = tmp_path / "voice.wav"
        config = _make_model_config("kokoro")

        with patch.dict(sys.modules, {"kokoro": mock_kokoro, "soundfile": mock_sf}):
            if "autopilot.source.voiceover" in sys.modules:
                del sys.modules["autopilot.source.voiceover"]
            from autopilot.source.voiceover import generate_voiceover

            generate_voiceover("Test narration text", output_path, config)

        pipeline_instance.assert_called_once()
        call_args = pipeline_instance.call_args
        assert "Test narration text" in str(call_args)


# ---------------------------------------------------------------------------
# ElevenLabs engine tests
# ---------------------------------------------------------------------------


class TestElevenLabsEngine:
    """Tests for the ElevenLabs API engine path."""

    def test_elevenlabs_calls_api(self, tmp_path):
        """ElevenLabs engine makes an API call and writes response to file."""
        output_path = tmp_path / "voice.wav"
        config = _make_model_config("elevenlabs")

        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=iter([b"fake_audio_data"]))
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.voiceover" in sys.modules:
                del sys.modules["autopilot.source.voiceover"]
            from autopilot.source.voiceover import generate_voiceover

            with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
                result = generate_voiceover("Hello world", output_path, config)

        assert result == output_path
        mock_requests.post.assert_called_once()

    def test_elevenlabs_uses_api_key_from_env(self, tmp_path):
        """ElevenLabs engine reads ELEVENLABS_API_KEY from environment."""
        output_path = tmp_path / "voice.wav"
        config = _make_model_config("elevenlabs")

        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=iter([b"audio_data"]))
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.voiceover" in sys.modules:
                del sys.modules["autopilot.source.voiceover"]
            from autopilot.source.voiceover import generate_voiceover

            with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "my-secret-key"}):
                generate_voiceover("Test text", output_path, config)

        # Verify the API key was passed in headers
        call_kwargs = mock_requests.post.call_args
        assert "my-secret-key" in str(call_kwargs)

    def test_elevenlabs_post_has_timeout(self, tmp_path):
        """ElevenLabs POST request includes a timeout parameter."""
        output_path = tmp_path / "voice.wav"
        config = _make_model_config("elevenlabs")

        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content = MagicMock(return_value=iter([b"fake_audio"]))
        mock_response.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_response

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.voiceover" in sys.modules:
                del sys.modules["autopilot.source.voiceover"]
            from autopilot.source.voiceover import generate_voiceover

            with patch.dict("os.environ", {"ELEVENLABS_API_KEY": "test-key"}):
                generate_voiceover("Hello world", output_path, config)

        call_kwargs = mock_requests.post.call_args
        assert call_kwargs.kwargs.get("timeout") is not None, (
            "ElevenLabs POST request must include a timeout parameter"
        )

    def test_elevenlabs_missing_api_key_raises(self, tmp_path):
        """ElevenLabs engine raises VoiceoverError when API key is missing."""
        output_path = tmp_path / "voice.wav"
        config = _make_model_config("elevenlabs")

        if "autopilot.source.voiceover" in sys.modules:
            del sys.modules["autopilot.source.voiceover"]
        from autopilot.source.voiceover import VoiceoverError, generate_voiceover

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("ELEVENLABS_API_KEY", None)
            with pytest.raises(VoiceoverError, match="ELEVENLABS_API_KEY"):
                generate_voiceover("Hello", output_path, config)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestVoiceoverErrors:
    """Error handling in voiceover generation."""

    def test_unknown_engine_raises(self, tmp_path):
        """Unknown tts_engine value raises VoiceoverError."""
        output_path = tmp_path / "voice.wav"
        config = _make_model_config("unknown_engine")

        with patch.dict(sys.modules):
            if "autopilot.source.voiceover" in sys.modules:
                del sys.modules["autopilot.source.voiceover"]
            from autopilot.source.voiceover import VoiceoverError, generate_voiceover

            with pytest.raises(VoiceoverError, match="unknown_engine"):
                generate_voiceover("Hello", output_path, config)
