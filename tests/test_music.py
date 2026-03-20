"""Tests for music sourcing (autopilot.source.music)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from autopilot.source import MusicRequest

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

def _make_mock_audiocraft() -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Create mock audiocraft and torchaudio modules.

    Returns:
        Tuple of (mock_audiocraft, mock_torchaudio).
    """
    mock_audiocraft = MagicMock()
    mock_audiocraft_models = MagicMock()
    mock_audiocraft.models = mock_audiocraft_models

    mock_model = MagicMock()
    mock_model.set_generation_params = MagicMock()
    # generate returns a tensor-like mock; wav[0].cpu() must return a MagicMock
    mock_tensor = MagicMock()
    mock_model.generate.return_value = mock_tensor
    mock_model.sample_rate = 32000
    mock_audiocraft_models.MusicGen.get_pretrained.return_value = mock_model

    mock_torchaudio = MagicMock()
    mock_torchaudio.save = MagicMock()

    mock_torch = MagicMock()

    return mock_audiocraft, mock_audiocraft_models, mock_torchaudio, mock_torch


def _make_model_config(music_engine: str = "musicgen") -> MagicMock:
    """Create a mock ModelConfig with specified music_engine."""
    config = MagicMock()
    config.music_engine = music_engine
    return config


def _make_music_request(
    mood: str = "upbeat acoustic",
    duration: float = 30.0,
    start_time: str = "00:01:00.000",
) -> MusicRequest:
    """Create a MusicRequest for testing."""
    return MusicRequest(mood=mood, duration=duration, start_time=start_time)


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------

class TestMusicPublicAPI:
    """Verify MusicError, source_music surface."""

    def test_music_error_importable(self):
        """MusicError is importable from music module."""
        from autopilot.source.music import MusicError

        assert MusicError is not None

    def test_music_error_is_exception(self):
        """MusicError is a subclass of Exception."""
        from autopilot.source.music import MusicError

        assert issubclass(MusicError, Exception)
        err = MusicError("test error")
        assert str(err) == "test error"

    def test_source_music_signature(self):
        """source_music has request, config, output_dir params."""
        from autopilot.source.music import source_music

        sig = inspect.signature(source_music)
        params = list(sig.parameters.keys())
        assert "request" in params
        assert "config" in params
        assert "output_dir" in params

    def test_all_exports(self):
        """__all__ includes MusicError and source_music."""
        from autopilot.source import music

        assert "MusicError" in music.__all__
        assert "source_music" in music.__all__


# ---------------------------------------------------------------------------
# MusicGen engine tests
# ---------------------------------------------------------------------------

class TestMusicGenEngine:
    """Tests for the MusicGen (audiocraft) engine path."""

    def test_musicgen_generates_audio(self, tmp_path):
        """MusicGen engine generates and saves an audio file."""
        mock_ac, mock_ac_models, mock_ta, mock_torch = _make_mock_audiocraft()
        request = _make_music_request()
        config = _make_model_config("musicgen")

        with patch.dict(sys.modules, {
            "audiocraft": mock_ac,
            "audiocraft.models": mock_ac_models,
            "torchaudio": mock_ta,
            "torch": mock_torch,
        }):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import source_music

            result = source_music(request, config, tmp_path)

        assert result is not None
        assert isinstance(result, Path)

    def test_musicgen_calls_generate_with_mood(self, tmp_path):
        """MusicGen model.generate is called with the mood description."""
        mock_ac, mock_ac_models, mock_ta, mock_torch = _make_mock_audiocraft()
        request = _make_music_request(mood="gentle piano")
        config = _make_model_config("musicgen")

        with patch.dict(sys.modules, {
            "audiocraft": mock_ac,
            "audiocraft.models": mock_ac_models,
            "torchaudio": mock_ta,
            "torch": mock_torch,
        }):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import source_music

            source_music(request, config, tmp_path)

        mock_ac_models.MusicGen.get_pretrained.assert_called_once()
        model = mock_ac_models.MusicGen.get_pretrained.return_value
        model.generate.assert_called_once()
        call_args = model.generate.call_args
        assert "gentle piano" in str(call_args)

    def test_musicgen_model_cached_across_calls(self, tmp_path):
        """MusicGen model is loaded once and cached across multiple calls."""
        mock_ac, mock_ac_models, mock_ta, mock_torch = _make_mock_audiocraft()
        request1 = _make_music_request(mood="upbeat acoustic")
        request2 = _make_music_request(mood="gentle piano")
        config = _make_model_config("musicgen")

        with patch.dict(sys.modules, {
            "audiocraft": mock_ac,
            "audiocraft.models": mock_ac_models,
            "torchaudio": mock_ta,
            "torch": mock_torch,
        }):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import _musicgen_cache, source_music

            # Clear cache to start fresh
            _musicgen_cache.clear()

            source_music(request1, config, tmp_path)
            source_music(request2, config, tmp_path)

        # get_pretrained should be called only ONCE (cached on second call)
        assert mock_ac_models.MusicGen.get_pretrained.call_count == 1, (
            "MusicGen.get_pretrained should be called once and cached, "
            f"but was called {mock_ac_models.MusicGen.get_pretrained.call_count} times"
        )

    def test_musicgen_cache_clearable(self, tmp_path):
        """_musicgen_cache can be cleared for test isolation."""
        mock_ac, mock_ac_models, mock_ta, mock_torch = _make_mock_audiocraft()

        with patch.dict(sys.modules, {
            "audiocraft": mock_ac,
            "audiocraft.models": mock_ac_models,
            "torchaudio": mock_ta,
            "torch": mock_torch,
        }):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import _musicgen_cache

            _musicgen_cache["test"] = "value"
            assert len(_musicgen_cache) == 1
            _musicgen_cache.clear()
            assert len(_musicgen_cache) == 0

    def test_musicgen_saves_to_output_dir(self, tmp_path):
        """MusicGen saves file within the provided output_dir."""
        mock_ac, mock_ac_models, mock_ta, mock_torch = _make_mock_audiocraft()
        request = _make_music_request()
        config = _make_model_config("musicgen")

        with patch.dict(sys.modules, {
            "audiocraft": mock_ac,
            "audiocraft.models": mock_ac_models,
            "torchaudio": mock_ta,
            "torch": mock_torch,
        }):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import source_music

            source_music(request, config, tmp_path)

        # torchaudio.save should have been called
        mock_ta.save.assert_called_once()
        # The saved path should be under output_dir
        saved_path = mock_ta.save.call_args[0][0]
        assert str(tmp_path) in str(saved_path)


# ---------------------------------------------------------------------------
# Freesound search tests
# ---------------------------------------------------------------------------

class TestFreesoundSearch:
    """Tests for the Freesound API search fallback."""

    def test_freesound_search_with_results(self, tmp_path):
        """Freesound API returns results and downloads audio."""
        mock_requests = MagicMock()
        # Search response
        search_response = MagicMock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "results": [
                {
                    "id": 12345,
                    "name": "upbeat-music.wav",
                    "previews": {"preview-hq-mp3": "https://example.com/audio.mp3"},
                }
            ]
        }
        search_response.raise_for_status = MagicMock()
        # Download response
        download_response = MagicMock()
        download_response.status_code = 200
        download_response.content = b"fake_audio_data"
        download_response.raise_for_status = MagicMock()
        mock_requests.get.side_effect = [search_response, download_response]

        request = _make_music_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import _search_freesound

            with patch.dict("os.environ", {"FREESOUND_API_KEY": "test-key"}):
                result = _search_freesound(request, tmp_path)

        assert result is not None
        assert isinstance(result, Path)

    def test_freesound_empty_results_returns_none(self, tmp_path):
        """Freesound API returns empty results → None."""
        mock_requests = MagicMock()
        search_response = MagicMock()
        search_response.status_code = 200
        search_response.json.return_value = {"results": []}
        search_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = search_response

        request = _make_music_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import _search_freesound

            with patch.dict("os.environ", {"FREESOUND_API_KEY": "test-key"}):
                result = _search_freesound(request, tmp_path)

        assert result is None

    def test_freesound_uses_api_key_from_env(self, tmp_path):
        """Freesound search uses FREESOUND_API_KEY env var."""
        mock_requests = MagicMock()
        search_response = MagicMock()
        search_response.status_code = 200
        search_response.json.return_value = {"results": []}
        search_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = search_response

        request = _make_music_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import _search_freesound

            with patch.dict("os.environ", {"FREESOUND_API_KEY": "my-fs-key"}):
                _search_freesound(request, tmp_path)

        # Verify API key was used in the request
        call_args = mock_requests.get.call_args
        assert "my-fs-key" in str(call_args)


# ---------------------------------------------------------------------------
# fetch_list_only config tests
# ---------------------------------------------------------------------------

    def test_freesound_search_has_timeout(self, tmp_path):
        """Freesound search API call includes a timeout parameter."""
        mock_requests = MagicMock()
        search_response = MagicMock()
        search_response.status_code = 200
        search_response.json.return_value = {"results": []}
        search_response.raise_for_status = MagicMock()
        mock_requests.get.return_value = search_response

        request = _make_music_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import _search_freesound

            with patch.dict("os.environ", {"FREESOUND_API_KEY": "test-key"}):
                _search_freesound(request, tmp_path)

        call_kwargs = mock_requests.get.call_args
        assert call_kwargs.kwargs.get("timeout") is not None, (
            "Freesound search request must include a timeout parameter"
        )

    def test_freesound_download_has_timeout(self, tmp_path):
        """Freesound download call includes a timeout parameter."""
        mock_requests = MagicMock()
        search_response = MagicMock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "results": [
                {
                    "id": 12345,
                    "name": "track.wav",
                    "previews": {"preview-hq-mp3": "https://example.com/audio.mp3"},
                }
            ]
        }
        search_response.raise_for_status = MagicMock()
        download_response = MagicMock()
        download_response.status_code = 200
        download_response.content = b"fake_audio"
        download_response.raise_for_status = MagicMock()
        mock_requests.get.side_effect = [search_response, download_response]

        request = _make_music_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import _search_freesound

            with patch.dict("os.environ", {"FREESOUND_API_KEY": "test-key"}):
                _search_freesound(request, tmp_path)

        # Download call is the second .get() call
        download_call = mock_requests.get.call_args_list[1]
        assert download_call.kwargs.get("timeout") is not None, (
            "Freesound download request must include a timeout parameter"
        )

    def test_freesound_sanitizes_filename(self, tmp_path):
        """Freesound track names with path separators/special chars are sanitized."""
        mock_requests = MagicMock()
        search_response = MagicMock()
        search_response.status_code = 200
        search_response.json.return_value = {
            "results": [
                {
                    "id": 99999,
                    "name": "my/track\\name<>:.wav\x00evil",
                    "previews": {"preview-hq-mp3": "https://example.com/audio.mp3"},
                }
            ]
        }
        search_response.raise_for_status = MagicMock()
        download_response = MagicMock()
        download_response.status_code = 200
        download_response.content = b"fake_audio"
        download_response.raise_for_status = MagicMock()
        mock_requests.get.side_effect = [search_response, download_response]

        request = _make_music_request()

        with patch.dict(sys.modules, {"requests": mock_requests}):
            if "autopilot.source.music" in sys.modules:
                del sys.modules["autopilot.source.music"]
            from autopilot.source.music import _search_freesound

            with patch.dict("os.environ", {"FREESOUND_API_KEY": "test-key"}):
                result = _search_freesound(request, tmp_path)

        assert result is not None
        filename = result.name
        # Must not contain path separators or dangerous characters
        for char in ['/', '\\', '<', '>', ':', '\x00']:
            assert char not in filename, (
                f"Filename {filename!r} contains unsafe character {char!r}"
            )
        # File must be directly in output_dir, not a subdirectory
        assert result.parent == tmp_path


class TestFetchListOnly:
    """Tests for fetch_list_only engine mode."""

    def test_fetch_list_only_returns_none(self, tmp_path):
        """When music_engine is fetch_list_only, source_music returns None."""
        request = _make_music_request()
        config = _make_model_config("fetch_list_only")

        if "autopilot.source.music" in sys.modules:
            del sys.modules["autopilot.source.music"]
        from autopilot.source.music import source_music

        result = source_music(request, config, tmp_path)
        assert result is None
