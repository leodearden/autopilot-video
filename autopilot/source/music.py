"""Music sourcing via MusicGen (local) or Freesound API."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autopilot.config import ModelConfig
    from autopilot.source import MusicRequest

__all__ = [
    "MusicError",
    "source_music",
]

logger = logging.getLogger(__name__)

_FREESOUND_API_URL = "https://freesound.org/apiv2"


class MusicError(Exception):
    """Raised for any music sourcing error."""


# Module-level cache for MusicGen model (avoids 10-30s reload per call).
_musicgen_cache: dict[str, Any] = {}


def _get_musicgen_model(name: str) -> Any:
    """Get or create a cached MusicGen model instance.

    Args:
        name: Model name (e.g. 'facebook/musicgen-small').

    Returns:
        MusicGen model instance.
    """
    if name not in _musicgen_cache:
        from audiocraft.models.musicgen import MusicGen

        _musicgen_cache[name] = MusicGen.get_pretrained(name)
    return _musicgen_cache[name]


def source_music(request: MusicRequest, config: ModelConfig, output_dir: Path) -> Path | None:
    """Source a music track matching the request.

    Dispatches based on config.music_engine:
    - 'musicgen': Generate locally using audiocraft's MusicGen model.
    - 'fetch_list_only': Skip generation, return None (deferred to manual fetch).

    Falls back to Freesound API search if MusicGen is unavailable or fails.

    Args:
        request: MusicRequest with mood, duration, start_time.
        config: ModelConfig with music_engine selection.
        output_dir: Directory where generated/downloaded files are saved.

    Returns:
        Path to the generated/downloaded audio file, or None if unresolved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.music_engine == "fetch_list_only":
        logger.info("Music engine is fetch_list_only; skipping generation")
        return None

    if config.music_engine == "musicgen":
        try:
            return _generate_musicgen(request, output_dir)
        except MusicError as exc:
            logger.warning(
                "MusicGen failed for mood=%r, trying Freesound: %s",
                request.mood,
                exc,
            )
            return _search_freesound(request, output_dir)

    raise MusicError(
        f"Unknown music engine: {config.music_engine!r}. Supported: musicgen, fetch_list_only"
    )


def _generate_musicgen(request: MusicRequest, output_dir: Path) -> Path:
    """Generate music using audiocraft's MusicGen model.

    Args:
        request: MusicRequest with mood and duration.
        output_dir: Directory for the generated file.

    Returns:
        Path to the generated .wav file.

    Raises:
        MusicError: If generation fails.
    """
    try:
        import audiocraft.models  # noqa: F401 — ensure audiocraft is importable
        import torchaudio
    except ImportError as e:
        raise MusicError(f"MusicGen dependencies not installed: {e}") from e

    try:
        logger.info(
            "Generating music with MusicGen: mood=%r, duration=%.1fs",
            request.mood,
            request.duration,
        )

        model = _get_musicgen_model("facebook/musicgen-small")
        model.set_generation_params(duration=request.duration)

        wav = model.generate([request.mood])

        # Create a deterministic filename from the mood
        mood_hash = hashlib.md5(request.mood.encode()).hexdigest()[:8]
        filename = f"music_{mood_hash}_{request.start_time.replace(':', '-')}.wav"
        output_path = output_dir / filename

        # Save the generated audio
        torchaudio.save(
            str(output_path),
            wav[0].cpu(),
            model.sample_rate,
        )

        logger.info("MusicGen output saved to %s", output_path)
        return output_path

    except MusicError:
        raise
    except Exception as e:
        raise MusicError(f"MusicGen generation failed: {e}") from e


def _search_freesound(request: MusicRequest, output_dir: Path) -> Path | None:
    """Search Freesound for music matching the mood.

    Args:
        request: MusicRequest with mood to search for.
        output_dir: Directory to download results into.

    Returns:
        Path to the downloaded audio file, or None if no results.
    """
    api_key = os.environ.get("FREESOUND_API_KEY")
    if not api_key:
        logger.warning("FREESOUND_API_KEY not set; cannot search Freesound")
        return None

    try:
        import requests as _requests
    except ImportError:
        logger.warning("requests library not installed; cannot search Freesound")
        return None

    try:
        logger.info("Searching Freesound for mood=%r", request.mood)

        response = _requests.get(
            f"{_FREESOUND_API_URL}/search/text/",
            params={
                "query": request.mood,
                "filter": f"duration:[{max(1, request.duration - 10)} TO {request.duration + 30}]",
                "fields": "id,name,previews",
                "token": api_key,
                "page_size": 5,
            },
            timeout=(10, 60),
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            logger.info("No Freesound results for mood=%r", request.mood)
            return None

        # Download the first result's preview
        top = results[0]
        preview_url = top.get("previews", {}).get("preview-hq-mp3")
        if not preview_url:
            logger.warning("No preview URL for Freesound result %s", top.get("id"))
            return None

        dl_response = _requests.get(preview_url, timeout=(10, 60))
        dl_response.raise_for_status()

        safe_name = re.sub(r"[^\w\-. ]", "_", top.get("name", "track"))[:50]
        filename = f"freesound_{top['id']}_{safe_name}"
        output_path = output_dir / filename
        output_path.write_bytes(dl_response.content)

        logger.info("Downloaded Freesound track to %s", output_path)
        return output_path

    except Exception as e:
        logger.warning("Freesound search failed: %s", e)
        return None
