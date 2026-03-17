"""Audio normalizer — EBU R128 loudness normalization via ffmpeg."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_audio(media_path: Path, output_dir: Path) -> Path:
    """Normalize audio to EBU R128 and write a WAV to *output_dir*.

    Parameters
    ----------
    media_path:
        Input media file (video or audio).
    output_dir:
        Directory for the output WAV file.  Created if it doesn't exist.

    Returns
    -------
    Path
        The output WAV file path (``output_dir / '<stem>.wav'``).

    Raises
    ------
    subprocess.CalledProcessError
        If ffmpeg exits with a non-zero status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{media_path.stem}.wav"

    # Skip if output already exists and is non-empty
    if output_path.exists() and output_path.stat().st_size > 0:
        logger.info("Skipping %s — normalized output already exists", media_path.name)
        return output_path

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(media_path),
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-y",
            str(output_path),
        ],
        check=True,
    )
    return output_path
