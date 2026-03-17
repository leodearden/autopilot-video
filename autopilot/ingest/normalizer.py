"""Audio normalizer — EBU R128 loudness normalization via ffmpeg."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def normalize_audio(
    media_path: Path,
    output_dir: Path,
    *,
    root_dir: Path | None = None,
) -> Path:
    """Normalize audio to EBU R128 and write a WAV to *output_dir*.

    Parameters
    ----------
    media_path:
        Input media file (video or audio).
    output_dir:
        Directory for the output WAV file.  Created if it doesn't exist.
    root_dir:
        Optional root directory of the input tree.  When provided, the
        relative path from *root_dir* to *media_path*'s parent is
        mirrored under *output_dir*, preventing filename collisions for
        files with the same stem in different subdirectories (e.g.
        ``day1/DJI_0001.mp4`` and ``day2/DJI_0001.mp4``).

    Returns
    -------
    Path
        The output WAV file path.

    Raises
    ------
    subprocess.CalledProcessError
        If ffmpeg exits with a non-zero status.
    """
    if root_dir is not None:
        relative_parent = media_path.parent.relative_to(root_dir)
        target_dir = output_dir / relative_parent
    else:
        target_dir = output_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{media_path.stem}.wav"

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
