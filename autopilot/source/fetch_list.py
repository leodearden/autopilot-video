"""Generate a human-readable fetch list for unresolved asset requests."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from autopilot.source import AssetRequest, BrollRequest, MusicRequest, VoiceoverRequest

__all__ = [
    "generate_fetch_list",
]

logger = logging.getLogger(__name__)

_SUGGESTED_SOURCES = {
    "Music": "Freesound, MusicGen, AudioJungle",
    "B-Roll": "Pexels, Pixabay, Artgrid",
    "Voiceover": "Kokoro TTS, ElevenLabs, manual recording",
}


def generate_fetch_list(
    unresolved: Sequence[AssetRequest], output_path: Path
) -> None:
    """Write a human-readable markdown table listing unresolved asset requests.

    Args:
        unresolved: List of AssetRequest objects that could not be resolved
            automatically. Each may be a MusicRequest, BrollRequest, or
            VoiceoverRequest.
        output_path: Path where the markdown file will be written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Unresolved Asset Fetch List",
        "",
        "| Type | Description | Start Time | Duration | Suggested Sources | Priority |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for req in unresolved:
        row = _format_row(req)
        if row:
            lines.append(row)

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(
        "Fetch list written to %s (%d unresolved items)", output_path, len(unresolved)
    )


def _format_row(req: AssetRequest) -> str | None:
    """Format an AssetRequest as a markdown table row.

    Args:
        req: An asset request (MusicRequest, BrollRequest, or VoiceoverRequest).

    Returns:
        Markdown table row string, or None if type is unrecognized.
    """
    if isinstance(req, MusicRequest):
        req_type = "Music"
        description = req.mood
        duration = req.duration
        start_time = req.start_time
        priority = "High" if duration > 30 else "Medium"
    elif isinstance(req, BrollRequest):
        req_type = "B-Roll"
        description = req.description
        duration = req.duration
        start_time = req.start_time
        priority = "High" if duration > 3 else "Medium"
    elif isinstance(req, VoiceoverRequest):
        req_type = "Voiceover"
        description = req.text
        duration = req.duration
        start_time = req.start_time
        priority = "High"  # Voiceovers are always high priority
    else:
        logger.warning("Unknown request type: %s", type(req).__name__)
        return None

    sources = _SUGGESTED_SOURCES.get(req_type, "N/A")

    return (
        f"| {req_type} | {description} | {start_time}"
        f" | {duration:.1f}s | {sources} | {priority} |"
    )
