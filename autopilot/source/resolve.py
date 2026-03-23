"""EDL asset resolution orchestrator.

Processes an EDL, attempts to resolve each asset request (music, voiceover,
B-roll), updates the EDL with resolved file paths, collects unresolved
requests into a fetch list, and persists the updated EDL.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autopilot.source import BrollRequest, MusicRequest, VoiceoverRequest
from autopilot.source.broll import source_broll
from autopilot.source.fetch_list import generate_fetch_list
from autopilot.source.music import source_music
from autopilot.source.voiceover import generate_voiceover

if TYPE_CHECKING:
    from autopilot.config import ModelConfig
    from autopilot.db import CatalogDB
    from autopilot.source import AssetRequest

__all__ = [
    "resolve_edl_assets",
]

logger = logging.getLogger(__name__)


def resolve_edl_assets(
    edl: dict[str, Any],
    config: ModelConfig,
    output_dir: Path,
    db: CatalogDB,
    *,
    narrative_id: str | None = None,
) -> dict[str, Any]:
    """Resolve all asset requests in an EDL.

    Extracts music, voiceover, and B-roll requests from the EDL, attempts
    to source each one, and updates the EDL entries with resolved file paths.
    Unresolved requests are collected into a fetch list.

    Args:
        edl: EDL dict with music, voiceovers, broll_requests arrays.
        config: ModelConfig with tts_engine and music_engine settings.
        output_dir: Directory for generated/downloaded asset files.
        db: CatalogDB instance for persisting the updated EDL.
        narrative_id: Optional narrative ID for DB persistence.

    Returns:
        Dict with:
            - 'edl': Updated EDL dict with resolved_path fields.
            - 'unresolved': List of AssetRequest objects that couldn't be resolved.
            - 'fetch_list_path': Path to the fetch list file (if unresolved exist).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    unresolved: list[AssetRequest] = []

    # --- Resolve music ---
    music_dir = output_dir / "music"
    music_dir.mkdir(parents=True, exist_ok=True)
    for entry in edl.get("music", []):
        request = MusicRequest(
            mood=entry.get("mood", ""),
            duration=float(entry.get("duration", 0)),
            start_time=entry.get("start_time", "00:00:00.000"),
        )
        try:
            path = source_music(request, config, music_dir)
        except Exception as e:
            logger.warning("Music sourcing failed for %r: %s", request.mood, e)
            path = None

        if path is not None:
            entry["resolved_path"] = str(path)
            request.resolved_path = path
            logger.info("Resolved music: %s → %s", request.mood, path)
        else:
            entry["resolved_path"] = None
            unresolved.append(request)
            logger.info("Unresolved music: %s", request.mood)

    # --- Resolve voiceovers ---
    vo_dir = output_dir / "voiceovers"
    vo_dir.mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(edl.get("voiceovers", [])):
        vo_request = VoiceoverRequest(
            text=entry.get("text", ""),
            start_time=entry.get("start_time", "00:00:00.000"),
            duration=float(entry.get("duration", 0)),
        )
        vo_output = vo_dir / f"voiceover_{i:03d}.wav"
        try:
            path = generate_voiceover(vo_request.text, vo_output, config)
        except Exception as e:
            logger.warning("Voiceover generation failed: %s", e)
            path = None

        if path is not None:
            entry["resolved_path"] = str(path)
            vo_request.resolved_path = path
            logger.info("Resolved voiceover %d → %s", i, path)
        else:
            entry["resolved_path"] = None
            unresolved.append(vo_request)
            logger.info("Unresolved voiceover %d", i)

    # --- Resolve B-roll ---
    broll_dir = output_dir / "broll"
    broll_dir.mkdir(parents=True, exist_ok=True)
    for entry in edl.get("broll_requests", []):
        broll_request = BrollRequest(
            description=entry.get("description", ""),
            duration=float(entry.get("duration", 0)),
            start_time=entry.get("start_time", "00:00:00.000"),
        )
        try:
            paths = source_broll(broll_request, broll_dir)
        except Exception as e:
            logger.warning("B-roll sourcing failed for %r: %s", broll_request.description, e)
            paths = None

        if paths:
            # Use the first downloaded file as the resolved path
            entry["resolved_path"] = str(paths[0])
            broll_request.resolved_path = paths[0]
            logger.info(
                "Resolved B-roll: %s → %s (%d options)",
                broll_request.description,
                paths[0],
                len(paths),
            )
        else:
            entry["resolved_path"] = None
            unresolved.append(broll_request)
            logger.info("Unresolved B-roll: %s", broll_request.description)

    # --- Generate fetch list for unresolved ---
    fetch_list_path = None
    if unresolved:
        try:
            fetch_list_path = output_dir / "fetch_list.md"
            generate_fetch_list(unresolved, fetch_list_path)
            logger.info(
                "Generated fetch list with %d unresolved items at %s",
                len(unresolved),
                fetch_list_path,
            )
        except Exception as e:
            logger.warning("Failed to generate fetch list: %s", e)
            fetch_list_path = None

    # --- Persist updated EDL ---
    if narrative_id is not None:
        try:
            with db:
                db.upsert_edit_plan(narrative_id, json.dumps(edl))
                db.update_narrative_status(narrative_id, "sourced")
            logger.info("Updated edit plan for narrative %s", narrative_id)
        except Exception as e:
            logger.warning("Failed to persist updated EDL: %s", e)

    result: dict[str, Any] = {
        "edl": edl,
        "unresolved": unresolved,
    }
    if fetch_list_path:
        result["fetch_list_path"] = fetch_list_path

    return result
