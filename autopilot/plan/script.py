"""Script generation: narrative storyboard assembly and scene-by-scene scripting.

Provides build_narrative_storyboard() for assembling structured text from
a narrative's activity clusters with per-shot data, and generate_script()
for LLM-powered scene-by-scene script creation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from autopilot.llm import LlmError, invoke_claude

if TYPE_CHECKING:
    from autopilot.config import LLMConfig
    from autopilot.db import CatalogDB

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "script_writer.md"

__all__ = [
    "ScriptError",
    "build_narrative_storyboard",
    "generate_script",
]


class ScriptError(Exception):
    """Raised for all script generation failures."""


def build_narrative_storyboard(narrative_id: str, db: CatalogDB) -> str:
    """Build a structured text storyboard for a narrative's activity clusters.

    Assembles per-shot data from the narrative's activity clusters including
    transcripts, visual descriptions, YOLO detections, faces, and audio events.

    Args:
        narrative_id: ID of the narrative to build the storyboard for.
        db: Catalog database instance.

    Returns:
        Structured text storyboard suitable for LLM consumption.

    Raises:
        ScriptError: If the narrative is not found.
    """
    narrative = db.get_narrative(narrative_id)
    if narrative is None:
        raise ScriptError(f"Narrative not found: {narrative_id}")

    # Parse activity cluster IDs
    cluster_ids_raw = narrative.get("activity_cluster_ids_json")
    if not cluster_ids_raw:
        cluster_ids: list[str] = []
    else:
        try:
            cluster_ids = json.loads(str(cluster_ids_raw))
        except json.JSONDecodeError as e:
            raise ScriptError(f"Corrupt activity_cluster_ids_json in narrative: {e}") from e

    title = str(narrative.get("title") or "Untitled")

    if not cluster_ids:
        logger.info("Narrative %s has no activity clusters", narrative_id)
        return f"# L-Storyboard: {title}\n\nNo activity clusters assigned to this narrative."

    # Build cluster lookup from all clusters
    all_clusters = db.get_activity_clusters()
    cluster_map = {str(c["cluster_id"]): c for c in all_clusters}

    # Build face cluster label lookup
    all_face_clusters = db.get_face_clusters()
    fc_label_map: dict[int, str] = {}
    for fc in all_face_clusters:
        try:
            cid = fc["cluster_id"]
            fc_label_map[int(cid)] = str(fc.get("label") or f"Face #{cid}")  # type: ignore[call-overload]
        except (ValueError, TypeError):
            pass

    sections: list[str] = [f"# L-Storyboard: {title}\n"]

    for cid in cluster_ids:
        cluster = cluster_map.get(cid)
        if cluster is None:
            logger.warning("Cluster %s not found in DB, skipping", cid)
            continue
        section = _build_cluster_section(cluster, db, fc_label_map)
        sections.append(section)

    return "\n\n".join(sections)


def _build_cluster_section(
    cluster: dict[str, object],
    db: CatalogDB,
    fc_label_map: dict[int, str],
) -> str:
    """Build storyboard section for a single activity cluster.

    For each clip in the cluster, segments by shot boundaries and gathers
    transcript, visual descriptions, detections, faces, and audio events.
    """
    cluster_id = str(cluster.get("cluster_id", "unknown"))
    label = str(cluster.get("label") or "Unlabeled")
    description = str(cluster.get("description") or "")

    try:
        clip_ids = json.loads(str(cluster["clip_ids_json"]))
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Corrupt clip_ids_json in cluster %s: %s", cluster_id, e)
        return f"## Cluster: {cluster_id} — {label}\n\nSkipped: corrupt clip data."

    lines: list[str] = [f"## Cluster: {cluster_id} — {label}"]
    if description:
        lines.append(f"Description: {description}")
    lines.append("")

    shot_counter = 0
    for clip_id in clip_ids:
        media = db.get_media(clip_id)
        has_dur = media and media.get("duration_seconds")
        clip_duration = float(media["duration_seconds"]) if has_dur else 0.0  # type: ignore[index,arg-type]
        has_fps = media and media.get("fps")
        clip_fps = float(media["fps"]) if has_fps else 30.0  # type: ignore[index,arg-type]

        shots = _get_shots_for_clip(clip_id, clip_duration, db)

        # Gather signals for the whole clip once
        transcript = db.get_transcript(clip_id)
        transcript_segments = _parse_transcript_segments(transcript, clip_id)
        all_detections = db.get_detections_for_media(clip_id)
        all_faces = db.get_faces_for_media(clip_id, include_embedding=False)
        all_audio_events = db.get_audio_events_for_media(clip_id)
        captions = db.get_captions_for_media(clip_id)

        for shot in shots:
            shot_counter += 1
            shot_start = shot["start_time"]
            shot_end = shot["end_time"]
            shot_duration = shot_end - shot_start

            lines.append(f"### Shot {shot_counter} (Clip: {clip_id})")
            lines.append(f"- Duration: {shot_duration:.1f}s")
            lines.append(f"- Source: {clip_id} [{shot_start:.1f}s – {shot_end:.1f}s]")

            # Transcript segments in this shot's time range
            shot_texts = [
                seg["text"]
                for seg in transcript_segments
                if seg["start"] < shot_end and seg["end"] > shot_start
            ]
            if shot_texts:
                lines.append(f"- Transcript: {' '.join(shot_texts)}")

            # Visual descriptions (captions) overlapping this shot
            shot_captions = [
                str(c["caption"])
                for c in captions
                if c["start_time"] is not None
                and c["end_time"] is not None
                and float(c["start_time"]) < shot_end  # type: ignore[arg-type]
                and float(c["end_time"]) > shot_start  # type: ignore[arg-type]
            ]
            if shot_captions:
                lines.append(f"- Visual: {'; '.join(shot_captions)}")

            # Detections in frame range
            start_frame = int(shot_start * clip_fps)
            end_frame = int(shot_end * clip_fps)
            shot_dets = _get_detections_in_range(
                all_detections,
                start_frame,
                end_frame,
            )
            if shot_dets:
                lines.append(f"- Objects: {', '.join(shot_dets)}")

            # Faces in frame range
            shot_people = _get_faces_in_range(
                all_faces,
                start_frame,
                end_frame,
                fc_label_map,
            )
            if shot_people:
                lines.append(f"- People: {', '.join(shot_people)}")

            # Audio events in time range
            shot_audio = _get_audio_in_range(
                all_audio_events,
                shot_start,
                shot_end,
            )
            if shot_audio:
                lines.append(f"- Audio: {', '.join(shot_audio)}")

            lines.append("")

    return "\n".join(lines)


def _get_shots_for_clip(
    clip_id: str,
    clip_duration: float,
    db: CatalogDB,
) -> list[dict[str, float]]:
    """Get shot boundaries for a clip, falling back to single shot."""
    boundaries_rows = db.get_boundaries(clip_id)
    if isinstance(boundaries_rows, list) and boundaries_rows:
        # Use the first method's boundaries
        row = boundaries_rows[0]
        try:
            shots_data = json.loads(str(row["boundaries_json"]))
            if isinstance(shots_data, list) and shots_data:
                return [
                    {
                        "start_time": float(s.get("start_time", 0)),
                        "end_time": float(s.get("end_time", clip_duration)),
                    }
                    for s in shots_data
                    if isinstance(s, dict)
                ]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Corrupt boundaries_json for %s: %s", clip_id, e)
    elif isinstance(boundaries_rows, dict) and boundaries_rows:
        try:
            shots_data = json.loads(str(boundaries_rows["boundaries_json"]))
            if isinstance(shots_data, list) and shots_data:
                return [
                    {
                        "start_time": float(s.get("start_time", 0)),
                        "end_time": float(s.get("end_time", clip_duration)),
                    }
                    for s in shots_data
                    if isinstance(s, dict)
                ]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning("Corrupt boundaries_json for %s: %s", clip_id, e)

    # Fallback: treat entire clip as one shot
    return [{"start_time": 0.0, "end_time": clip_duration or 0.0}]


def _parse_transcript_segments(
    transcript: dict[str, object] | None,
    clip_id: str,
) -> list[dict]:
    """Parse transcript segments JSON, returning empty list on failure."""
    if not transcript or not transcript.get("segments_json"):
        return []
    try:
        segments = json.loads(str(transcript["segments_json"]))
        return [
            s
            for s in segments
            if isinstance(s, dict) and "text" in s and "start" in s and "end" in s
        ]
    except json.JSONDecodeError:
        logger.warning("Corrupt segments_json for media %s, skipping", clip_id)
        return []


def _get_detections_in_range(
    all_detections: list[dict[str, object]],
    start_frame: int,
    end_frame: int,
) -> list[str]:
    """Extract unique detection class names in a frame range."""
    classes: list[str] = []
    seen: set[str] = set()
    for det_row in all_detections:
        try:
            frame = int(det_row["frame_number"])  # type: ignore[call-overload]
        except (TypeError, ValueError):
            logger.warning(
                "Skipping detection with invalid frame_number: %s",
                det_row.get("frame_number"),
            )
            continue
        if frame < start_frame or frame > end_frame:
            continue
        try:
            det_list = json.loads(str(det_row["detections_json"]))
        except json.JSONDecodeError:
            continue
        for det in det_list:
            if isinstance(det, dict) and "class" in det:
                cls = str(det["class"])
                if cls not in seen:
                    seen.add(cls)
                    classes.append(cls)
    return classes


def _get_faces_in_range(
    all_faces: list[dict[str, object]],
    start_frame: int,
    end_frame: int,
    fc_label_map: dict[int, str],
) -> list[str]:
    """Extract unique face labels in a frame range."""
    labels: list[str] = []
    seen: set[str] = set()
    for face in all_faces:
        try:
            frame = int(face["frame_number"])  # type: ignore[call-overload]
        except (TypeError, ValueError):
            logger.warning("Skipping face with invalid frame_number: %s", face.get("frame_number"))
            continue
        if frame < start_frame or frame > end_frame:
            continue
        cid = face.get("cluster_id")
        if cid is not None:
            try:
                label = fc_label_map.get(int(cid), f"Face #{cid}")  # type: ignore[call-overload]
            except (ValueError, TypeError):
                label = f"Face #{cid}"
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def _get_audio_in_range(
    all_audio_events: list[dict[str, object]],
    start_time: float,
    end_time: float,
) -> list[str]:
    """Extract unique audio event class names in a time range."""
    classes: list[str] = []
    seen: set[str] = set()
    for ev_row in all_audio_events:
        try:
            ts = float(ev_row["timestamp_seconds"])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            logger.warning(
                "Skipping audio event with invalid timestamp_seconds: %s",
                ev_row.get("timestamp_seconds"),
            )
            continue
        if ts < start_time or ts > end_time:
            continue
        try:
            ev_list = json.loads(str(ev_row["events_json"]))
        except json.JSONDecodeError:
            continue
        for ev in ev_list:
            if isinstance(ev, dict) and "class" in ev:
                cls = str(ev["class"])
                if cls not in seen:
                    seen.add(cls)
                    classes.append(cls)
    return classes


def generate_script(narrative_id: str, db: CatalogDB, config: LLMConfig) -> dict:
    """Generate a scene-by-scene script for a narrative using Claude Opus.

    Builds the narrative storyboard, sends it with the narrative description
    to the LLM, parses the JSON response, stores it in the DB, and updates
    the narrative status to 'scripted'.

    Args:
        narrative_id: ID of the narrative to script.
        db: Catalog database instance.
        config: LLM configuration with planning_model.

    Returns:
        Parsed script dict with scenes, broll_needs, and quality_flags.

    Raises:
        ScriptError: If narrative not found, LLM call fails, or response is malformed.
    """
    # Load narrative
    narrative = db.get_narrative(narrative_id)
    if narrative is None:
        raise ScriptError(f"Narrative not found: {narrative_id}")

    # Build storyboard
    storyboard = build_narrative_storyboard(narrative_id, db)

    # Load prompt
    try:
        system_prompt = _PROMPT_PATH.read_text()
    except OSError as e:
        raise ScriptError(f"Failed to load script_writer prompt: {e}") from e

    # Build user message with narrative description + storyboard
    description = str(narrative.get("description") or "")
    title = str(narrative.get("title") or "")
    arc_notes = str(narrative.get("arc_notes") or "")
    emotional_journey = str(narrative.get("emotional_journey") or "")
    duration = narrative.get("proposed_duration_seconds", 0)

    user_message = (
        f"## Approved Narrative\n\n"
        f"**Title**: {title}\n"
        f"**Description**: {description}\n"
        f"**Proposed duration**: {duration} seconds\n"
        f"**Arc**: {arc_notes}\n"
        f"**Emotional journey**: {emotional_journey}\n\n"
        f"## Full L-Storyboard\n\n{storyboard}"
    )

    # Call LLM
    response_text = _call_llm(user_message, system_prompt, config)

    # Parse response
    script_data = _parse_script_response(response_text)

    # Store in DB
    with db:
        db.upsert_narrative_script(narrative_id, json.dumps(script_data))
        db.update_narrative_status(narrative_id, "scripted")

    return script_data


def _call_llm(user_message: str, system_prompt: str, config: LLMConfig) -> str:
    """Call Claude Opus for script generation.

    Args:
        user_message: User message with storyboard and narrative info.
        system_prompt: System prompt from script_writer.md.
        config: LLM config with planning_model.

    Returns:
        Raw response text from the LLM.

    Raises:
        ScriptError: If the API call fails or response is empty.
    """
    try:
        text = invoke_claude(
            prompt=user_message,
            system=system_prompt,
            model=config.planning_model,
            max_tokens=8192,
        )
    except LlmError as e:
        raise ScriptError(f"LLM API call failed: {e}") from e

    assert isinstance(text, str)  # type guard: simple call returns str
    return text


def _parse_script_response(text: str) -> dict:
    """Parse LLM response text into a script dict.

    Args:
        text: Raw LLM response (possibly wrapped in code blocks).

    Returns:
        Dict with scenes, broll_needs, quality_flags.

    Raises:
        ScriptError: If JSON parsing fails or structure is invalid.
    """
    try:
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
        else:
            json_str = text.strip()
        data = json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        raise ScriptError(f"Failed to parse script response as JSON: {e}") from e

    if not isinstance(data, dict):
        raise ScriptError(f"Expected JSON object, got {type(data).__name__}")

    if "scenes" not in data:
        raise ScriptError("Script response missing 'scenes' field")

    return data
