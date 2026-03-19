"""Narrative organization: storyboard construction and narrative proposals.

Provides build_master_storyboard() for assembling structured text from
activity clusters, and propose_narratives() for LLM-powered narrative
planning with format_for_review() for human checkpoint display.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.config import AutopilotConfig
    from autopilot.db import CatalogDB

__all__ = [
    "Narrative",
    "NarrativeError",
    "build_master_storyboard",
    "format_for_review",
    "propose_narratives",
]

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "narrative_planner.md"


class NarrativeError(Exception):
    """Raised for all narrative organization failures."""


@dataclass
class Narrative:
    """A proposed video narrative."""

    narrative_id: str = ""
    title: str = ""
    description: str = ""
    proposed_duration_seconds: float = 0.0
    activity_cluster_ids: list[str] = field(default_factory=list)
    arc: dict[str, str] = field(default_factory=dict)
    emotional_journey: str = ""
    reasoning: str = ""
    status: str = "proposed"


def _build_cluster_summary(
    cluster: dict[str, object],
    db: CatalogDB,
) -> dict[str, str]:
    """Build an enriched summary dict for an activity cluster.

    Queries the DB for all signals associated with the cluster's clips:
    media duration, transcripts, YOLO detections, audio events, and faces.

    Args:
        cluster: Activity cluster row dict from the DB.
        db: Catalog database instance.

    Returns:
        Dict with keys: label, description, duration, key_moments,
        people_present, emotional_tone, visual_quality_notes.
    """
    try:
        clip_ids = json.loads(str(cluster["clip_ids_json"]))
    except json.JSONDecodeError as e:
        raise NarrativeError(f"Corrupt clip_ids_json in cluster: {e}") from e

    summary: dict[str, str] = {
        "label": str(cluster.get("label") or "Unlabeled"),
        "description": str(cluster.get("description") or ""),
    }

    # --- Duration from media files ---
    total_duration = 0.0
    for cid in clip_ids:
        media = db.get_media(cid)
        if media and media.get("duration_seconds"):
            try:
                total_duration += float(media["duration_seconds"])  # type: ignore[arg-type]
            except (ValueError, TypeError):
                pass
    if total_duration > 0:
        minutes = total_duration / 60
        summary["duration"] = f"{total_duration:.0f}s ({minutes:.1f} min)"
    else:
        summary["duration"] = "unknown"

    # --- Key moments: transcript highlights + detection density peaks ---
    transcript_texts: list[str] = []
    for cid in clip_ids:
        tr = db.get_transcript(cid)
        if tr and tr.get("segments_json"):
            try:
                segments = json.loads(str(tr["segments_json"]))
            except json.JSONDecodeError:
                logger.warning("Corrupt segments_json for media %s, skipping", cid)
                continue
            for seg in segments:
                if isinstance(seg, dict) and "text" in seg:
                    transcript_texts.append(str(seg["text"]))

    # Detection class counts (for event density peaks)
    detection_counter: Counter[str] = Counter()
    for cid in clip_ids:
        dets = db.get_detections_for_media(cid)
        for det_row in dets:
            try:
                det_list = json.loads(str(det_row["detections_json"]))
            except json.JSONDecodeError:
                logger.warning("Corrupt detections_json for media %s, skipping", cid)
                continue
            for det in det_list:
                if isinstance(det, dict) and "class" in det:
                    detection_counter[str(det["class"])] += 1

    key_moments_parts: list[str] = []
    if transcript_texts:
        # Include first few transcript lines as highlights
        highlights = transcript_texts[:5]
        key_moments_parts.append("Transcript: " + " | ".join(highlights))
    top_detections = detection_counter.most_common(5)
    if top_detections:
        det_str = ", ".join(f"{cls} ({count})" for cls, count in top_detections)
        key_moments_parts.append(f"Detected objects: {det_str}")
    summary["key_moments"] = "; ".join(key_moments_parts) if key_moments_parts else ""

    # --- People present: face cluster labels ---
    face_cluster_ids: set[int] = set()
    for cid in clip_ids:
        faces = db.get_faces_for_media(cid)
        for face in faces:
            cid_val = face.get("cluster_id")
            if cid_val is not None:
                try:
                    face_cluster_ids.add(int(cid_val))  # type: ignore[arg-type]
                except (ValueError, TypeError):
                    pass

    people_labels: list[str] = []
    if face_cluster_ids:
        all_face_clusters = db.get_face_clusters()
        fc_map = {int(fc["cluster_id"]): fc.get("label") for fc in all_face_clusters}  # type: ignore[arg-type]
        for fcid in sorted(face_cluster_ids):
            label = fc_map.get(fcid)
            if label:
                people_labels.append(str(label))
            else:
                people_labels.append(f"Face #{fcid}")
    summary["people_present"] = ", ".join(people_labels) if people_labels else ""

    # --- Emotional tone from audio events + transcript ---
    audio_counter: Counter[str] = Counter()
    for cid in clip_ids:
        events = db.get_audio_events_for_media(cid)
        for ev_row in events:
            try:
                ev_list = json.loads(str(ev_row["events_json"]))
            except json.JSONDecodeError:
                logger.warning("Corrupt events_json for media %s, skipping", cid)
                continue
            for ev in ev_list:
                if isinstance(ev, dict) and "class" in ev:
                    audio_counter[str(ev["class"])] += 1

    tone_parts: list[str] = []
    top_audio = audio_counter.most_common(5)
    if top_audio:
        tone_parts.append(
            "Audio: " + ", ".join(f"{cls} ({count})" for cls, count in top_audio)
        )
    if transcript_texts:
        tone_parts.append(f"Speech detected ({len(transcript_texts)} segments)")
    summary["emotional_tone"] = "; ".join(tone_parts) if tone_parts else ""

    # --- Visual quality notes from detection confidence ---
    confidences: list[float] = []
    for cid in clip_ids:
        dets = db.get_detections_for_media(cid)
        for det_row in dets:
            try:
                det_list = json.loads(str(det_row["detections_json"]))
            except json.JSONDecodeError:
                continue
            for det in det_list:
                if isinstance(det, dict) and "confidence" in det:
                    try:
                        confidences.append(float(det["confidence"]))
                    except (ValueError, TypeError):
                        pass

    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        min_conf = min(confidences)
        quality_notes: list[str] = [f"Avg detection confidence: {avg_conf:.2f}"]
        if min_conf < 0.5:
            quality_notes.append(
                f"Some low-confidence frames (min: {min_conf:.2f}) - "
                "possible motion blur or poor framing"
            )
        if avg_conf >= 0.8:
            quality_notes.append("Overall good visual quality")
        elif avg_conf >= 0.6:
            quality_notes.append("Moderate visual quality")
        else:
            quality_notes.append("Low visual quality - many unclear frames")
        summary["visual_quality_notes"] = "; ".join(quality_notes)
    else:
        summary["visual_quality_notes"] = ""

    return summary


def _load_and_fill_prompt(config: AutopilotConfig) -> str:
    """Load narrative_planner.md and fill creator profile placeholders.

    Args:
        config: Full autopilot config with creator profile.

    Returns:
        Prompt text with all placeholders filled.

    Raises:
        NarrativeError: If the prompt file cannot be loaded.
    """
    try:
        template = _PROMPT_PATH.read_text()
    except OSError as e:
        raise NarrativeError(f"Failed to load prompt: {e}") from e

    creator = config.creator
    return (
        template
        .replace("{creator_name}", creator.name)
        .replace("{channel_style}", creator.channel_style)
        .replace("{target_audience}", creator.target_audience)
        .replace("{default_video_duration}", creator.default_video_duration_minutes)
        .replace("{narration_style}", creator.narration_style)
        .replace("{music_preference}", creator.music_preference)
    )


def _call_llm(
    storyboard: str,
    config: AutopilotConfig,
) -> str:
    """Call Claude Opus for narrative proposals.

    Args:
        storyboard: Structured text storyboard.
        config: Full autopilot config with LLM and creator settings.

    Returns:
        Raw response text from the LLM.

    Raises:
        NarrativeError: If the API call fails or response is empty.
    """
    import anthropic  # type: ignore[reportMissingImports]

    try:
        system_prompt = _load_and_fill_prompt(config)
    except NarrativeError:
        raise
    except OSError as e:
        raise NarrativeError(f"Failed to load prompt: {e}") from e

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config.llm.planning_model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": storyboard}],
        )
    except Exception as e:
        raise NarrativeError(f"LLM API call failed: {e}") from e

    if not response.content:
        raise NarrativeError("Empty response from LLM")

    return response.content[0].text


_REQUIRED_NARRATIVE_FIELDS = {"title", "activity_cluster_ids", "proposed_duration_seconds"}


def _parse_narratives(text: str) -> list[Narrative]:
    """Parse LLM response text into Narrative objects.

    Args:
        text: Raw LLM response text (possibly wrapped in code blocks).

    Returns:
        List of Narrative objects.

    Raises:
        NarrativeError: If JSON parsing fails or required fields are missing.
    """
    # Extract JSON from code block if present
    try:
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
        else:
            json_str = text.strip()
        data = json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        raise NarrativeError(f"Failed to parse LLM response as JSON: {e}") from e

    if not isinstance(data, list):
        raise NarrativeError(f"Expected JSON array, got {type(data).__name__}")

    narratives: list[Narrative] = []
    for entry in data:
        if not isinstance(entry, dict):
            raise NarrativeError(f"Expected JSON object in array, got {type(entry).__name__}")
        missing = _REQUIRED_NARRATIVE_FIELDS - set(entry.keys())
        if missing:
            raise NarrativeError(
                f"Narrative entry missing required fields: {sorted(missing)}"
            )
        arc = entry.get("arc", {})
        if not isinstance(arc, dict):
            arc = {}
        narratives.append(Narrative(
            narrative_id=str(uuid.uuid4()),
            title=str(entry["title"]),
            description=str(entry.get("reasoning", "")),
            proposed_duration_seconds=float(entry["proposed_duration_seconds"]),
            activity_cluster_ids=list(entry["activity_cluster_ids"]),
            arc=arc,
            emotional_journey=str(entry.get("emotional_journey", "")),
            reasoning=str(entry.get("reasoning", "")),
            status="proposed",
        ))

    return narratives


def build_master_storyboard(db: CatalogDB) -> str:
    """Build a structured text storyboard from all activity clusters.

    Iterates over activity clusters in the database, enriches each with
    signal data (transcripts, detections, audio events, faces), and
    formats everything into structured text suitable for LLM consumption.

    Args:
        db: Catalog database instance.

    Returns:
        Structured text storyboard.
    """
    clusters = db.get_activity_clusters()
    if not clusters:
        logger.info("No activity clusters for storyboard")
        return "# Master Storyboard\n\nNo activity clusters found."

    sections: list[str] = ["# Master Storyboard\n"]

    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id", "unknown"))
        summary = _build_cluster_summary(cluster, db)

        section_lines: list[str] = [
            f"## Cluster: {cluster_id}",
            f"- **Label**: {summary['label']}",
        ]
        if summary["description"]:
            section_lines.append(f"- **Description**: {summary['description']}")
        section_lines.append(f"- **Duration**: {summary['duration']}")
        if summary["key_moments"]:
            section_lines.append(f"- **Key moments**: {summary['key_moments']}")
        if summary["people_present"]:
            section_lines.append(f"- **People present**: {summary['people_present']}")
        if summary["emotional_tone"]:
            section_lines.append(f"- **Emotional tone**: {summary['emotional_tone']}")
        if summary["visual_quality_notes"]:
            section_lines.append(
                f"- **Visual quality**: {summary['visual_quality_notes']}"
            )

        sections.append("\n".join(section_lines))

    return "\n\n".join(sections)


def propose_narratives(
    storyboard: str,
    db: CatalogDB,
    config: AutopilotConfig,
) -> list[Narrative]:
    """Propose video narratives using LLM analysis of the storyboard.

    Sends the master storyboard to Claude Opus for narrative planning,
    parses the response into Narrative objects, and stores them in the DB.

    Args:
        storyboard: Structured text storyboard from build_master_storyboard.
        db: Catalog database instance for storing proposals.
        config: Full autopilot config with creator profile and LLM settings.

    Returns:
        List of proposed Narrative objects.

    Raises:
        NarrativeError: If LLM call fails or response is malformed.
    """
    try:
        response_text = _call_llm(storyboard, config)
    except NarrativeError:
        raise
    except Exception as e:
        raise NarrativeError(f"Narrative proposal failed: {e}") from e

    narratives = _parse_narratives(response_text)

    # Store each narrative in the DB
    for narrative in narratives:
        db.insert_narrative(
            narrative.narrative_id,
            title=narrative.title,
            description=narrative.description,
            proposed_duration_seconds=narrative.proposed_duration_seconds,
            activity_cluster_ids_json=json.dumps(narrative.activity_cluster_ids),
            arc_notes=json.dumps(narrative.arc),
            status=narrative.status,
        )

    logger.info("Proposed %d narratives", len(narratives))
    return narratives


def format_for_review(narratives: list[Narrative]) -> str:
    """Format proposed narratives for human review.

    Produces a human-readable summary of each narrative with numbered
    entries showing title, duration, included clusters, arc, emotional
    journey, and reasoning.

    Args:
        narratives: List of Narrative objects to format.

    Returns:
        Formatted review text.
    """
    raise NotImplementedError
