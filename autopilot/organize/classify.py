"""Activity labeling using Claude Sonnet for cluster classification.

Provides label_activities() for generating human-readable labels and
descriptions for activity clusters using LLM analysis.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.config import LLMConfig
    from autopilot.db import CatalogDB

__all__ = ["ClassifyError", "label_activities"]

logger = logging.getLogger(__name__)


class ClassifyError(Exception):
    """Raised for all activity classification failures."""


def _assemble_cluster_summary(
    cluster: dict[str, object],
    db: CatalogDB,
) -> dict[str, str]:
    """Assemble a summary dict for an activity cluster.

    Queries the DB for transcripts, YOLO detections, audio events, and GPS
    information for all clips in the cluster.

    Args:
        cluster: Activity cluster row from the DB.
        db: Catalog database instance.

    Returns:
        Dict with keys: time_range, transcripts, detections, audio_events, gps.
    """
    clip_ids = json.loads(str(cluster["clip_ids_json"]))
    time_start = str(cluster.get("time_start", ""))
    time_end = str(cluster.get("time_end", ""))

    summary: dict[str, str] = {
        "time_range": f"{time_start} to {time_end}",
    }

    # Gather transcripts
    transcript_texts: list[str] = []
    for cid in clip_ids:
        tr = db.get_transcript(cid)
        if tr and tr.get("segments_json"):
            segments = json.loads(str(tr["segments_json"]))
            for seg in segments:
                if isinstance(seg, dict) and "text" in seg:
                    transcript_texts.append(str(seg["text"]))
    summary["transcripts"] = " ".join(transcript_texts) if transcript_texts else ""

    # Gather YOLO detection classes ranked by frequency
    detection_counter: Counter[str] = Counter()
    for cid in clip_ids:
        dets = db.get_detections_for_media(cid)
        for det_row in dets:
            det_list = json.loads(str(det_row["detections_json"]))
            for det in det_list:
                if isinstance(det, dict) and "class" in det:
                    detection_counter[str(det["class"])] += 1
    top_detections = detection_counter.most_common(10)
    summary["detections"] = (
        ", ".join(f"{cls} ({count})" for cls, count in top_detections)
        if top_detections
        else ""
    )

    # Gather audio events ranked by probability
    audio_counter: Counter[str] = Counter()
    for cid in clip_ids:
        events = db.get_audio_events_for_media(cid)
        for ev_row in events:
            ev_list = json.loads(str(ev_row["events_json"]))
            for ev in ev_list:
                if isinstance(ev, dict) and "class" in ev:
                    audio_counter[str(ev["class"])] += 1
    top_audio = audio_counter.most_common(10)
    summary["audio_events"] = (
        ", ".join(f"{cls} ({count})" for cls, count in top_audio)
        if top_audio
        else ""
    )

    # GPS
    gps_lat = cluster.get("gps_center_lat")
    gps_lon = cluster.get("gps_center_lon")
    if gps_lat is not None and gps_lon is not None:
        summary["gps"] = f"{gps_lat}, {gps_lon}"
    else:
        summary["gps"] = ""

    return summary


def _load_prompt() -> str:
    """Load the activity_label.md prompt template."""
    prompt_path = Path(__file__).parent.parent / "prompts" / "activity_label.md"
    return prompt_path.read_text()


def _call_llm(
    summary: dict[str, str],
    config: LLMConfig,
) -> dict[str, object]:
    """Call Claude to label an activity cluster.

    Args:
        summary: Cluster summary dict from _assemble_cluster_summary.
        config: LLM configuration with model selection.

    Returns:
        Parsed JSON dict with label, description, split_recommended, split_reason.

    Raises:
        ClassifyError: If the API call fails or response is malformed.
    """
    import anthropic  # type: ignore[reportMissingImports]

    prompt_text = _load_prompt()

    # Format summary as user message
    user_content = json.dumps(summary, indent=2)

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=config.utility_model,
            max_tokens=1024,
            system=prompt_text,
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception as e:
        raise ClassifyError(f"LLM API call failed: {e}") from e

    # Extract text from response
    if not response.content:
        raise ClassifyError("Empty response from LLM")
    text = response.content[0].text

    # Parse JSON from response
    try:
        # Try to find JSON in markdown code block first
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
        else:
            json_str = text.strip()
        result = json.loads(json_str)
    except (json.JSONDecodeError, IndexError) as e:
        raise ClassifyError(f"Failed to parse LLM response as JSON: {e}") from e

    # Validate required fields
    if "label" not in result or "description" not in result:
        raise ClassifyError(
            f"LLM response missing required fields: {list(result.keys())}"
        )

    return result


def label_activities(db: CatalogDB, config: LLMConfig) -> None:
    """Label all unlabeled activity clusters using Claude.

    Iterates over activity_clusters in the database, assembles summaries,
    calls the LLM for labeling, and updates the DB with results.

    Args:
        db: Catalog database instance.
        config: LLM configuration with model selection.
    """
    clusters = db.get_activity_clusters()
    if not clusters:
        logger.info("No activity clusters to label")
        return

    labeled = 0
    skipped = 0
    for cluster in clusters:
        # Idempotency: skip already-labeled clusters
        if cluster.get("label"):
            skipped += 1
            continue

        cluster_id = str(cluster["cluster_id"])
        logger.info("Labeling cluster %s", cluster_id)

        summary = _assemble_cluster_summary(cluster, db)
        result = _call_llm(summary, config)

        # Log split recommendation if present
        if result.get("split_recommended"):
            split_reason = result.get("split_reason", "no reason given")
            logger.warning(
                "Split recommended for cluster %s: %s",
                cluster_id,
                split_reason,
            )

        # Update DB
        db.update_activity_cluster(
            cluster_id,
            label=str(result["label"]),
            description=str(result["description"]),
        )
        labeled += 1

    logger.info("Labeled %d clusters (%d skipped)", labeled, skipped)
