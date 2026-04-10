"""Render routing — selects fast vs slow path per clip and assembles output.

Routes each EDL clip to either the FFmpeg fast path (static crops) or
the MoviePy slow path (dynamic per-frame crops, PiP overlays), then
concatenates all rendered segments into the final output.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from autopilot.render.ffmpeg_render import RenderError, render_simple
from autopilot.render.moviepy_render import ComplexRenderError, render_complex

if TYPE_CHECKING:
    from autopilot.config import OutputConfig
    from autopilot.db import CatalogDB

__all__ = ["RoutingError", "route_and_render"]

#: Default timeout (seconds) for final video concatenation via FFmpeg.
CONCAT_TIMEOUT_SECONDS: int = 1800

logger = logging.getLogger(__name__)


# Crop modes that use static (constant) crop coordinates — FFmpeg fast path
_FAST_CROP_MODES = {"center", "manual_offset", "stabilize_only"}

# Crop modes that produce dynamic per-frame crops — MoviePy slow path
_SLOW_CROP_MODES = {"auto_subject"}


class RoutingError(Exception):
    """Raised for all render routing and assembly failures."""


def _classify_clip(clip: dict, crop_modes: dict) -> str:
    """Classify a clip as 'fast' (FFmpeg) or 'slow' (MoviePy).

    Args:
        clip: EDL clip entry dict with clip_id and optional overlay field.
        crop_modes: Mapping of clip_id -> crop mode string from EDL.

    Returns:
        'fast' or 'slow'.
    """
    # Complex overlays force slow path
    overlay = clip.get("overlay")
    if overlay in ("pip", "split_screen"):
        return "slow"

    # Check crop mode
    clip_id = clip.get("clip_id")
    if clip_id is None:
        return "fast"
    mode = crop_modes.get(clip_id, "center")  # default to center (fast)

    if mode in _SLOW_CROP_MODES:
        return "slow"

    return "fast"


def route_and_render(
    narrative_id: str,
    db: CatalogDB,
    config: OutputConfig,
    output_dir: Path,
) -> Path:
    """Route EDL clips to appropriate renderers and assemble final output.

    Args:
        narrative_id: ID of the narrative to render.
        db: Catalog database handle for loading EDL, media, and crop data.
        config: Output encoding configuration.
        output_dir: Base directory for rendered output files.

    Returns:
        Path to the final rendered output file.

    Raises:
        RoutingError: If routing or rendering fails.
    """
    # -- Load EDL from database -------------------------------------------------
    edit_plan = db.get_edit_plan(narrative_id)
    if edit_plan is None:
        raise RoutingError(f"No edit plan found for narrative {narrative_id!r}")

    edl_json_str = edit_plan.get("edl_json")
    if not edl_json_str:
        raise RoutingError(f"Edit plan for narrative {narrative_id!r} has no edl_json")

    try:
        edl = json.loads(str(edl_json_str))
    except json.JSONDecodeError as e:
        raise RoutingError(f"Corrupt edl_json for narrative {narrative_id!r}: {e}") from e

    clips = edl.get("clips", [])
    _audio_settings = edl.get("audio_settings", {})
    music_tracks = edl.get("music", [])
    voiceovers = edl.get("voiceovers", [])

    # Convert crop_modes list-of-dicts to lookup dicts
    # (following otio_export.py pattern: cm["clip_id"] -> cm["mode"])
    raw_crop_modes = edl.get("crop_modes", [])
    crop_modes: dict[str, str] = {}
    crop_meta: dict[str, dict[str, Any]] = {}
    if isinstance(raw_crop_modes, list):
        for cm in raw_crop_modes:
            cid = cm.get("clip_id", "")
            crop_modes[cid] = cm.get("mode", "")
            crop_meta[cid] = cm
    elif isinstance(raw_crop_modes, dict):
        # Legacy format: already a flat dict of clip_id -> mode
        crop_modes = raw_crop_modes

    # Get narrative metadata for output directory
    narrative = db.get_narrative(narrative_id)
    narrative_title = "untitled"
    if narrative:
        narrative_title = str(narrative.get("title", "untitled"))

    # -- Render each clip ------------------------------------------------------
    segments: list[Path] = []
    resolved_clips: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="render_") as work_dir_str:
        work_dir = Path(work_dir_str)

        for i, clip in enumerate(clips):
            # Guard-first: validate clip_id presence before any assignment.
            # clip_id is str|None — None means the clip has no ID in the EDL.
            clip_id: str | None = clip.get("clip_id")

            if clip_id is None:
                if "source_path" not in clip:
                    raise RoutingError(
                        f"Clip at index {i} has no clip_id and no source_path"
                    )
                # Clip with source_path but no clip_id is allowed — skip
                # crop/transcript lookups; log for visibility.
                logger.warning(
                    "Clip at index %d has no clip_id; crop and transcript "
                    "lookups will be skipped",
                    i,
                )

            # Resolve source_path from DB when not already present in clip
            if "source_path" not in clip:
                # clip_id is guaranteed non-None here — the guard above raises
                # RoutingError when clip_id is None and source_path is absent.
                # Explicit raise (not assert) so python -O cannot strip it.
                if clip_id is None:
                    raise RoutingError(
                        f"Internal error: clip {i} reached DB lookup without clip_id"
                    )
                media = db.get_media(clip_id)
                if media is None:
                    raise RoutingError(
                        f"No media record for clip {clip_id!r} — cannot resolve source_path"
                    )
                file_path = media.get("file_path")
                if not file_path:
                    raise RoutingError(
                        f"Media record for clip {clip_id!r} is missing file_path"
                    )
                clip = {**clip, "source_path": str(file_path)}

            resolved_clips.append(clip)
            classification = _classify_clip(clip, crop_modes)

            if classification == "slow" and clip_id is None:
                raise RoutingError(
                    f"Clip at index {i} has overlay requiring "
                    f"slow path but no clip_id for crop lookup"
                )

            segment_path = work_dir / f"segment_{i:04d}.mp4"

            # Load crop_path from DB when applicable (only when clip_id exists)
            crop_path = None
            target_aspect = getattr(config, "primary_aspect", "16:9")

            if clip_id is not None:
                media_id = clip_id  # clip_id in EDL == media_id in DB
                cm_entry = crop_meta.get(clip_id, {})

                if classification == "slow":
                    # Slow-path clips require crop data
                    subject_track_id = cm_entry.get("subject_track_id", 0)
                    crop_record = db.get_crop_path(
                        media_id,
                        target_aspect,
                        subject_track_id,
                    )
                    if crop_record is None:
                        raise RoutingError(f"No crop data for slow-path clip {clip_id}")
                    path_data = crop_record.get("path_data")
                    if not path_data:
                        msg = f"crop_record has empty/null path_data for clip {clip_id}"
                        raise RoutingError(msg)
                    crop_path = np.array(path_data, dtype=np.float64)
                elif crop_modes.get(clip_id):
                    # Fast-path clips may have optional static crop
                    subject_track_id = cm_entry.get("subject_track_id", 0)
                    crop_record = db.get_crop_path(
                        media_id,
                        target_aspect,
                        subject_track_id,
                    )
                    if crop_record is not None:
                        path_data = crop_record.get("path_data")
                        if path_data is not None:
                            crop_path = np.array(path_data, dtype=np.float64)

            try:
                if classification == "fast":
                    render_simple(clip, crop_path, segment_path, config)
                else:
                    render_complex(clip, crop_path, segment_path, config)
                segments.append(segment_path)
            except RenderError as e:
                raise RoutingError(f"Fast-path render failed for clip {clip_id}: {e}") from e
            except ComplexRenderError as e:
                raise RoutingError(f"Complex render failed for clip {clip_id}: {e}") from e

        if not segments:
            raise RoutingError("No clips to render")

        # -- Concatenate segments ----------------------------------------------
        output_dir = output_dir / narrative_title
        output_dir.mkdir(parents=True, exist_ok=True)
        final_output = output_dir / "final.mp4"

        # Build concat file list
        concat_list = work_dir / "concat.txt"
        with open(concat_list, "w") as f:
            for seg in segments:
                f.write(f"file '{seg}'\n")

        # Build final ffmpeg command for concatenation + audio mixing
        cmd: list[str] = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list)]

        # Add music and voiceover inputs
        audio_inputs: list[str] = []
        input_idx = 1  # 0 is the concat input
        for music in music_tracks:
            music_path = music.get("path", "")
            if music_path:
                cmd.extend(["-i", music_path])
                audio_inputs.append(f"[{input_idx}:a]")
                input_idx += 1

        for vo in voiceovers:
            vo_path = vo.get("path", "")
            if vo_path:
                cmd.extend(["-i", vo_path])
                audio_inputs.append(f"[{input_idx}:a]")
                input_idx += 1

        # -- Collect subtitle segments from per-clip transcripts ----------------
        all_subtitle_segs: list[dict] = []
        cumulative_offset = 0.0
        for clip in resolved_clips:
            media_id: str | None = clip.get("clip_id")
            transcript = db.get_transcript(media_id) if media_id is not None else None
            if transcript and transcript.get("segments_json"):
                try:
                    clip_segs = json.loads(str(transcript["segments_json"]))
                    for seg in clip_segs:
                        all_subtitle_segs.append(
                            {
                                "start": seg.get("start", 0.0) + cumulative_offset,
                                "end": seg.get("end", 0.0) + cumulative_offset,
                                "text": seg.get("text", ""),
                            }
                        )
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Failed to parse transcript for clip %s", media_id)
            in_tc = clip.get("in_timecode", "00:00:00.000")
            out_tc = clip.get("out_timecode", in_tc)
            from autopilot.plan.validator import timecode_to_seconds

            in_sec = timecode_to_seconds(in_tc)
            out_sec = timecode_to_seconds(out_tc)
            cumulative_offset += out_sec - in_sec

        # Generate SRT file if we have subtitle segments
        srt_path: Path | None = None
        if all_subtitle_segs:
            try:
                srt_path = work_dir / "subtitles.srt"
                _generate_srt(all_subtitle_segs, srt_path)
            except OSError:
                logger.warning("Failed to write subtitle file, skipping")
                srt_path = None

        # -- Build filter/codec section ----------------------------------------
        has_audio_mix = bool(audio_inputs)
        has_video_filter = srt_path is not None

        if has_audio_mix and has_video_filter:
            # Both audio mixing and video filter — single filter_complex
            all_audio = ["[0:a]"] + audio_inputs
            fc = (
                f"[0:v]subtitles={srt_path}[vout];"
                + "".join(all_audio)
                + f"amix=inputs={len(all_audio)}:duration=longest[aout]"
            )
            cmd.extend(["-filter_complex", fc])
            cmd.extend(["-map", "[vout]", "-map", "[aout]"])
        elif has_audio_mix:
            # Audio mixing only
            all_audio = ["[0:a]"] + audio_inputs
            mix_filter = "".join(all_audio) + f"amix=inputs={len(all_audio)}:duration=longest[aout]"
            cmd.extend(["-filter_complex", mix_filter])
            cmd.extend(["-map", "0:v", "-map", "[aout]"])
        elif has_video_filter:
            # Video filter only (subtitles), no audio mixing
            cmd.extend(["-vf", f"subtitles={srt_path}"])
        else:
            # Pure concat — no filters at all, stream copy
            cmd.extend(["-c", "copy"])

        cmd.append(str(final_output))

        try:
            subprocess.run(cmd, check=True, timeout=CONCAT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as e:
            raise RoutingError(
                f"Final concatenation timed out after {CONCAT_TIMEOUT_SECONDS}s"
            ) from e
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as e:
            raise RoutingError(f"Final concatenation failed: {e}") from e

    return final_output


def _generate_srt(segments: list[dict], srt_path: Path) -> None:
    """Generate an SRT subtitle file from ASR transcript segments.

    Args:
        segments: List of dicts with 'start', 'end', 'text' keys.
        srt_path: Path to write the SRT file.
    """
    with open(srt_path, "w") as f:
        for i, seg in enumerate(segments, start=1):
            start = _seconds_to_srt_time(seg.get("start", 0.0))
            end = _seconds_to_srt_time(seg.get("end", 0.0))
            text = seg.get("text", "")
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds.

    Returns:
        SRT timestamp string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
