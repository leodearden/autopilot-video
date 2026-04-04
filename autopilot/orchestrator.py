"""Pipeline orchestrator — DAG-based stage execution for autopilot-video."""

from __future__ import annotations

import enum
import functools
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineOrchestrator",
    "StageDefinition",
    "StageResult",
    "StageStatus",
    "request_shutdown",
    "shutdown_requested",
]

# Module-level shutdown flag — thread-safe via threading.Event.
_shutdown_event = threading.Event()


def shutdown_requested() -> bool:
    """Return True if a graceful shutdown has been requested."""
    return _shutdown_event.is_set()


def request_shutdown() -> None:
    """Request a graceful shutdown of the pipeline."""
    _shutdown_event.set()
    logger.info("Shutdown requested")


def _reset_shutdown() -> None:
    """Reset the shutdown flag (for testing only)."""
    _shutdown_event.clear()


class StageStatus(enum.Enum):
    """Execution status of a pipeline stage."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class StageDefinition:
    """Definition of a single pipeline stage.

    Attributes:
        name: Unique stage identifier (e.g. 'INGEST', 'ANALYZE').
        func: Callable that implements the stage, accepting (config, db) kwargs.
        dependencies: Names of stages that must complete before this one.
        estimated_seconds: Estimated wall-clock time for this stage.
    """

    name: str
    func: Callable[..., Any]
    dependencies: list[str] = field(default_factory=list)
    estimated_seconds: float = 0


@dataclass
class StageResult:
    """Result of executing a single pipeline stage.

    Attributes:
        status: Final status after execution.
        elapsed_seconds: Wall-clock time spent on this stage.
        error_message: Error description if status is ERROR, else None.
    """

    status: StageStatus
    elapsed_seconds: float = 0.0
    error_message: str | None = None


def _stage_stub(name: str) -> Callable[..., Any]:
    """Create a stub function for an unimplemented stage."""

    def _stub(**kwargs: Any) -> None:
        logger.info("Stage %s not yet implemented", name)

    _stub.__name__ = f"_stage_{name.lower()}"
    _stub.__qualname__ = f"_stage_{name.lower()}"
    return _stub


def _start_job(
    db: Any,
    stage: str,
    job_type: str,
    *,
    target_id: str | None = None,
    target_label: str | None = None,
    worker: str | None = None,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> tuple[str, float]:
    """Create a pipeline_jobs record with status='running' and return (job_id, start_mono).

    All db calls are wrapped in try/except for resilience — job tracking must
    never break the pipeline.
    """
    job_id = uuid4().hex
    start_mono = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    try:
        db.insert_job(
            job_id,
            stage,
            job_type,
            target_id=target_id,
            target_label=target_label,
            status="running",
            started_at=started_at,
            worker=worker,
            run_id=run_id,
        )
    except Exception as exc:
        logger.warning("Failed to insert job %s: %s", job_id, exc)
    if emit_fn is not None:
        emit_fn("job_started", stage=stage, job_id=job_id)
    return job_id, start_mono


def _finish_job(
    db: Any,
    job_id: str,
    start_mono: float,
    *,
    status: str = "done",
    error_message: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
    stage: str | None = None,
) -> None:
    """Update a pipeline_jobs record with final status and duration.

    All db calls are wrapped in try/except for resilience.
    """
    duration = time.monotonic() - start_mono
    finished_at = datetime.now(timezone.utc).isoformat()
    try:
        db.update_job(
            job_id,
            status=status,
            finished_at=finished_at,
            duration_seconds=duration,
            error_message=error_message,
        )
    except Exception as exc:
        logger.warning("Failed to update job %s: %s", job_id, exc)
    if emit_fn is not None:
        event_type = "job_completed" if status == "done" else "job_error"
        emit_fn(event_type, stage=stage, job_id=job_id)


@functools.wraps(_start_job)  # type: ignore[arg-type]
def _track_job(
    db: Any,
    stage: str,
    job_type: str,
    **kwargs: Any,
):
    """Context manager that wraps _start_job/_finish_job for clean job tracking.

    Usage::

        with _track_job(db, 'INGEST', 'ingest_file', run_id=run_id) as job_id:
            # do work
    """
    from contextlib import contextmanager as _cm

    @_cm
    def _inner():
        job_id, start_mono = _start_job(db, stage, job_type, **kwargs)
        try:
            yield job_id
        except Exception as exc:
            _finish_job(
                db,
                job_id,
                start_mono,
                status="error",
                error_message=str(exc),
                emit_fn=kwargs.get("emit_fn"),
                stage=stage,
            )
            raise
        else:
            _finish_job(
                db,
                job_id,
                start_mono,
                status="done",
                emit_fn=kwargs.get("emit_fn"),
                stage=stage,
            )

    return _inner()


def _run_ingest(
    *,
    config: Any,
    db: Any,
    force: bool = False,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> None:
    """INGEST stage: scan directory, insert media, normalize audio, mark duplicates."""
    from autopilot.ingest import dedup, normalizer, scanner

    files = scanner.scan_directory(config.input_dir, max_workers=None)
    norm_dir = config.output_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint/resume: skip media already in the DB
    skipped = 0
    if not force:
        to_process = []
        for mf in files:
            media_id = mf.sha256_prefix or mf.file_path.stem
            if db.get_media(media_id) is not None:
                skipped += 1
            else:
                to_process.append(mf)
        if skipped > 0:
            logger.info(
                "Resuming INGEST: %d/%d files already ingested",
                skipped,
                len(files),
            )
    else:
        to_process = list(files)

    ingested = 0
    for mf in to_process:
        if shutdown_requested():
            break
        media_id = mf.sha256_prefix or mf.file_path.stem
        try:
            if run_id is not None:
                with _track_job(
                    db,
                    "INGEST",
                    "ingest_file",
                    target_id=media_id,
                    target_label=str(mf.file_path.name),
                    worker="cpu",
                    run_id=run_id,
                    emit_fn=emit_fn,
                ):
                    db.insert_media(
                        media_id,
                        str(mf.file_path),
                        sha256_prefix=mf.sha256_prefix,
                        codec=mf.codec,
                        resolution_w=mf.resolution_w,
                        resolution_h=mf.resolution_h,
                        fps=mf.fps,
                        duration_seconds=mf.duration_seconds,
                        created_at=mf.created_at,
                        gps_lat=mf.gps_lat,
                        gps_lon=mf.gps_lon,
                        audio_channels=mf.audio_channels,
                        metadata_json=mf.metadata_json,
                    )
                    normalizer.normalize_audio(mf.file_path, norm_dir, root_dir=config.input_dir)
            else:
                db.insert_media(
                    media_id,
                    str(mf.file_path),
                    sha256_prefix=mf.sha256_prefix,
                    codec=mf.codec,
                    resolution_w=mf.resolution_w,
                    resolution_h=mf.resolution_h,
                    fps=mf.fps,
                    duration_seconds=mf.duration_seconds,
                    created_at=mf.created_at,
                    gps_lat=mf.gps_lat,
                    gps_lon=mf.gps_lon,
                    audio_channels=mf.audio_channels,
                    metadata_json=mf.metadata_json,
                )
                normalizer.normalize_audio(mf.file_path, norm_dir, root_dir=config.input_dir)
            ingested += 1
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", mf.file_path, exc)
            continue

    if run_id is not None:
        with _track_job(
            db,
            "INGEST",
            "dedup",
            worker="cpu",
            run_id=run_id,
            emit_fn=emit_fn,
        ):
            dedup.mark_duplicates(db)
    else:
        dedup.mark_duplicates(db)
    logger.info("Ingest complete: %d/%d files ingested", ingested, len(files))


def _run_analyze(
    *,
    config: Any,
    db: Any,
    force: bool = False,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> None:
    """ANALYZE stage: run all analysis passes on each media file."""
    from autopilot.analyze import asr, audio_events, embeddings, faces, objects, scenes
    from autopilot.analyze.gpu_scheduler import GPUScheduler

    scheduler = GPUScheduler(
        total_vram=0,
        device=config.processing.gpu_device,
    )

    all_media = db.list_all_media()
    media_list = [m for m in all_media if m.get("status") != "duplicate"]

    # Per-pass checkpoint mapping: (has_method, label)
    pass_checks: list[tuple[str, str]] = [
        ("has_transcript", "transcribed"),
        ("has_boundaries", "shot-detected"),
        ("has_detections", "object-detected"),
        ("has_faces", "face-detected"),
        ("has_embeddings", "embedded"),
        ("has_audio_events", "audio-classified"),
    ]

    # Log resume counts per pass
    if not force:
        for has_method, label in pass_checks:
            checker = getattr(db, has_method)
            done_count = sum(1 for m in media_list if checker(m["id"]))
            if done_count > 0:
                logger.info(
                    "Resuming ANALYZE: %d/%d media already %s",
                    done_count,
                    len(media_list),
                    label,
                )

    # Helper to optionally wrap a call in _track_job
    _jkw: dict[str, Any] = {"run_id": run_id, "emit_fn": emit_fn}

    successes = 0
    try:
        for media in media_list:
            if shutdown_requested():
                break
            media_id = media["id"]
            file_path = Path(media["file_path"])
            audio_path = file_path  # audio extracted from same file
            try:
                if force or not db.has_transcript(media_id):
                    if run_id is not None:
                        with _track_job(
                            db,
                            "ANALYZE",
                            "asr",
                            target_id=media_id,
                            worker="gpu",
                            **_jkw,
                        ):
                            asr.transcribe_media(
                                media_id,
                                audio_path,
                                db,
                                scheduler,
                                config.models,
                                batch_size=config.processing.batch_size_whisper,
                            )
                    else:
                        asr.transcribe_media(
                            media_id,
                            audio_path,
                            db,
                            scheduler,
                            config.models,
                            batch_size=config.processing.batch_size_whisper,
                        )
                if force or not db.has_boundaries(media_id):
                    if run_id is not None:
                        with _track_job(
                            db,
                            "ANALYZE",
                            "scenes",
                            target_id=media_id,
                            worker="gpu",
                            **_jkw,
                        ):
                            scenes.detect_shots(
                                media_id,
                                file_path,
                                db,
                                scheduler,
                            )
                    else:
                        scenes.detect_shots(media_id, file_path, db, scheduler)
                if force or not db.has_detections(media_id):
                    if run_id is not None:
                        with _track_job(
                            db,
                            "ANALYZE",
                            "objects",
                            target_id=media_id,
                            worker="gpu",
                            **_jkw,
                        ):
                            objects.detect_objects(
                                media_id,
                                file_path,
                                db,
                                scheduler,
                                config.models,
                                sparse=False,
                            )
                    else:
                        objects.detect_objects(
                            media_id,
                            file_path,
                            db,
                            scheduler,
                            config.models,
                            sparse=False,
                        )
                if force or not db.has_faces(media_id):
                    if run_id is not None:
                        with _track_job(
                            db,
                            "ANALYZE",
                            "faces",
                            target_id=media_id,
                            worker="gpu",
                            **_jkw,
                        ):
                            faces.detect_faces(
                                media_id,
                                file_path,
                                db,
                                scheduler,
                                config.models,
                            )
                    else:
                        faces.detect_faces(
                            media_id,
                            file_path,
                            db,
                            scheduler,
                            config.models,
                        )
                if force or not db.has_embeddings(media_id):
                    if run_id is not None:
                        with _track_job(
                            db,
                            "ANALYZE",
                            "embeddings",
                            target_id=media_id,
                            worker="gpu",
                            **_jkw,
                        ):
                            embeddings.compute_embeddings(
                                media_id,
                                file_path,
                                db,
                                scheduler,
                                config.models,
                            )
                    else:
                        embeddings.compute_embeddings(
                            media_id,
                            file_path,
                            db,
                            scheduler,
                            config.models,
                        )
                if force or not db.has_audio_events(media_id):
                    if run_id is not None:
                        with _track_job(
                            db,
                            "ANALYZE",
                            "audio_events",
                            target_id=media_id,
                            worker="gpu",
                            **_jkw,
                        ):
                            audio_events.classify_audio_events(
                                media_id,
                                audio_path,
                                db,
                                scheduler,
                            )
                    else:
                        audio_events.classify_audio_events(
                            media_id,
                            audio_path,
                            db,
                            scheduler,
                        )
                successes += 1
            except Exception:
                logger.exception("Analysis failed for media %s", media_id)

        if run_id is not None:
            with _track_job(
                db,
                "ANALYZE",
                "face_clustering",
                worker="cpu",
                **_jkw,
            ):
                faces.cluster_faces(db, eps=0.5, min_samples=3)
        else:
            faces.cluster_faces(db, eps=0.5, min_samples=3)
        logger.info(
            "Analyze complete: %d/%d media processed",
            successes,
            len(media_list),
        )
    finally:
        scheduler.force_unload_all()


def _run_classify(
    *,
    config: Any,
    db: Any,
    force: bool = False,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> None:
    """CLASSIFY stage: cluster and label activities."""
    from autopilot.organize import classify, cluster

    if not force:
        existing = db.get_activity_clusters()
        if existing and all(c.get("label") for c in existing):
            logger.info(
                "Resuming CLASSIFY: %d labeled clusters already exist, skipping",
                len(existing),
            )
            return

    _jkw: dict[str, Any] = {"run_id": run_id, "emit_fn": emit_fn}
    if run_id is not None:
        with _track_job(db, "CLASSIFY", "cluster_activities", worker="cpu", **_jkw):
            cluster.cluster_activities(db)
        with _track_job(db, "CLASSIFY", "label_activities", worker="cpu", **_jkw):
            classify.label_activities(db, config.llm)
    else:
        cluster.cluster_activities(db)
        classify.label_activities(db, config.llm)
    logger.info("Classify complete")


def _run_narrate(
    *,
    config: Any,
    db: Any,
    force: bool = False,
    human_review_fn: Callable | None = None,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> None:
    """NARRATE stage: build storyboard, propose narratives, human review."""
    from autopilot.organize import narratives

    if not force:
        existing = db.list_narratives("approved")
        if existing:
            logger.info(
                "Resuming NARRATE: %d approved narratives already exist, skipping",
                len(existing),
            )
            return

    _jkw: dict[str, Any] = {"run_id": run_id, "emit_fn": emit_fn}
    if run_id is not None:
        with _track_job(db, "NARRATE", "build_storyboard", worker="cpu", **_jkw):
            storyboard = narratives.build_master_storyboard(db)
        with _track_job(db, "NARRATE", "propose_narratives", worker="cpu", **_jkw):
            proposed = narratives.propose_narratives(storyboard, db, config)
    else:
        storyboard = narratives.build_master_storyboard(db)
        proposed = narratives.propose_narratives(storyboard, db, config)
    formatted = narratives.format_for_review(proposed)

    if human_review_fn is not None:
        approved_ids = set(human_review_fn(formatted, proposed))
    else:
        # Auto-approve all when no callback provided
        approved_ids = {n.narrative_id for n in proposed}

    for narr in proposed:
        if narr.narrative_id in approved_ids:
            db.update_narrative_status(narr.narrative_id, "approved")
        else:
            db.update_narrative_status(narr.narrative_id, "rejected")

    logger.info(
        "Narrate complete: %d proposed, %d approved",
        len(proposed),
        len(approved_ids),
    )


def _run_script(
    *,
    config: Any,
    db: Any,
    force: bool = False,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> None:
    """SCRIPT stage: generate scripts for each approved narrative."""
    from autopilot.plan import script

    approved = db.list_narratives("approved")

    # Checkpoint/resume: skip narratives that already have a script
    skipped = 0
    if not force:
        to_process = []
        for narr in approved:
            nid = narr["narrative_id"]
            if db.get_narrative_script(nid) is not None:
                skipped += 1
            else:
                to_process.append(narr)
        if skipped > 0:
            logger.info(
                "Resuming SCRIPT: %d/%d narratives already scripted",
                skipped,
                len(approved),
            )
    else:
        to_process = list(approved)

    _jkw: dict[str, Any] = {"run_id": run_id, "emit_fn": emit_fn}
    successes = 0
    for narr in to_process:
        if shutdown_requested():
            break
        nid = narr["narrative_id"]
        try:
            if run_id is not None:
                with _track_job(
                    db,
                    "SCRIPT",
                    "generate_script",
                    target_id=nid,
                    worker="cpu",
                    **_jkw,
                ):
                    script.generate_script(nid, db, config.llm)
            else:
                script.generate_script(nid, db, config.llm)
            successes += 1
        except Exception:
            logger.exception("Script generation failed for narrative %s", nid)

    if approved and successes == 0 and skipped == 0:
        raise RuntimeError("All narratives failed script generation")

    logger.info("Script complete: %d/%d succeeded", successes, len(approved))


def _run_edl(
    *,
    config: Any,
    db: Any,
    force: bool = False,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> None:
    """EDL stage: generate EDL, validate, and export OTIO per narrative."""
    from autopilot.plan import edl as edl_mod
    from autopilot.plan import otio_export, validator

    approved = db.list_narratives("approved")

    # Checkpoint/resume: skip narratives that already have an edit plan with edl_json
    skipped = 0
    if not force:
        to_process = []
        for narr in approved:
            nid = narr["narrative_id"]
            existing = db.get_edit_plan(nid)
            if existing is not None and existing.get("edl_json"):
                skipped += 1
            else:
                to_process.append(narr)
        if skipped > 0:
            logger.info(
                "Resuming EDL: %d/%d narratives already have edit plans",
                skipped,
                len(approved),
            )
    else:
        to_process = list(approved)

    _jkw: dict[str, Any] = {"run_id": run_id, "emit_fn": emit_fn}
    successes = 0
    for narr in to_process:
        if shutdown_requested():
            break
        nid = narr["narrative_id"]
        # Skip narratives without scripts
        if db.get_narrative_script(nid) is None:
            logger.warning("Skipping EDL for %s: no script", nid)
            continue
        try:
            if run_id is not None:
                with _track_job(
                    db,
                    "EDL",
                    "generate_edl",
                    target_id=nid,
                    worker="cpu",
                    **_jkw,
                ):
                    edl = edl_mod.generate_edl(nid, db, config.llm)
                with _track_job(
                    db,
                    "EDL",
                    "validate_edl",
                    target_id=nid,
                    worker="cpu",
                    **_jkw,
                ):
                    val_result = validator.validate_edl(edl, db)
                otio_path = config.output_dir / nid / "timeline.otio"
                otio_path.parent.mkdir(parents=True, exist_ok=True)
                with _track_job(
                    db,
                    "EDL",
                    "otio_export",
                    target_id=nid,
                    worker="cpu",
                    **_jkw,
                ):
                    otio_export.export_otio(edl, otio_path, db)
            else:
                edl = edl_mod.generate_edl(nid, db, config.llm)
                val_result = validator.validate_edl(edl, db)
                otio_path = config.output_dir / nid / "timeline.otio"
                otio_path.parent.mkdir(parents=True, exist_ok=True)
                otio_export.export_otio(edl, otio_path, db)

            db.upsert_edit_plan(nid, otio_path=str(otio_path))
            successes += 1
        except Exception:
            logger.exception("EDL generation failed for narrative %s", nid)

    if approved and successes == 0 and skipped == 0:
        raise RuntimeError("All narratives failed EDL generation")

    logger.info("EDL complete: %d/%d succeeded", successes, len(approved))


def _run_source_assets(
    *,
    config: Any,
    db: Any,
    force: bool = False,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> None:
    """SOURCE_ASSETS stage: resolve assets for each narrative with an edit plan."""
    from autopilot.source import resolve

    approved = db.list_narratives("approved")

    # Checkpoint/resume: skip narratives where asset directory exists and is non-empty
    skipped = 0
    if not force:
        to_process = []
        for narr in approved:
            nid = narr["narrative_id"]
            asset_dir = config.output_dir / "assets" / nid
            if asset_dir.exists() and any(asset_dir.iterdir()):
                skipped += 1
            else:
                to_process.append(narr)
        if skipped > 0:
            logger.info(
                "Resuming SOURCE_ASSETS: %d/%d narratives already have assets",
                skipped,
                len(approved),
            )
    else:
        to_process = list(approved)

    _jkw: dict[str, Any] = {"run_id": run_id, "emit_fn": emit_fn}
    successes = 0
    for narr in to_process:
        if shutdown_requested():
            break
        nid = narr["narrative_id"]
        plan = db.get_edit_plan(nid)
        if plan is None:
            logger.warning("Skipping source for %s: no edit plan", nid)
            continue
        try:
            if run_id is not None:
                with _track_job(
                    db,
                    "SOURCE_ASSETS",
                    "resolve_assets",
                    target_id=nid,
                    worker="cpu",
                    **_jkw,
                ):
                    edl = json.loads(plan["edl_json"])
                    asset_dir = config.output_dir / "assets" / nid
                    asset_dir.mkdir(parents=True, exist_ok=True)
                    resolve.resolve_edl_assets(
                        edl,
                        config.models,
                        asset_dir,
                        db,
                        narrative_id=nid,
                    )
            else:
                edl = json.loads(plan["edl_json"])
                asset_dir = config.output_dir / "assets" / nid
                asset_dir.mkdir(parents=True, exist_ok=True)
                resolve.resolve_edl_assets(
                    edl,
                    config.models,
                    asset_dir,
                    db,
                    narrative_id=nid,
                )
            successes += 1
        except Exception:
            logger.exception("Source resolution failed for narrative %s", nid)

    if approved and successes == 0 and skipped == 0:
        raise RuntimeError("All narratives failed source resolution")

    logger.info("Source complete: %d/%d succeeded", successes, len(approved))


def _run_render(
    *,
    config: Any,
    db: Any,
    force: bool = False,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> None:
    """RENDER stage: route, render, and validate per narrative."""
    from autopilot.render import router
    from autopilot.render import validate as render_validate

    approved = db.list_narratives("approved")

    # Checkpoint/resume: skip narratives with existing render output on disk
    skipped = 0
    if not force:
        to_process = []
        for narr in approved:
            nid = narr["narrative_id"]
            plan = db.get_edit_plan(nid)
            render_path = plan.get("render_path") if plan else None
            if render_path and Path(render_path).exists():
                skipped += 1
            else:
                to_process.append(narr)
        if skipped > 0:
            logger.info(
                "Resuming RENDER: %d/%d narratives already rendered",
                skipped,
                len(approved),
            )
    else:
        to_process = list(approved)

    _jkw: dict[str, Any] = {"run_id": run_id, "emit_fn": emit_fn}
    successes = 0
    for narr in to_process:
        if shutdown_requested():
            break
        nid = narr["narrative_id"]
        plan = db.get_edit_plan(nid)
        if plan is None:
            logger.warning("Skipping render for %s: no edit plan", nid)
            continue
        try:
            if run_id is not None:
                with _track_job(
                    db,
                    "RENDER",
                    "render",
                    target_id=nid,
                    worker="gpu",
                    **_jkw,
                ):
                    output_path = router.route_and_render(
                        nid,
                        db,
                        config.output,
                        config.output_dir,
                    )
                edl = json.loads(plan["edl_json"])
                with _track_job(
                    db,
                    "RENDER",
                    "validate_render",
                    target_id=nid,
                    worker="cpu",
                    **_jkw,
                ):
                    report = render_validate.validate_render(
                        output_path,
                        edl,
                        config.output,
                    )
            else:
                output_path = router.route_and_render(
                    nid,
                    db,
                    config.output,
                    config.output_dir,
                )
                edl = json.loads(plan["edl_json"])
                report = render_validate.validate_render(
                    output_path,
                    edl,
                    config.output,
                )
            for issue in report.issues:
                logger.warning(
                    "Render validation [%s] %s: %s",
                    nid,
                    issue.severity,
                    issue.message,
                )
            db.upsert_edit_plan(nid, render_path=str(output_path))
            successes += 1
        except Exception:
            logger.exception("Render failed for narrative %s", nid)

    if approved and successes == 0 and skipped == 0:
        raise RuntimeError("All narratives failed render")

    logger.info("Render complete: %d/%d succeeded", successes, len(approved))


def _run_upload(
    *,
    config: Any,
    db: Any,
    force: bool = False,
    run_id: str | None = None,
    emit_fn: Callable[..., Any] | None = None,
) -> None:
    """UPLOAD stage: upload video and extract thumbnail per narrative."""
    from autopilot.upload import thumbnail, youtube

    approved = db.list_narratives("approved")

    # Checkpoint/resume: skip narratives that already have uploads
    skipped = 0
    if not force:
        to_process = []
        for narr in approved:
            nid = narr["narrative_id"]
            if db.get_upload(nid) is not None:
                skipped += 1
            else:
                to_process.append(narr)
        if skipped > 0:
            logger.info(
                "Resuming UPLOAD: %d/%d narratives already uploaded",
                skipped,
                len(approved),
            )
    else:
        to_process = list(approved)

    _jkw: dict[str, Any] = {"run_id": run_id, "emit_fn": emit_fn}
    successes = 0
    for narr in to_process:
        if shutdown_requested():
            break
        nid = narr["narrative_id"]
        plan = db.get_edit_plan(nid)
        if plan is None:
            logger.warning("Skipping upload for %s: no edit plan", nid)
            continue
        render_path = plan.get("render_path")
        if not render_path:
            logger.warning("Skipping upload for %s: no render_path in edit plan", nid)
            continue
        video_path = Path(render_path)
        try:
            if run_id is not None:
                with _track_job(
                    db,
                    "UPLOAD",
                    "upload_video",
                    target_id=nid,
                    worker="cpu",
                    **_jkw,
                ):
                    youtube.upload_video(nid, video_path, db, config.youtube)
                with _track_job(
                    db,
                    "UPLOAD",
                    "extract_thumbnail",
                    target_id=nid,
                    worker="cpu",
                    **_jkw,
                ):
                    thumbnail.extract_best_thumbnail(nid, video_path, db)
            else:
                youtube.upload_video(nid, video_path, db, config.youtube)
                thumbnail.extract_best_thumbnail(nid, video_path, db)
            successes += 1
        except Exception:
            logger.exception("Upload failed for narrative %s", nid)

    if approved and successes == 0 and skipped == 0:
        raise RuntimeError("All narratives failed upload")

    logger.info("Upload complete: %d/%d succeeded", successes, len(approved))


class PipelineOrchestrator:
    """Manages a DAG of pipeline stages and executes them in topological order."""

    def __init__(
        self,
        budget_seconds: float | None = None,
        human_review_fn: Callable | None = None,
        force: bool = False,
    ) -> None:
        self.budget_seconds = budget_seconds
        self.human_review_fn = human_review_fn
        self.force = force

        # Wrap _run_narrate with human_review_fn via partial
        narrate_func = functools.partial(
            _run_narrate,
            human_review_fn=human_review_fn,
        )

        self.stages: list[StageDefinition] = [
            StageDefinition(
                name="INGEST",
                func=_run_ingest,
                dependencies=[],
                estimated_seconds=600,
            ),
            StageDefinition(
                name="ANALYZE",
                func=_run_analyze,
                dependencies=["INGEST"],
                estimated_seconds=1800,
            ),
            StageDefinition(
                name="CLASSIFY",
                func=_run_classify,
                dependencies=["ANALYZE"],
                estimated_seconds=900,
            ),
            StageDefinition(
                name="NARRATE",
                func=narrate_func,
                dependencies=["CLASSIFY"],
                estimated_seconds=300,
            ),
            StageDefinition(
                name="SCRIPT",
                func=_run_script,
                dependencies=["NARRATE"],
                estimated_seconds=300,
            ),
            StageDefinition(
                name="EDL",
                func=_run_edl,
                dependencies=["SCRIPT"],
                estimated_seconds=120,
            ),
            StageDefinition(
                name="SOURCE_ASSETS",
                func=_run_source_assets,
                dependencies=["EDL"],
                estimated_seconds=1200,
            ),
            StageDefinition(
                name="RENDER",
                func=_run_render,
                dependencies=["EDL", "SOURCE_ASSETS"],
                estimated_seconds=3600,
            ),
            StageDefinition(
                name="UPLOAD",
                func=_run_upload,
                dependencies=["RENDER"],
                estimated_seconds=600,
            ),
        ]
        self._stage_map: dict[str, StageDefinition] = {s.name: s for s in self.stages}

    # Mapping from orchestrator stage names to DB gate stage names.
    _STAGE_TO_GATE: dict[str, str] = {
        "SOURCE_ASSETS": "source",
    }

    @staticmethod
    def _gate_stage_name(stage_name: str) -> str:
        """Map an orchestrator stage name to the corresponding DB gate name."""
        return PipelineOrchestrator._STAGE_TO_GATE.get(stage_name, stage_name.lower())

    def _check_gate(self, stage_name: str) -> str:
        """Check the gate for a stage and return the gate decision.

        Returns:
            'approved' if the stage should run, 'skipped' if it should be skipped.
        """
        try:
            return self._check_gate_inner(stage_name)
        except Exception as exc:
            logger.warning("Gate check failed for %s: %s — auto-approving", stage_name, exc)
            return "approved"

    def _check_gate_inner(self, stage_name: str) -> str:
        """Inner gate check logic (called by _check_gate with resilience wrapper)."""
        gate_name = self._gate_stage_name(stage_name)
        gate = self._db.get_gate(gate_name)

        if gate is None:
            logger.warning(
                "Gate not found for stage %s (gate=%s), auto-approving",
                stage_name,
                gate_name,
            )
            return "approved"

        mode = gate.get("mode", "auto")

        if mode in ("auto", "notify"):
            decided_at = datetime.now(timezone.utc).isoformat()
            self._db.update_gate(
                gate_name,
                status="approved",
                decided_by="system",
                decided_at=decided_at,
            )
            self._emit_event("gate_passed", stage=stage_name)
            return "approved"

        if mode == "pause":
            # Set gate to waiting and emit event
            self._db.update_gate(gate_name, status="waiting")
            self._emit_event("gate_waiting", stage=stage_name)
            logger.info("[GATE-WAITING] %s — waiting for external approval", stage_name)

            timeout_hours = gate.get("timeout_hours")
            wait_start = time.monotonic()

            # Poll for approval/skip
            while not shutdown_requested():
                time.sleep(2)

                # Check timeout
                if timeout_hours is not None and timeout_hours > 0:
                    elapsed_hours = (time.monotonic() - wait_start) / 3600
                    if elapsed_hours > timeout_hours:
                        self._db.update_gate(
                            gate_name,
                            status="approved",
                            decided_by="system",
                            decided_at=datetime.now(timezone.utc).isoformat(),
                            notes="auto-approved: timeout",
                        )
                        self._emit_event(
                            "gate_approved",
                            stage=stage_name,
                            payload={"reason": "timeout"},
                        )
                        logger.info("[GATE-TIMEOUT] %s — auto-approved after timeout", stage_name)
                        return "approved"

                current = self._db.get_gate(gate_name)
                if current is None:
                    return "approved"
                status = current.get("status", "waiting")
                if status == "approved":
                    self._emit_event("gate_approved", stage=stage_name)
                    return "approved"
                if status == "skipped":
                    self._emit_event("gate_skipped", stage=stage_name)
                    return "skipped"

            # Shutdown requested
            return "skipped"

        return "approved"

    def _emit_event(
        self,
        event_type: str,
        stage: str | None = None,
        job_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Emit a pipeline event to the database.

        Args:
            event_type: Event type string (e.g. 'stage_started').
            stage: Stage name associated with the event.
            job_id: Optional job identifier.
            payload: Optional dict to serialize as JSON payload.
        """
        try:
            payload_json = json.dumps(payload) if payload is not None else None
            self._db.insert_event(
                event_type=event_type,
                stage=stage,
                job_id=job_id,
                payload_json=payload_json,
            )
            logger.debug("Event: %s stage=%s", event_type, stage)
        except Exception as exc:
            logger.warning("Failed to emit event %s: %s", event_type, exc)

    def execution_order(self) -> list[str]:
        """Return stage names in topologically sorted execution order.

        Uses Kahn's algorithm with sorted queue for determinism.
        """
        # Build in-degree map and adjacency list
        in_degree: dict[str, int] = {s.name: 0 for s in self.stages}
        dependents: dict[str, list[str]] = {s.name: [] for s in self.stages}

        for stage in self.stages:
            in_degree[stage.name] = len(stage.dependencies)
            for dep in stage.dependencies:
                dependents[dep].append(stage.name)

        # Start with nodes that have no dependencies
        queue = sorted([name for name, deg in in_degree.items() if deg == 0])
        result: list[str] = []

        while queue:
            current = queue.pop(0)
            result.append(current)
            for dependent in sorted(dependents[current]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    queue.sort()

        return result

    def run(
        self,
        config: Any,
        db: Any,
        dry_run: bool = False,
    ) -> dict[str, StageResult]:
        """Execute pipeline stages in topological order.

        Args:
            config: AutopilotConfig instance.
            db: CatalogDB instance.
            dry_run: If True, log execution plan without calling stage functions.

        Returns:
            Dict mapping stage names to their StageResult.
        """
        order = self.execution_order()
        results: dict[str, StageResult] = {}
        errored_stages: set[str] = set()
        pipeline_start = time.monotonic()

        # --- Run tracking: create pipeline_runs record ---
        self._db = db
        try:
            run_id = uuid4().hex
            started_at = datetime.now(timezone.utc).isoformat()
            self._run_id: str | None = run_id
            db.insert_run(
                run_id,
                started_at=started_at,
                config_snapshot=str(config),
                status="running",
                budget_remaining_seconds=self.budget_seconds,
            )
        except Exception as exc:
            logger.warning("Run tracking unavailable: %s", exc)
            self._run_id = None

        # --- Gate initialization ---
        try:
            db.init_default_gates()
            for gate in db.get_all_gates():
                db.update_gate(gate["stage"], status="idle")
        except Exception as exc:
            logger.warning("Gate initialization failed: %s", exc)

        for stage_name in order:
            if shutdown_requested():
                for remaining in order:
                    if remaining not in results:
                        results[remaining] = StageResult(
                            status=StageStatus.SKIPPED,
                            elapsed_seconds=0.0,
                        )
                logger.info("Shutdown requested — skipping remaining stages")
                break

            stage = self._stage_map[stage_name]

            # Check if any dependency errored — skip if so
            skip = False
            for dep in stage.dependencies:
                if dep in errored_stages:
                    skip = True
                    break

            if skip:
                results[stage_name] = StageResult(
                    status=StageStatus.SKIPPED,
                    elapsed_seconds=0.0,
                )
                errored_stages.add(stage_name)
                logger.info("[SKIPPED] %s (dependency failed)", stage_name)
                continue

            if dry_run:
                logger.info(
                    "[DRY-RUN] %s (est. %ss)",
                    stage_name,
                    stage.estimated_seconds,
                )
                results[stage_name] = StageResult(
                    status=StageStatus.SKIPPED,
                    elapsed_seconds=0.0,
                )
                continue

            # Check gate — may block (pause mode) or skip
            gate_result = self._check_gate(stage_name)
            if gate_result == "skipped":
                results[stage_name] = StageResult(
                    status=StageStatus.SKIPPED,
                    elapsed_seconds=0.0,
                )
                logger.info("[GATE-SKIPPED] %s", stage_name)
                continue

            logger.info("[RUNNING] %s", stage_name)
            # Emit stage_started event and update current_stage on run
            if self._run_id is not None:
                self._emit_event("stage_started", stage=stage_name)
                try:
                    db.update_run(self._run_id, current_stage=stage_name)
                except Exception as exc:
                    logger.warning("Run tracking update failed: %s", exc)
            t0 = time.monotonic()
            try:
                stage.func(
                    config=config,
                    db=db,
                    force=self.force,
                    run_id=self._run_id,
                    emit_fn=self._emit_event,
                )
                elapsed = time.monotonic() - t0
                results[stage_name] = StageResult(
                    status=StageStatus.DONE,
                    elapsed_seconds=elapsed,
                )
                if self._run_id is not None:
                    self._emit_event(
                        "stage_completed",
                        stage=stage_name,
                        payload={"elapsed": elapsed},
                    )
                logger.info("[DONE] %s (%.1fs)", stage_name, elapsed)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                results[stage_name] = StageResult(
                    status=StageStatus.ERROR,
                    elapsed_seconds=elapsed,
                    error_message=str(exc),
                )
                errored_stages.add(stage_name)
                if self._run_id is not None:
                    self._emit_event(
                        "stage_error",
                        stage=stage_name,
                        payload={"error": str(exc)},
                    )
                logger.error("[ERROR] %s: %s", stage_name, exc)

            # Progress reporting after each stage
            cumulative = time.monotonic() - pipeline_start
            if self.budget_seconds and self.budget_seconds > 0:
                budget_remaining = self.budget_seconds - cumulative
                if self._run_id is not None:
                    try:
                        db.update_run(
                            self._run_id,
                            budget_remaining_seconds=budget_remaining,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Run tracking update failed: %s",
                            exc,
                        )
                pct = (cumulative / self.budget_seconds) * 100
                logger.info(
                    "[PROGRESS] %.1fs / %.1fs budget (%.1f%%)",
                    cumulative,
                    self.budget_seconds,
                    pct,
                )
            else:
                logger.info(
                    "[PROGRESS] %.1fs elapsed (no budget set)",
                    cumulative,
                )

        total_elapsed = time.monotonic() - pipeline_start

        # Summary table
        summary_parts = []
        for sn in order:
            if sn in results:
                r = results[sn]
                summary_parts.append(f"  {sn}: {r.status.value} ({r.elapsed_seconds:.1f}s)")
        summary = "\n".join(summary_parts)
        logger.info(
            "Pipeline complete (%.1fs)\n%s",
            total_elapsed,
            summary,
        )

        # --- Run tracking: finalize run record ---
        if self._run_id is not None:
            final_status = "failed" if errored_stages else "completed"
            try:
                db.update_run(
                    self._run_id,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    status=final_status,
                    wall_clock_seconds=total_elapsed,
                )
            except Exception as exc:
                logger.warning("Run tracking update failed: %s", exc)
            if errored_stages:
                self._emit_event(
                    "run_failed",
                    payload={"duration": total_elapsed},
                )
            else:
                self._emit_event(
                    "run_completed",
                    payload={"duration": total_elapsed},
                )

        # Check budget
        if self.budget_seconds is not None and total_elapsed > self.budget_seconds:
            logger.warning(
                "Pipeline exceeded budget: %.1fs elapsed vs %.1fs budget",
                total_elapsed,
                self.budget_seconds,
            )

        return results
