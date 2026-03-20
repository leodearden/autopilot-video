"""Pipeline orchestrator — DAG-based stage execution for autopilot-video."""

from __future__ import annotations

import enum
import functools
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.analyze import asr, audio_events, embeddings, faces, objects, scenes
from autopilot.analyze.gpu_scheduler import GPUScheduler
from autopilot.ingest import dedup, normalizer, scanner
from autopilot.organize import classify, cluster, narratives
from autopilot.plan import edl as edl_mod
from autopilot.plan import otio_export, script, validator
from autopilot.render import router
from autopilot.render import validate as render_validate
from autopilot.source import resolve
from autopilot.upload import thumbnail, youtube

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineOrchestrator",
    "StageDefinition",
    "StageResult",
    "StageStatus",
]


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


def _run_ingest(*, config: Any, db: Any) -> None:
    """INGEST stage: scan directory, insert media, normalize audio, mark duplicates."""
    files = scanner.scan_directory(config.input_dir, max_workers=None)
    norm_dir = config.output_dir / "normalized"
    norm_dir.mkdir(parents=True, exist_ok=True)

    for mf in files:
        media_id = mf.sha256_prefix or mf.file_path.stem
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
        normalizer.normalize_audio(
            mf.file_path, norm_dir, root_dir=config.input_dir
        )

    dedup.mark_duplicates(db)
    logger.info("Ingest complete: %d files scanned", len(files))


def _run_analyze(*, config: Any, db: Any) -> None:
    """ANALYZE stage: run all analysis passes on each media file."""
    scheduler = GPUScheduler(
        total_vram=0,
        device=config.processing.gpu_device,
    )

    all_media = db.list_all_media()
    media_list = [m for m in all_media if m.get("status") != "duplicate"]

    for media in media_list:
        media_id = media["id"]
        file_path = Path(media["file_path"])
        audio_path = file_path  # audio extracted from same file

        asr.transcribe_media(
            media_id, audio_path, db, scheduler, config.models,
            batch_size=config.processing.batch_size_whisper,
        )
        scenes.detect_shots(media_id, file_path, db, scheduler)
        objects.detect_objects(
            media_id, file_path, db, scheduler, config.models,
            sparse=False,
        )
        faces.detect_faces(media_id, file_path, db, scheduler, config.models)
        embeddings.compute_embeddings(
            media_id, file_path, db, scheduler, config.models,
        )
        audio_events.classify_audio_events(
            media_id, audio_path, db, scheduler,
        )

    faces.cluster_faces(db, eps=0.5, min_samples=3)
    logger.info("Analyze complete: %d media processed", len(media_list))


def _run_classify(*, config: Any, db: Any) -> None:
    """CLASSIFY stage: cluster and label activities."""
    cluster.cluster_activities(db)
    classify.label_activities(db, config.llm)
    logger.info("Classify complete")


def _run_narrate(
    *, config: Any, db: Any, human_review_fn: Callable | None = None,
) -> None:
    """NARRATE stage: build storyboard, propose narratives, human review."""
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


def _run_script(*, config: Any, db: Any) -> None:
    """SCRIPT stage: generate scripts for each approved narrative."""
    approved = db.list_narratives("approved")
    successes = 0
    for narr in approved:
        nid = narr["narrative_id"]
        try:
            script.generate_script(nid, db, config.llm)
            successes += 1
        except Exception:
            logger.exception("Script generation failed for narrative %s", nid)

    if approved and successes == 0:
        raise RuntimeError("All narratives failed script generation")

    logger.info("Script complete: %d/%d succeeded", successes, len(approved))


def _run_edl(*, config: Any, db: Any) -> None:
    """EDL stage: generate EDL, validate, and export OTIO per narrative."""
    approved = db.list_narratives("approved")
    successes = 0
    for narr in approved:
        nid = narr["narrative_id"]
        # Skip narratives without scripts
        if db.get_narrative_script(nid) is None:
            logger.warning("Skipping EDL for %s: no script", nid)
            continue
        try:
            edl = edl_mod.generate_edl(nid, db, config.llm)
            val_result = validator.validate_edl(edl, db)
            otio_path = config.output_dir / nid / "timeline.otio"
            otio_path.parent.mkdir(parents=True, exist_ok=True)
            otio_export.export_otio(edl, otio_path, db)


            db.upsert_edit_plan(
                nid,
                json.dumps(edl),
                otio_path=str(otio_path),
                validation_json=json.dumps(
                    {"passed": val_result.passed}
                ),
            )
            successes += 1
        except Exception:
            logger.exception("EDL generation failed for narrative %s", nid)

    logger.info("EDL complete: %d/%d succeeded", successes, len(approved))


def _run_source_assets(*, config: Any, db: Any) -> None:
    """SOURCE_ASSETS stage: resolve assets for each narrative with an edit plan."""
    approved = db.list_narratives("approved")
    successes = 0
    for narr in approved:
        nid = narr["narrative_id"]
        plan = db.get_edit_plan(nid)
        if plan is None:
            logger.warning("Skipping source for %s: no edit plan", nid)
            continue
        try:
            edl = json.loads(plan["edl_json"])
            asset_dir = config.output_dir / "assets" / nid
            asset_dir.mkdir(parents=True, exist_ok=True)
            resolve.resolve_edl_assets(
                edl, config.models, asset_dir, db, narrative_id=nid,
            )
            successes += 1
        except Exception:
            logger.exception("Source resolution failed for narrative %s", nid)

    logger.info("Source complete: %d/%d succeeded", successes, len(approved))


def _run_render(*, config: Any, db: Any) -> None:
    """RENDER stage: route, render, and validate per narrative."""
    approved = db.list_narratives("approved")
    successes = 0
    for narr in approved:
        nid = narr["narrative_id"]
        plan = db.get_edit_plan(nid)
        if plan is None:
            logger.warning("Skipping render for %s: no edit plan", nid)
            continue
        try:
            output_path = router.route_and_render(nid, db, config.output)
            edl = json.loads(plan["edl_json"])
            report = render_validate.validate_render(output_path, edl, config.output)
            for issue in report.issues:
                logger.warning(
                    "Render validation [%s] %s: %s",
                    nid, issue.severity, issue.message,
                )
            successes += 1
        except Exception:
            logger.exception("Render failed for narrative %s", nid)

    logger.info("Render complete: %d/%d succeeded", successes, len(approved))


def _run_upload(*, config: Any, db: Any) -> None:
    """UPLOAD stage: upload video and extract thumbnail per narrative."""
    approved = db.list_narratives("approved")
    successes = 0
    for narr in approved:
        nid = narr["narrative_id"]
        # Find rendered video
        render_dir = config.output_dir / "renders" / nid
        video_path = render_dir / "output.mp4"
        if not video_path.exists():
            logger.warning("Skipping upload for %s: no rendered video", nid)
            continue
        try:
            youtube.upload_video(nid, video_path, db, config.youtube)
            thumbnail.extract_best_thumbnail(nid, video_path, db)
            successes += 1
        except Exception:
            logger.exception("Upload failed for narrative %s", nid)

    logger.info("Upload complete: %d/%d succeeded", successes, len(approved))


class PipelineOrchestrator:
    """Manages a DAG of pipeline stages and executes them in topological order."""

    def __init__(
        self,
        budget_seconds: float | None = None,
        human_review_fn: Callable | None = None,
    ) -> None:
        self.budget_seconds = budget_seconds
        self.human_review_fn = human_review_fn

        # Wrap _run_narrate with human_review_fn via partial
        narrate_func = functools.partial(
            _run_narrate, human_review_fn=human_review_fn,
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
        self._stage_map: dict[str, StageDefinition] = {
            s.name: s for s in self.stages
        }

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

        for stage_name in order:
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

            logger.info("[RUNNING] %s", stage_name)
            t0 = time.monotonic()
            try:
                stage.func(config=config, db=db)
                elapsed = time.monotonic() - t0
                results[stage_name] = StageResult(
                    status=StageStatus.DONE,
                    elapsed_seconds=elapsed,
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
                logger.error("[ERROR] %s: %s", stage_name, exc)

            # Progress reporting after each stage
            cumulative = time.monotonic() - pipeline_start
            if self.budget_seconds and self.budget_seconds > 0:
                pct = (cumulative / self.budget_seconds) * 100
                logger.info(
                    "[PROGRESS] %.1fs / %.1fs budget (%.1f%%)",
                    cumulative, self.budget_seconds, pct,
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
                summary_parts.append(
                    f"  {sn}: {r.status.value} ({r.elapsed_seconds:.1f}s)"
                )
        summary = "\n".join(summary_parts)
        logger.info(
            "Pipeline complete (%.1fs)\n%s", total_elapsed, summary,
        )

        # Check budget
        if self.budget_seconds is not None and total_elapsed > self.budget_seconds:
            logger.warning(
                "Pipeline exceeded budget: %.1fs elapsed vs %.1fs budget",
                total_elapsed,
                self.budget_seconds,
            )

        return results
