"""Tests for the pipeline orchestrator (autopilot.orchestrator)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Pre-import submodules of packages with empty __init__.py so that
# @patch("autopilot.<pkg>.<sub>") can find the attribute.  Packages whose
# __init__.py already re-exports (ingest, organize, render, upload) don't
# need this.
import autopilot.analyze.asr  # noqa: F401
import autopilot.analyze.audio_events  # noqa: F401
import autopilot.analyze.embeddings  # noqa: F401
import autopilot.analyze.faces  # noqa: F401
import autopilot.analyze.gpu_scheduler  # noqa: F401
import autopilot.analyze.objects  # noqa: F401
import autopilot.analyze.scenes  # noqa: F401
import autopilot.plan.edl  # noqa: F401
import autopilot.plan.otio_export  # noqa: F401
import autopilot.plan.script  # noqa: F401
import autopilot.plan.validator  # noqa: F401
import autopilot.source.resolve  # noqa: F401
from autopilot.orchestrator import PipelineOrchestrator, StageDefinition, StageStatus

EXPECTED_STAGES = [
    "INGEST",
    "ANALYZE",
    "CLASSIFY",
    "NARRATE",
    "SCRIPT",
    "EDL",
    "SOURCE_ASSETS",
    "RENDER",
    "UPLOAD",
]

EXPECTED_DEPS = {
    "INGEST": [],
    "ANALYZE": ["INGEST"],
    "CLASSIFY": ["ANALYZE"],
    "NARRATE": ["CLASSIFY"],
    "SCRIPT": ["NARRATE"],
    "EDL": ["SCRIPT"],
    "SOURCE_ASSETS": ["EDL"],
    "RENDER": ["EDL", "SOURCE_ASSETS"],
    "UPLOAD": ["RENDER"],
}


class TestStageDefinition:
    """Tests for StageDefinition dataclass and StageStatus enum."""

    def test_stage_definition_fields(self) -> None:
        """StageDefinition stores name, func, dependencies, and estimated_seconds."""

        def func(config: object, db: object) -> None:
            pass

        stage = StageDefinition(
            name="INGEST",
            func=func,
            dependencies=[],
            estimated_seconds=300,
        )
        assert stage.name == "INGEST"
        assert stage.func is func
        assert stage.dependencies == []
        assert stage.estimated_seconds == 300

    def test_stage_definition_defaults(self) -> None:
        """Dependencies defaults to empty list, estimated_seconds defaults to 0."""

        def noop(config: object, db: object) -> None:
            pass

        stage = StageDefinition(name="TEST", func=noop)
        assert stage.dependencies == []
        assert stage.estimated_seconds == 0

    def test_stage_status_enum(self) -> None:
        """StageStatus has PENDING, RUNNING, DONE, SKIPPED, ERROR values."""
        assert StageStatus.PENDING.value == "pending"
        assert StageStatus.RUNNING.value == "running"
        assert StageStatus.DONE.value == "done"
        assert StageStatus.SKIPPED.value == "skipped"
        assert StageStatus.ERROR.value == "error"


class TestPipelineOrchestrator:
    """Tests for PipelineOrchestrator initialization."""

    def test_has_all_nine_stages(self) -> None:
        """PipelineOrchestrator has exactly 9 stages with correct names."""
        orch = PipelineOrchestrator()
        stage_names = [s.name for s in orch.stages]
        assert stage_names == EXPECTED_STAGES

    def test_stage_dependencies(self) -> None:
        """Each stage has the correct dependency list."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            assert stage.dependencies == EXPECTED_DEPS[stage.name], f"{stage.name} deps mismatch"

    def test_stages_have_callable_functions(self) -> None:
        """Every stage has a callable func attribute."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            assert callable(stage.func), f"{stage.name} func is not callable"

    def test_stages_have_positive_estimated_seconds(self) -> None:
        """Every stage has estimated_seconds >= 0."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            assert stage.estimated_seconds >= 0, f"{stage.name} has negative estimated_seconds"


class TestTopologicalSort:
    """Tests for execution_order() topological sort."""

    def test_execution_order_respects_dependencies(self) -> None:
        """Each stage appears after all its dependencies in execution_order()."""
        orch = PipelineOrchestrator()
        order = orch.execution_order()
        positions = {name: i for i, name in enumerate(order)}
        for stage in orch.stages:
            for dep in stage.dependencies:
                assert positions[dep] < positions[stage.name], (
                    f"{dep} must come before {stage.name}"
                )

    def test_execution_order_is_deterministic(self) -> None:
        """execution_order() returns the same result each call."""
        orch = PipelineOrchestrator()
        first = orch.execution_order()
        for _ in range(5):
            assert orch.execution_order() == first

    def test_ingest_is_first(self) -> None:
        """INGEST is always the first stage in execution order."""
        orch = PipelineOrchestrator()
        order = orch.execution_order()
        assert order[0] == "INGEST"

    def test_upload_is_last(self) -> None:
        """UPLOAD is always the last stage in execution order."""
        orch = PipelineOrchestrator()
        order = orch.execution_order()
        assert order[-1] == "UPLOAD"


class TestStageFunctions:
    """Tests for stage function registration (replaced stubs with real functions)."""

    def test_stage_functions_are_not_stubs(self) -> None:
        """Each stage's func is a real function, not a stub."""
        from autopilot.orchestrator import _stage_stub

        orch = PipelineOrchestrator()
        for stage in orch.stages:
            # For functools.partial, check the wrapped function
            func = getattr(stage.func, "func", stage.func)
            assert func is not _stage_stub, f"{stage.name} still uses _stage_stub"

    def test_stage_functions_accept_config_and_db_kwargs(self) -> None:
        """Each stage function's signature accepts config and db kwargs."""
        import inspect

        orch = PipelineOrchestrator()
        for stage in orch.stages:
            func = getattr(stage.func, "func", stage.func)
            sig = inspect.signature(func)
            param_names = set(sig.parameters.keys())
            assert "config" in param_names, f"{stage.name} func missing 'config' parameter"
            assert "db" in param_names, f"{stage.name} func missing 'db' parameter"


class TestRun:
    """Tests for PipelineOrchestrator.run() basic execution."""

    def test_run_calls_all_stages_in_order(self) -> None:
        """run() calls all 9 stage functions in topological order."""
        orch = PipelineOrchestrator()
        call_order: list[str] = []
        for stage in orch.stages:
            name = stage.name
            mock_fn = MagicMock(side_effect=lambda _n=name, **kw: call_order.append(_n))
            stage.func = mock_fn

        mock_config = MagicMock()
        mock_db = MagicMock()
        orch.run(config=mock_config, db=mock_db)

        assert call_order == orch.execution_order()

    def test_run_passes_config_and_db(self) -> None:
        """run() passes config and db to each stage function."""
        orch = PipelineOrchestrator()
        mocks: dict[str, MagicMock] = {}
        for stage in orch.stages:
            mock_fn = MagicMock()
            mocks[stage.name] = mock_fn
            stage.func = mock_fn

        mock_config = MagicMock()
        mock_db = MagicMock()
        orch.run(config=mock_config, db=mock_db)

        for name, mock_fn in mocks.items():
            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["config"] is mock_config
            assert call_kwargs["db"] is mock_db
            assert call_kwargs["force"] is False

    def test_run_returns_results_dict(self) -> None:
        """run() returns a dict mapping stage names to StageResult."""
        from autopilot.orchestrator import StageResult

        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        results = orch.run(config=MagicMock(), db=MagicMock())

        assert isinstance(results, dict)
        assert set(results.keys()) == {s.name for s in orch.stages}
        for name, result in results.items():
            assert isinstance(result, StageResult), f"{name} result not a StageResult"
            assert result.status == StageStatus.DONE
            assert result.elapsed_seconds >= 0


class TestProgressReporting:
    """Tests for progress reporting during run()."""

    def test_run_prints_stage_status(self, caplog: pytest.LogCaptureFixture) -> None:
        """run() logs RUNNING and DONE for each stage."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        log_text = caplog.text
        for stage in orch.stages:
            assert f"[RUNNING] {stage.name}" in log_text, f"Missing RUNNING log for {stage.name}"
            assert f"[DONE] {stage.name}" in log_text, f"Missing DONE log for {stage.name}"

    def test_run_prints_elapsed_time(self, caplog: pytest.LogCaptureFixture) -> None:
        """run() logs elapsed time for each completed stage."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        # Each DONE line should have a time like "(0.0s)"
        done_lines = [r.message for r in caplog.records if "[DONE]" in r.message]
        assert len(done_lines) == 9
        for line in done_lines:
            assert "s)" in line, f"Missing elapsed time in: {line}"

    def test_run_prints_total_elapsed(self, caplog: pytest.LogCaptureFixture) -> None:
        """run() logs total pipeline elapsed time at end."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        assert any("Pipeline complete" in r.message for r in caplog.records)


class TestErrorHandling:
    """Tests for error handling in run()."""

    def test_run_catches_stage_error(self) -> None:
        """run() doesn't crash when a stage raises; returns ERROR status."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()
        # Make ANALYZE raise
        orch._stage_map["ANALYZE"].func = MagicMock(side_effect=RuntimeError("analyze failed"))

        results = orch.run(config=MagicMock(), db=MagicMock())

        assert results["ANALYZE"].status == StageStatus.ERROR

    def test_run_skips_dependents_on_error(self) -> None:
        """When ANALYZE errors, CLASSIFY and all downstream are SKIPPED."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()
        orch._stage_map["ANALYZE"].func = MagicMock(side_effect=RuntimeError("analyze failed"))

        results = orch.run(config=MagicMock(), db=MagicMock())

        assert results["INGEST"].status == StageStatus.DONE
        assert results["ANALYZE"].status == StageStatus.ERROR
        # All stages that depend (transitively) on ANALYZE should be SKIPPED
        for name in ["CLASSIFY", "NARRATE", "SCRIPT", "EDL", "SOURCE_ASSETS", "RENDER", "UPLOAD"]:
            assert results[name].status == StageStatus.SKIPPED, f"{name} should be SKIPPED"

    def test_run_reports_error_message(self) -> None:
        """Error message is captured in the stage result."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()
        orch._stage_map["INGEST"].func = MagicMock(side_effect=RuntimeError("ingest boom"))

        results = orch.run(config=MagicMock(), db=MagicMock())

        assert results["INGEST"].error_message == "ingest boom"


class TestDryRun:
    """Tests for dry-run mode."""

    def test_dry_run_does_not_call_stages(self) -> None:
        """dry_run=True skips calling stage functions."""
        orch = PipelineOrchestrator()
        mocks: dict[str, MagicMock] = {}
        for stage in orch.stages:
            mock_fn = MagicMock()
            mocks[stage.name] = mock_fn
            stage.func = mock_fn

        orch.run(config=MagicMock(), db=MagicMock(), dry_run=True)

        for name, mock_fn in mocks.items():
            mock_fn.assert_not_called()

    def test_dry_run_prints_execution_plan(self, caplog: pytest.LogCaptureFixture) -> None:
        """dry_run=True logs DRY-RUN prefix for each stage."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock(), dry_run=True)

        log_text = caplog.text
        for stage in orch.stages:
            assert f"[DRY-RUN] {stage.name}" in log_text, f"Missing DRY-RUN log for {stage.name}"

    def test_dry_run_shows_estimated_time(self, caplog: pytest.LogCaptureFixture) -> None:
        """dry_run=True includes estimated_seconds in output."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock(), dry_run=True)

        for stage in orch.stages:
            expected = f"est. {stage.estimated_seconds}s"
            assert any(expected in r.message for r in caplog.records), (
                f"Missing estimated time for {stage.name}"
            )


class TestBudgetTracking:
    """Tests for wall-clock budget tracking."""

    def test_run_tracks_total_vs_budget(self) -> None:
        """Orchestrator created with budget_seconds makes it accessible."""
        orch = PipelineOrchestrator(budget_seconds=3600)
        assert orch.budget_seconds == 3600

        for stage in orch.stages:
            stage.func = MagicMock()

        orch.run(config=MagicMock(), db=MagicMock())
        # budget_seconds should remain accessible after run
        assert orch.budget_seconds == 3600

    def test_run_warns_when_over_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        """A warning is logged when cumulative time exceeds budget_seconds."""
        import time as _time

        orch = PipelineOrchestrator(budget_seconds=0.001)
        for stage in orch.stages:

            def slow_stage(_t=_time, **kw: object) -> None:
                _t.sleep(0.002)

            stage.func = slow_stage

        with caplog.at_level(logging.WARNING, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        assert any("exceeded budget" in r.message for r in caplog.records)


class TestIngestStage:
    """Tests for the real _run_ingest stage function."""

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_calls_scan_directory(
        self, mock_scanner, mock_normalizer, mock_dedup, minimal_config
    ):
        """_run_ingest calls scanner.scan_directory with config.input_dir."""
        from autopilot.orchestrator import _run_ingest

        mock_file = MagicMock()
        mock_file.file_path = Path("/fake/video.mp4")
        mock_scanner.scan_directory.return_value = [mock_file]
        db = MagicMock()

        _run_ingest(config=minimal_config, db=db)

        mock_scanner.scan_directory.assert_called_once_with(
            minimal_config.input_dir, max_workers=None
        )

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_inserts_media_into_db(
        self, mock_scanner, mock_normalizer, mock_dedup, minimal_config
    ):
        """_run_ingest calls db.insert_media for each scanned file."""
        from autopilot.orchestrator import _run_ingest

        mock_file1 = MagicMock()
        mock_file1.file_path = Path("/fake/v1.mp4")
        mock_file2 = MagicMock()
        mock_file2.file_path = Path("/fake/v2.mp4")
        mock_scanner.scan_directory.return_value = [mock_file1, mock_file2]
        db = MagicMock()
        db.get_media.return_value = None  # nothing ingested yet

        _run_ingest(config=minimal_config, db=db)

        assert db.insert_media.call_count == 2

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_normalizes_audio(
        self, mock_scanner, mock_normalizer, mock_dedup, minimal_config
    ):
        """_run_ingest calls normalize_audio for each media file."""
        from autopilot.orchestrator import _run_ingest

        mock_file = MagicMock()
        mock_file.file_path = Path("/fake/video.mp4")
        mock_scanner.scan_directory.return_value = [mock_file]
        db = MagicMock()
        db.get_media.return_value = None  # nothing ingested yet

        _run_ingest(config=minimal_config, db=db)

        mock_normalizer.normalize_audio.assert_called_once()

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_marks_duplicates(
        self, mock_scanner, mock_normalizer, mock_dedup, minimal_config
    ):
        """_run_ingest calls dedup.mark_duplicates with db."""
        from autopilot.orchestrator import _run_ingest

        mock_scanner.scan_directory.return_value = []
        db = MagicMock()

        _run_ingest(config=minimal_config, db=db)

        mock_dedup.mark_duplicates.assert_called_once_with(db)


class TestAnalyzeStage:
    """Tests for the real _run_analyze stage function."""

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_creates_gpu_scheduler(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """_run_analyze creates a GPUScheduler with config.processing settings."""
        from autopilot.orchestrator import _run_analyze

        db = MagicMock()
        db.list_all_media.return_value = []

        _run_analyze(config=minimal_config, db=db)

        mock_gpu_cls.assert_called_once()

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_runs_all_analysis_per_media(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """_run_analyze calls all 6 analysis functions for each media in DB."""
        from autopilot.orchestrator import _run_analyze

        media1 = {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"}
        media2 = {"id": "m2", "file_path": "/fake/v2.mp4", "status": "ingested"}
        db = MagicMock()
        db.list_all_media.return_value = [media1, media2]
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db)

        assert mock_asr.transcribe_media.call_count == 2
        assert mock_scenes.detect_shots.call_count == 2
        assert mock_objects.detect_objects.call_count == 2
        assert mock_faces.detect_faces.call_count == 2
        assert mock_embeddings.compute_embeddings.call_count == 2
        assert mock_audio_events.classify_audio_events.call_count == 2

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_calls_cluster_faces_after_analysis(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """_run_analyze calls cluster_faces once after all per-media analysis."""
        from autopilot.orchestrator import _run_analyze

        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"},
        ]
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db)

        mock_faces.cluster_faces.assert_called_once_with(db, eps=0.5, min_samples=3)

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_skips_duplicate_media(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """_run_analyze skips media with status='duplicate'."""
        from autopilot.orchestrator import _run_analyze

        media = [
            {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"},
            {"id": "m2", "file_path": "/fake/v2.mp4", "status": "duplicate"},
        ]
        db = MagicMock()
        db.list_all_media.return_value = media
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db)

        # Only m1 should be analyzed, not m2 (duplicate)
        assert mock_asr.transcribe_media.call_count == 1

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_calls_force_unload_all_on_success(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """_run_analyze calls scheduler.force_unload_all() after analysis completes."""
        from autopilot.orchestrator import _run_analyze

        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"},
        ]
        mock_scheduler = MagicMock()
        mock_gpu_cls.return_value = mock_scheduler

        _run_analyze(config=minimal_config, db=db)

        mock_scheduler.force_unload_all.assert_called_once()

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_calls_force_unload_all_on_error(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """scheduler.force_unload_all() is called even when analysis raises."""
        from autopilot.orchestrator import _run_analyze

        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"},
        ]
        mock_scheduler = MagicMock()
        mock_gpu_cls.return_value = mock_scheduler
        # Make cluster_faces raise to simulate error
        mock_faces.cluster_faces.side_effect = RuntimeError("GPU error")

        with pytest.raises(RuntimeError, match="GPU error"):
            _run_analyze(config=minimal_config, db=db)

        mock_scheduler.force_unload_all.assert_called_once()

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_continues_on_per_media_error(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """If analysis fails for one media, remaining media are still processed."""
        from autopilot.orchestrator import _run_analyze

        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"},
            {"id": "m2", "file_path": "/fake/v2.mp4", "status": "ingested"},
            {"id": "m3", "file_path": "/fake/v3.mp4", "status": "ingested"},
        ]
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False
        mock_scheduler = MagicMock()
        mock_gpu_cls.return_value = mock_scheduler
        # First media's ASR fails
        mock_asr.transcribe_media.side_effect = [
            RuntimeError("ASR failed"),
            None,
            None,
        ]

        _run_analyze(config=minimal_config, db=db)

        # m2 and m3 should still be analyzed (scenes.detect_shots for all 3)
        assert mock_scenes.detect_shots.call_count >= 2

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_logs_per_media_error(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
        caplog,
    ):
        """Error for a failed media is logged."""
        from autopilot.orchestrator import _run_analyze

        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"},
        ]
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False
        mock_scheduler = MagicMock()
        mock_gpu_cls.return_value = mock_scheduler
        mock_asr.transcribe_media.side_effect = RuntimeError("ASR failed")

        with caplog.at_level(logging.ERROR, logger="autopilot.orchestrator"):
            _run_analyze(config=minimal_config, db=db)

        assert any("m1" in r.message for r in caplog.records)

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_counts_failures(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
        caplog,
    ):
        """Log reports correct success/failure counts."""
        from autopilot.orchestrator import _run_analyze

        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"},
            {"id": "m2", "file_path": "/fake/v2.mp4", "status": "ingested"},
            {"id": "m3", "file_path": "/fake/v3.mp4", "status": "ingested"},
        ]
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False
        mock_scheduler = MagicMock()
        mock_gpu_cls.return_value = mock_scheduler
        mock_asr.transcribe_media.side_effect = [
            RuntimeError("fail"),
            None,
            None,
        ]

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            _run_analyze(config=minimal_config, db=db)

        assert any("2/3" in r.message for r in caplog.records)


class TestClassifyStage:
    """Tests for the real _run_classify stage function."""

    @patch("autopilot.organize.classify")
    @patch("autopilot.organize.cluster")
    def test_classify_calls_cluster_activities(self, mock_cluster, mock_classify, minimal_config):
        """_run_classify calls cluster.cluster_activities with db."""
        from autopilot.orchestrator import _run_classify

        db = MagicMock()
        db.get_activity_clusters.return_value = []  # nothing classified yet
        _run_classify(config=minimal_config, db=db)

        mock_cluster.cluster_activities.assert_called_once_with(db)

    @patch("autopilot.organize.classify")
    @patch("autopilot.organize.cluster")
    def test_classify_calls_label_activities(self, mock_cluster, mock_classify, minimal_config):
        """_run_classify calls classify.label_activities with db and config.llm."""
        from autopilot.orchestrator import _run_classify

        db = MagicMock()
        db.get_activity_clusters.return_value = []  # nothing classified yet
        _run_classify(config=minimal_config, db=db)

        mock_classify.label_activities.assert_called_once_with(db, minimal_config.llm)


class TestNarrateStage:
    """Tests for the real _run_narrate stage function."""

    @patch("autopilot.organize.narratives")
    def test_narrate_builds_storyboard(self, mock_narratives, minimal_config):
        """_run_narrate calls build_master_storyboard with db."""
        from autopilot.orchestrator import _run_narrate

        mock_narratives.build_master_storyboard.return_value = "storyboard text"
        mock_narratives.propose_narratives.return_value = []
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()
        db.list_narratives.return_value = []  # no checkpoint hit

        _run_narrate(config=minimal_config, db=db)

        mock_narratives.build_master_storyboard.assert_called_once_with(db)

    @patch("autopilot.organize.narratives")
    def test_narrate_proposes_narratives(self, mock_narratives, minimal_config):
        """_run_narrate calls propose_narratives with storyboard, db, config."""
        from autopilot.orchestrator import _run_narrate

        mock_narratives.build_master_storyboard.return_value = "storyboard"
        mock_narratives.propose_narratives.return_value = []
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()
        db.list_narratives.return_value = []  # no checkpoint hit

        _run_narrate(config=minimal_config, db=db)

        mock_narratives.propose_narratives.assert_called_once_with("storyboard", db, minimal_config)

    @patch("autopilot.organize.narratives")
    def test_narrate_calls_human_review_callback(self, mock_narratives, minimal_config):
        """_run_narrate invokes human_review_fn with formatted text and narratives."""
        from autopilot.orchestrator import _run_narrate

        narr = MagicMock()
        narr.narrative_id = "n1"
        mock_narratives.build_master_storyboard.return_value = "sb"
        mock_narratives.propose_narratives.return_value = [narr]
        mock_narratives.format_for_review.return_value = "review text"
        db = MagicMock()
        db.list_narratives.return_value = []  # no checkpoint hit

        review_fn = MagicMock(return_value=["n1"])
        _run_narrate(config=minimal_config, db=db, human_review_fn=review_fn)

        review_fn.assert_called_once_with("review text", [narr])

    @patch("autopilot.organize.narratives")
    def test_narrate_auto_approves_when_no_callback(self, mock_narratives, minimal_config):
        """Without human_review_fn all narratives get status='approved'."""
        from autopilot.orchestrator import _run_narrate

        narr = MagicMock()
        narr.narrative_id = "n1"
        mock_narratives.build_master_storyboard.return_value = "sb"
        mock_narratives.propose_narratives.return_value = [narr]
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()
        db.list_narratives.return_value = []  # no checkpoint hit

        _run_narrate(config=minimal_config, db=db)

        db.update_narrative_status.assert_any_call("n1", "approved")

    @patch("autopilot.organize.narratives")
    def test_narrate_respects_review_rejections(self, mock_narratives, minimal_config):
        """human_review_fn returns subset of IDs; rejected ones get status='rejected'."""
        from autopilot.orchestrator import _run_narrate

        narr1 = MagicMock()
        narr1.narrative_id = "n1"
        narr2 = MagicMock()
        narr2.narrative_id = "n2"
        mock_narratives.build_master_storyboard.return_value = "sb"
        mock_narratives.propose_narratives.return_value = [narr1, narr2]
        mock_narratives.format_for_review.return_value = "review"
        db = MagicMock()
        db.list_narratives.return_value = []  # no checkpoint hit

        # Only approve n1, reject n2
        review_fn = MagicMock(return_value=["n1"])
        _run_narrate(config=minimal_config, db=db, human_review_fn=review_fn)

        db.update_narrative_status.assert_any_call("n1", "approved")
        db.update_narrative_status.assert_any_call("n2", "rejected")


class TestNarrateResume:
    """Tests for _run_narrate checkpoint/resume logic."""

    @patch("autopilot.organize.narratives")
    def test_narrate_skips_when_approved_narratives_exist(self, mock_narratives, minimal_config):
        """_run_narrate skips LLM proposal when approved narratives already exist and force=False."""
        from autopilot.orchestrator import _run_narrate

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1", "status": "approved"},
            {"narrative_id": "n2", "status": "approved"},
        ]

        _run_narrate(config=minimal_config, db=db)

        mock_narratives.propose_narratives.assert_not_called()

    @patch("autopilot.organize.narratives")
    def test_narrate_logs_resume_message(self, mock_narratives, minimal_config, caplog):
        """_run_narrate logs 'Resuming NARRATE: N approved narratives already exist, skipping'."""
        from autopilot.orchestrator import _run_narrate

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1", "status": "approved"},
            {"narrative_id": "n2", "status": "approved"},
            {"narrative_id": "n3", "status": "approved"},
        ]

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            _run_narrate(config=minimal_config, db=db)

        assert any(
            "Resuming NARRATE" in r.message and "3" in r.message for r in caplog.records
        )

    @patch("autopilot.organize.narratives")
    def test_narrate_force_repropose_even_with_existing(self, mock_narratives, minimal_config):
        """_run_narrate with force=True re-proposes even when approved narratives exist."""
        from autopilot.orchestrator import _run_narrate

        mock_narratives.build_master_storyboard.return_value = "sb"
        mock_narratives.propose_narratives.return_value = []
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1", "status": "approved"},
        ]

        _run_narrate(config=minimal_config, db=db, force=True)

        mock_narratives.propose_narratives.assert_called_once()

    @patch("autopilot.organize.narratives")
    def test_narrate_proceeds_when_no_approved_narratives(self, mock_narratives, minimal_config):
        """_run_narrate proceeds normally when no approved narratives exist (first-time run)."""
        from autopilot.orchestrator import _run_narrate

        mock_narratives.build_master_storyboard.return_value = "sb"
        mock_narratives.propose_narratives.return_value = []
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()
        db.list_narratives.return_value = []

        _run_narrate(config=minimal_config, db=db)

        mock_narratives.propose_narratives.assert_called_once()


class TestScriptStage:
    """Tests for the real _run_script stage function."""

    @patch("autopilot.plan.script")
    def test_script_generates_per_approved_narrative(self, mock_script, minimal_config):
        """_run_script calls generate_script once per approved narrative."""
        from autopilot.orchestrator import _run_script

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_narrative_script.return_value = None
        mock_script.generate_script.return_value = {"scenes": []}

        _run_script(config=minimal_config, db=db)

        assert mock_script.generate_script.call_count == 2

    @patch("autopilot.plan.script")
    def test_script_continues_on_per_narrative_error(self, mock_script, minimal_config):
        """First narrative raises ScriptError, second still gets generated."""
        from autopilot.orchestrator import _run_script
        from autopilot.plan.script import ScriptError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_narrative_script.return_value = None
        mock_script.generate_script.side_effect = [
            ScriptError("failed"),
            {"scenes": []},
        ]

        _run_script(config=minimal_config, db=db)

        assert mock_script.generate_script.call_count == 2

    @patch("autopilot.plan.script")
    def test_script_raises_if_all_narratives_fail(self, mock_script, minimal_config):
        """If every narrative fails, stage itself raises RuntimeError."""
        from autopilot.orchestrator import _run_script
        from autopilot.plan.script import ScriptError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_narrative_script.return_value = None
        mock_script.generate_script.side_effect = ScriptError("fail")

        with pytest.raises(RuntimeError, match="All narratives failed"):
            _run_script(config=minimal_config, db=db)


class TestEdlStage:
    """Tests for the real _run_edl stage function."""

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_generates_validates_exports_per_narrative(
        self, mock_edl, mock_validator, mock_otio, minimal_config
    ):
        """_run_edl calls generate_edl, validate_edl, export_otio for each narrative."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_narrative_script.return_value = {"scenes": []}
        db.get_edit_plan.return_value = None
        mock_edl.generate_edl.return_value = {"timeline": []}
        mock_validator.validate_edl.return_value = MagicMock(passed=True)
        mock_otio.export_otio.return_value = Path("/out/timeline.otio")

        _run_edl(config=minimal_config, db=db)

        mock_edl.generate_edl.assert_called_once()
        mock_validator.validate_edl.assert_called_once()
        mock_otio.export_otio.assert_called_once()

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_stores_validation_result(
        self, mock_edl, mock_validator, mock_otio, minimal_config
    ):
        """_run_edl upsert only passes otio_path; generate_edl() handles edl_json and validation_json."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_narrative_script.return_value = {"scenes": []}
        db.get_edit_plan.return_value = None
        mock_edl.generate_edl.return_value = {"timeline": []}
        val_result = MagicMock(passed=True)
        mock_validator.validate_edl.return_value = val_result
        mock_otio.export_otio.return_value = Path("/out/timeline.otio")

        _run_edl(config=minimal_config, db=db)

        db.upsert_edit_plan.assert_called_once()
        call_args = db.upsert_edit_plan.call_args
        assert call_args.args == ("n1",), "Only narrative_id should be positional"
        assert "otio_path" in call_args.kwargs
        assert "validation_json" not in call_args.kwargs

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_continues_on_failure(self, mock_edl, mock_validator, mock_otio, minimal_config):
        """One narrative fails EDL, others still processed."""
        from autopilot.orchestrator import _run_edl
        from autopilot.plan.edl import EdlError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_narrative_script.return_value = {"scenes": []}
        db.get_edit_plan.return_value = None
        mock_edl.generate_edl.side_effect = [
            EdlError("fail"),
            {"timeline": []},
        ]
        mock_validator.validate_edl.return_value = MagicMock(passed=True)
        mock_otio.export_otio.return_value = Path("/out/timeline.otio")

        _run_edl(config=minimal_config, db=db)

        # Second narrative should still be processed
        assert mock_edl.generate_edl.call_count == 2

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_raises_if_all_narratives_fail(
        self, mock_edl, mock_validator, mock_otio, minimal_config
    ):
        """If every narrative fails EDL, stage raises RuntimeError."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_narrative_script.return_value = {"scenes": []}
        db.get_edit_plan.return_value = None
        mock_edl.generate_edl.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError, match="All narratives failed"):
            _run_edl(config=minimal_config, db=db)

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_upsert_only_passes_otio_path(
        self, mock_edl, mock_validator, mock_otio, minimal_config
    ):
        """_run_edl upsert_edit_plan only passes otio_path, not edl_json or validation_json.

        generate_edl() already persists edl_json and a rich validation_json.
        The orchestrator should only add otio_path after OTIO export.
        """
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_narrative_script.return_value = {"scenes": []}
        db.get_edit_plan.return_value = None
        mock_edl.generate_edl.return_value = {"timeline": []}
        mock_validator.validate_edl.return_value = MagicMock(passed=True)

        _run_edl(config=minimal_config, db=db)

        db.upsert_edit_plan.assert_called_once()
        call_args = db.upsert_edit_plan.call_args
        # Should only pass narrative_id and otio_path
        assert call_args.args == ("n1",), f"Expected only nid positional arg, got {call_args.args}"
        assert "otio_path" in call_args.kwargs, "otio_path kwarg missing"
        assert "validation_json" not in call_args.kwargs, "validation_json should not be passed"


class TestSourceStage:
    """Tests for the real _run_source_assets stage function."""

    @patch("autopilot.source.resolve")
    def test_source_resolves_assets_per_narrative(self, mock_resolve, minimal_config):
        """_run_source_assets calls resolve_edl_assets for each narrative with an edit plan."""
        from autopilot.orchestrator import _run_source_assets

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }
        mock_resolve.resolve_edl_assets.return_value = {"edl": {}, "unresolved": []}

        _run_source_assets(config=minimal_config, db=db)

        mock_resolve.resolve_edl_assets.assert_called_once()
        call_kwargs = mock_resolve.resolve_edl_assets.call_args
        assert call_kwargs[1]["narrative_id"] == "n1"

    @patch("autopilot.source.resolve")
    def test_source_raises_if_all_narratives_fail(self, mock_resolve, minimal_config):
        """If every narrative fails source resolution, stage raises RuntimeError."""
        from autopilot.orchestrator import _run_source_assets

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }
        mock_resolve.resolve_edl_assets.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError, match="All narratives failed"):
            _run_source_assets(config=minimal_config, db=db)


class TestRenderStage:
    """Tests for the real _run_render stage function."""

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_routes_and_validates_per_narrative(
        self, mock_router, mock_validate, minimal_config
    ):
        """_run_render calls route_and_render and validate_render per narrative."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }
        mock_router.route_and_render.return_value = Path("/out/video.mp4")
        mock_validate.validate_render.return_value = MagicMock(passed=True, issues=[])

        _run_render(config=minimal_config, db=db)

        mock_router.route_and_render.assert_called_once()
        mock_validate.validate_render.assert_called_once()

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_logs_validation_warnings(
        self, mock_router, mock_validate, minimal_config, caplog
    ):
        """Validation issues are logged."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }
        mock_router.route_and_render.return_value = Path("/out/video.mp4")
        issue = MagicMock()
        issue.severity = "warning"
        issue.message = "Low bitrate"
        mock_validate.validate_render.return_value = MagicMock(passed=True, issues=[issue])

        with caplog.at_level(logging.WARNING, logger="autopilot.orchestrator"):
            _run_render(config=minimal_config, db=db)

        assert any("Low bitrate" in r.message for r in caplog.records)

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_continues_on_failure(self, mock_router, mock_validate, minimal_config):
        """One narrative's render fails, others still processed."""
        from autopilot.orchestrator import _run_render
        from autopilot.render.router import RoutingError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }
        mock_router.route_and_render.side_effect = [
            RoutingError("fail"),
            Path("/out/video.mp4"),
        ]
        mock_validate.validate_render.return_value = MagicMock(passed=True, issues=[])

        _run_render(config=minimal_config, db=db)

        assert mock_router.route_and_render.call_count == 2

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_persists_render_path_to_db(self, mock_router, mock_validate, minimal_config):
        """_run_render persists render output path to DB via upsert_edit_plan."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }
        mock_router.route_and_render.return_value = Path("/out/renders/n1/output.mp4")
        mock_validate.validate_render.return_value = MagicMock(passed=True, issues=[])

        _run_render(config=minimal_config, db=db)

        db.upsert_edit_plan.assert_called_once()
        call_kwargs = db.upsert_edit_plan.call_args
        assert call_kwargs[1]["render_path"] == "/out/renders/n1/output.mp4"

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_raises_if_all_narratives_fail(self, mock_router, mock_validate, minimal_config):
        """If every narrative fails render, stage raises RuntimeError."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }
        mock_router.route_and_render.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError, match="All narratives failed"):
            _run_render(config=minimal_config, db=db)


class TestUploadStage:
    """Tests for the real _run_upload stage function."""

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_uploads_and_thumbnails_per_narrative(
        self, mock_youtube, mock_thumbnail, minimal_config
    ):
        """_run_upload calls upload_video and extract_best_thumbnail per narrative."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": "{}",
            "render_path": "/out/renders/n1/output.mp4",
        }
        db.get_upload.return_value = None

        mock_youtube.upload_video.return_value = "https://youtu.be/abc"
        mock_thumbnail.extract_best_thumbnail.return_value = Path("/thumb.jpg")

        _run_upload(config=minimal_config, db=db)

        mock_youtube.upload_video.assert_called_once()
        mock_thumbnail.extract_best_thumbnail.assert_called_once()

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_continues_on_failure(self, mock_youtube, mock_thumbnail, minimal_config):
        """One narrative fails upload, others still uploaded."""
        from autopilot.orchestrator import _run_upload
        from autopilot.upload.youtube import UploadError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "edl_json": "{}",
            "render_path": "/out/renders/output.mp4",
        }
        db.get_upload.return_value = None

        mock_youtube.upload_video.side_effect = [
            UploadError("fail"),
            "https://youtu.be/def",
        ]
        mock_thumbnail.extract_best_thumbnail.return_value = Path("/thumb.jpg")

        _run_upload(config=minimal_config, db=db)

        assert mock_youtube.upload_video.call_count == 2

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_reads_render_path_from_db(self, mock_youtube, mock_thumbnail, minimal_config):
        """_run_upload reads render_path from DB instead of filesystem convention."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": "{}",
            "render_path": "/db/stored/path.mp4",
        }
        db.get_upload.return_value = None
        mock_youtube.upload_video.return_value = "https://youtu.be/abc"
        mock_thumbnail.extract_best_thumbnail.return_value = Path("/thumb.jpg")

        _run_upload(config=minimal_config, db=db)

        # Verify upload_video was called with the DB-stored path
        call_args = mock_youtube.upload_video.call_args
        assert call_args[0][1] == Path("/db/stored/path.mp4")

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_skips_narrative_when_no_render_path(
        self, mock_youtube, mock_thumbnail, minimal_config, caplog
    ):
        """Narrative is skipped with warning when no render_path in edit_plan."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": "{}",
        }
        db.get_upload.return_value = None

        with caplog.at_level(logging.WARNING, logger="autopilot.orchestrator"):
            with pytest.raises(RuntimeError, match="All narratives failed"):
                _run_upload(config=minimal_config, db=db)

        mock_youtube.upload_video.assert_not_called()
        assert any("n1" in r.message for r in caplog.records)

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_raises_if_all_narratives_fail(
        self, mock_youtube, mock_thumbnail, minimal_config
    ):
        """If every narrative fails upload, stage raises RuntimeError."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "edl_json": "{}",
            "render_path": "/out/renders/output.mp4",
        }
        db.get_upload.return_value = None

        mock_youtube.upload_video.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError, match="All narratives failed"):
            _run_upload(config=minimal_config, db=db)


class TestRealStageRegistration:
    """Tests for PipelineOrchestrator registering real stage functions."""

    def test_orchestrator_registers_real_functions_not_stubs(self):
        """Each stage's func is not a stub."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            func = getattr(stage.func, "func", stage.func)
            name = getattr(func, "__name__", "")
            assert "stub" not in name.lower(), f"{stage.name} still uses a stub function"

    def test_orchestrator_accepts_human_review_fn_parameter(self):
        """PipelineOrchestrator(human_review_fn=callback) stores it."""
        callback = MagicMock()
        orch = PipelineOrchestrator(human_review_fn=callback)
        assert orch.human_review_fn is callback

    def test_orchestrator_human_review_fn_defaults_to_none(self):
        """PipelineOrchestrator() defaults human_review_fn to None."""
        orch = PipelineOrchestrator()
        assert orch.human_review_fn is None


class TestEnhancedProgress:
    """Tests for enhanced progress reporting in run()."""

    def test_run_logs_cumulative_elapsed_per_stage(self, caplog: pytest.LogCaptureFixture):
        """After each stage, log shows cumulative time and remaining budget."""
        orch = PipelineOrchestrator(budget_seconds=3600)
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        progress_lines = [r.message for r in caplog.records if "[PROGRESS]" in r.message]
        # Should have one progress line per stage
        assert len(progress_lines) == 9
        # Each should contain budget info
        for line in progress_lines:
            assert "budget" in line.lower() or "%" in line

    def test_run_logs_summary_table_at_end(self, caplog: pytest.LogCaptureFixture):
        """Final log message includes all stage names with their elapsed times."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        # Look for summary that mentions all stage names
        all_text = " ".join(r.message for r in caplog.records)
        for stage_name in [
            "INGEST",
            "ANALYZE",
            "CLASSIFY",
            "NARRATE",
            "SCRIPT",
            "EDL",
            "SOURCE_ASSETS",
            "RENDER",
            "UPLOAD",
        ]:
            assert stage_name in all_text


class TestShutdownAPI:
    """Tests for shutdown_requested() and request_shutdown() API."""

    def setup_method(self) -> None:
        """Reset shutdown state before each test."""
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def teardown_method(self) -> None:
        """Reset shutdown state after each test."""
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def test_shutdown_event_exists(self) -> None:
        """Module-level _shutdown_event is a threading.Event."""
        import threading

        from autopilot.orchestrator import _shutdown_event

        assert isinstance(_shutdown_event, threading.Event)

    def test_shutdown_requested_initially_false(self) -> None:
        """shutdown_requested() returns False initially."""
        from autopilot.orchestrator import shutdown_requested

        assert shutdown_requested() is False

    def test_request_shutdown_makes_shutdown_requested_true(self) -> None:
        """request_shutdown() makes shutdown_requested() return True."""
        from autopilot.orchestrator import request_shutdown, shutdown_requested

        request_shutdown()
        assert shutdown_requested() is True

    def test_reset_shutdown_restores_false(self) -> None:
        """_reset_shutdown() restores shutdown_requested() to False."""
        from autopilot.orchestrator import (
            _reset_shutdown,
            request_shutdown,
            shutdown_requested,
        )

        request_shutdown()
        assert shutdown_requested() is True
        _reset_shutdown()
        assert shutdown_requested() is False

    def test_shutdown_requested_in_all(self) -> None:
        """shutdown_requested is exported in __all__."""
        from autopilot import orchestrator

        assert "shutdown_requested" in orchestrator.__all__

    def test_request_shutdown_in_all(self) -> None:
        """request_shutdown is exported in __all__."""
        from autopilot import orchestrator

        assert "request_shutdown" in orchestrator.__all__


class TestShutdownBetweenStages:
    """Tests for shutdown checks between pipeline stages in run()."""

    def setup_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def teardown_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def test_run_skips_remaining_stages_after_shutdown(self) -> None:
        """When shutdown is requested after INGEST, remaining stages are SKIPPED."""
        from autopilot.orchestrator import request_shutdown

        orch = PipelineOrchestrator()
        call_order: list[str] = []

        def make_func(name: str):
            def func(**kw):
                call_order.append(name)
                if name == "INGEST":
                    request_shutdown()

            return func

        for stage in orch.stages:
            stage.func = make_func(stage.name)

        results = orch.run(config=MagicMock(), db=MagicMock())

        # Only INGEST should have been called
        assert call_order == ["INGEST"]
        assert results["INGEST"].status == StageStatus.DONE

        # All remaining stages should be SKIPPED
        for name in [
            "ANALYZE",
            "CLASSIFY",
            "NARRATE",
            "SCRIPT",
            "EDL",
            "SOURCE_ASSETS",
            "RENDER",
            "UPLOAD",
        ]:
            assert results[name].status == StageStatus.SKIPPED, (
                f"{name} should be SKIPPED after shutdown"
            )

    def test_run_logs_shutdown_skip(self, caplog: pytest.LogCaptureFixture) -> None:
        """run() logs a message when skipping stages due to shutdown."""
        from autopilot.orchestrator import request_shutdown

        orch = PipelineOrchestrator()

        def ingest_then_shutdown(**kw):
            request_shutdown()

        for stage in orch.stages:
            stage.func = MagicMock()
        orch._stage_map["INGEST"].func = ingest_then_shutdown

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        assert any(
            "shutdown" in r.message.lower() and "skip" in r.message.lower() for r in caplog.records
        )


class TestIngestShutdown:
    """Tests for _run_ingest breaking early on shutdown."""

    def setup_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def teardown_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()


# -- Checkpoint/resume tests for _run_ingest ---------------------------------


class TestIngestResume:
    """Tests for _run_ingest checkpoint/resume logic."""

    def setup_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def teardown_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_breaks_on_shutdown(
        self,
        mock_scanner,
        mock_normalizer,
        mock_dedup,
        minimal_config,
    ) -> None:
        """_run_ingest breaks early when shutdown is requested mid-iteration."""
        from autopilot.orchestrator import _run_ingest, request_shutdown

        # Create 3 mock files
        files = []
        for i in range(3):
            mf = MagicMock()
            mf.file_path = Path(f"/fake/v{i}.mp4")
            mf.sha256_prefix = f"sha{i}"
            files.append(mf)
        mock_scanner.scan_directory.return_value = files

        db = MagicMock()
        # Resume logic checks db.get_media(); return None so files aren't skipped
        db.get_media.return_value = None
        call_count = 0

        def normalize_and_shutdown(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_shutdown()

        mock_normalizer.normalize_audio.side_effect = normalize_and_shutdown

        _run_ingest(config=minimal_config, db=db)

        # Only the first file should be fully processed; second iteration
        # should see shutdown and break before processing
        assert mock_normalizer.normalize_audio.call_count == 1

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_skips_already_ingested_media(
        self,
        mock_scanner,
        mock_normalizer,
        mock_dedup,
        minimal_config,
    ):
        """_run_ingest skips insert+normalize for media already in the DB."""
        from autopilot.orchestrator import _run_ingest

        mock_file1 = MagicMock()
        mock_file1.sha256_prefix = "hash1"
        mock_file1.file_path = Path("/fake/v1.mp4")
        mock_file2 = MagicMock()
        mock_file2.sha256_prefix = "hash2"
        mock_file2.file_path = Path("/fake/v2.mp4")
        mock_scanner.scan_directory.return_value = [mock_file1, mock_file2]

        db = MagicMock()
        # hash1 already exists, hash2 is new
        db.get_media.side_effect = lambda mid: {"id": mid} if mid == "hash1" else None

        _run_ingest(config=minimal_config, db=db)

        # Only hash2 should get insert_media + normalize_audio
        assert db.insert_media.call_count == 1
        assert mock_normalizer.normalize_audio.call_count == 1

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_still_deduplicates_on_shutdown(
        self,
        mock_scanner,
        mock_normalizer,
        mock_dedup,
        minimal_config,
    ) -> None:
        """dedup.mark_duplicates is still called even when shutdown breaks the loop."""
        from autopilot.orchestrator import _run_ingest, request_shutdown

        mf = MagicMock()
        mf.file_path = Path("/fake/v0.mp4")
        mf.sha256_prefix = "sha0"
        mock_scanner.scan_directory.return_value = [mf, MagicMock(), MagicMock()]

        db = MagicMock()
        db.insert_media.side_effect = lambda *a, **kw: request_shutdown()

        _run_ingest(config=minimal_config, db=db)

        mock_dedup.mark_duplicates.assert_called_once_with(db)


class TestAnalyzeShutdown:
    """Tests for _run_analyze breaking early on shutdown."""

    def setup_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def teardown_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_breaks_on_shutdown(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ) -> None:
        """_run_analyze breaks after first media when shutdown is requested."""
        from autopilot.orchestrator import _run_analyze, request_shutdown

        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": "m1", "file_path": "/f/v1.mp4", "status": "ok"},
            {"id": "m2", "file_path": "/f/v2.mp4", "status": "ok"},
            {"id": "m3", "file_path": "/f/v3.mp4", "status": "ok"},
        ]
        # Resume logic: has_* methods must return False so passes aren't skipped
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False

        call_count = 0

        def asr_and_shutdown(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_shutdown()

        mock_asr.transcribe_media.side_effect = asr_and_shutdown

        _run_analyze(config=minimal_config, db=db)

        # Only 1 media should be analyzed
        assert mock_asr.transcribe_media.call_count == 1

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_unloads_gpu_on_shutdown(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ) -> None:
        """scheduler.force_unload_all() is still called on shutdown."""
        from autopilot.orchestrator import _run_analyze, request_shutdown

        mock_scheduler = MagicMock()
        mock_gpu_cls.return_value = mock_scheduler

        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": "m1", "file_path": "/f/v1.mp4", "status": "ok"},
        ]
        mock_asr.transcribe_media.side_effect = lambda *a, **kw: request_shutdown()

        _run_analyze(config=minimal_config, db=db)

        mock_scheduler.force_unload_all.assert_called_once()


class TestRemainingStagesShutdown:
    """Tests for shutdown in _run_script, _run_edl, _run_source_assets, _run_render, _run_upload."""

    def setup_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def teardown_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    @patch("autopilot.plan.script")
    def test_script_breaks_on_shutdown(self, mock_script, minimal_config) -> None:
        """_run_script breaks after first narrative when shutdown requested."""
        from autopilot.orchestrator import _run_script, request_shutdown

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        # Resume logic: get_narrative_script must return None so narratives
        # aren't filtered out as already-scripted
        db.get_narrative_script.return_value = None

        call_count = 0

        def gen_and_shutdown(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_shutdown()

        mock_script.generate_script.side_effect = gen_and_shutdown

        _run_script(config=minimal_config, db=db)

        assert mock_script.generate_script.call_count == 1

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_breaks_on_shutdown(
        self,
        mock_edl_mod,
        mock_validator,
        mock_otio,
        minimal_config,
    ) -> None:
        """_run_edl breaks after first narrative when shutdown requested."""
        from autopilot.orchestrator import _run_edl, request_shutdown

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        # Resume logic: get_edit_plan must return None so narratives aren't
        # filtered out as already having edit plans
        db.get_edit_plan.return_value = None
        db.get_narrative_script.return_value = "some script"
        mock_edl_mod.generate_edl.return_value = []
        mock_validator.validate_edl.return_value = MagicMock(passed=True)

        call_count = 0

        def edl_and_shutdown(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_shutdown()
            return []

        mock_edl_mod.generate_edl.side_effect = edl_and_shutdown

        _run_edl(config=minimal_config, db=db)

        assert mock_edl_mod.generate_edl.call_count == 1

    @patch("autopilot.source.resolve")
    def test_source_assets_breaks_on_shutdown(
        self,
        mock_resolve,
        minimal_config,
    ) -> None:
        """_run_source_assets breaks after first narrative when shutdown requested."""
        from autopilot.orchestrator import _run_source_assets, request_shutdown

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {"edl_json": "[]"}

        call_count = 0

        def resolve_and_shutdown(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_shutdown()

        mock_resolve.resolve_edl_assets.side_effect = resolve_and_shutdown

        _run_source_assets(config=minimal_config, db=db)

        assert mock_resolve.resolve_edl_assets.call_count == 1

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_breaks_on_shutdown(
        self,
        mock_router,
        mock_render_validate,
        minimal_config,
    ) -> None:
        """_run_render breaks after first narrative when shutdown requested."""
        from autopilot.orchestrator import _run_render, request_shutdown

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {"edl_json": "[]"}
        mock_router.route_and_render.return_value = Path("/fake/out.mp4")
        mock_render_validate.validate_render.return_value = MagicMock(issues=[])

        call_count = 0

        def render_and_shutdown(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_shutdown()
            return Path("/fake/out.mp4")

        mock_router.route_and_render.side_effect = render_and_shutdown

        _run_render(config=minimal_config, db=db)

        assert mock_router.route_and_render.call_count == 1

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_breaks_on_shutdown(
        self,
        mock_youtube,
        mock_thumbnail,
        minimal_config,
    ) -> None:
        """_run_upload breaks after first narrative when shutdown requested."""
        from autopilot.orchestrator import _run_upload, request_shutdown

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        # Resume logic: get_upload must return None so narratives aren't
        # filtered out as already-uploaded
        db.get_upload.return_value = None
        db.get_edit_plan.return_value = {
            "edl_json": "[]",
            "render_path": "/fake/out.mp4",
        }

        call_count = 0

        def upload_and_shutdown(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_shutdown()

        mock_youtube.upload_video.side_effect = upload_and_shutdown

        _run_upload(config=minimal_config, db=db)

        assert mock_youtube.upload_video.call_count == 1


class TestShutdownSkipDetails:
    """Tests for shutdown-skipped stage result details and summary logging."""

    def setup_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def teardown_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def test_shutdown_skipped_stages_have_no_error_message(self) -> None:
        """Stages skipped due to shutdown have status=SKIPPED and no error_message."""
        from autopilot.orchestrator import request_shutdown

        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        # Request shutdown after INGEST completes
        def ingest_then_shutdown(**kw):
            request_shutdown()

        orch._stage_map["INGEST"].func = ingest_then_shutdown

        results = orch.run(config=MagicMock(), db=MagicMock())

        # INGEST should be DONE with no error_message
        assert results["INGEST"].status == StageStatus.DONE
        assert results["INGEST"].error_message is None

        # All remaining stages should be SKIPPED (no error_message — the
        # between-stages shutdown check fills them as plain SKIPPED)
        for name in [
            "ANALYZE",
            "CLASSIFY",
            "NARRATE",
            "SCRIPT",
            "EDL",
            "SOURCE_ASSETS",
            "RENDER",
            "UPLOAD",
        ]:
            result = results[name]
            assert result.status == StageStatus.SKIPPED, f"{name} should be SKIPPED"
            assert result.error_message is None, (
                f"{name} should have error_message=None, got {result.error_message!r}"
            )

    def test_shutdown_skipped_distinct_from_dependency_skipped(self) -> None:
        """Shutdown-skipped stages have error_message; dependency-skipped do not."""
        orch = PipelineOrchestrator()

        call_order: list[str] = []

        def make_func(name: str):
            def func(**kw):
                call_order.append(name)
                if name == "ANALYZE":
                    raise RuntimeError("analysis failed")

            return func

        # Replace all stage funcs; ANALYZE will fail, then we request shutdown
        # after CLASSIFY (which should be dependency-skipped)
        for stage in orch.stages:
            stage.func = make_func(stage.name)

        # Make INGEST succeed, ANALYZE fail
        results = orch.run(config=MagicMock(), db=MagicMock())

        # ANALYZE errored → CLASSIFY should be dependency-skipped (no error_message)
        assert results["CLASSIFY"].status == StageStatus.SKIPPED
        assert results["CLASSIFY"].error_message is None

    def test_summary_log_includes_shutdown_skipped_stages(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pipeline summary log shows both done and skipped stages."""
        from autopilot.orchestrator import request_shutdown

        orch = PipelineOrchestrator()

        def ingest_then_shutdown(**kw):
            request_shutdown()

        for stage in orch.stages:
            stage.func = MagicMock()
        orch._stage_map["INGEST"].func = ingest_then_shutdown

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        # Find the summary log message
        summary_records = [r for r in caplog.records if "Pipeline complete" in r.message]
        assert len(summary_records) == 1, "Expected exactly one pipeline summary log"
        summary = summary_records[0].message

        # INGEST should show as done
        assert "INGEST: done" in summary
        # Remaining stages should show as skipped
        assert "ANALYZE: skipped" in summary
        assert "UPLOAD: skipped" in summary

    def test_shutdown_log_for_skipped_stages(self, caplog: pytest.LogCaptureFixture) -> None:
        """Shutdown emits a single log about skipping remaining stages."""
        from autopilot.orchestrator import request_shutdown

        orch = PipelineOrchestrator()

        def ingest_then_shutdown(**kw):
            request_shutdown()

        for stage in orch.stages:
            stage.func = MagicMock()
        orch._stage_map["INGEST"].func = ingest_then_shutdown

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        # The run() method emits a single shutdown-skip log message
        shutdown_logs = [
            r for r in caplog.records if "skipping remaining stages" in r.message.lower()
        ]
        assert len(shutdown_logs) == 1, f"Expected 1 shutdown-skip log, got {len(shutdown_logs)}"

        # The summary log should list all skipped stages
        summary_records = [r for r in caplog.records if "Pipeline complete" in r.message]
        assert len(summary_records) == 1
        summary = summary_records[0].message
        for name in [
            "ANALYZE",
            "CLASSIFY",
            "NARRATE",
            "SCRIPT",
            "EDL",
            "SOURCE_ASSETS",
            "RENDER",
            "UPLOAD",
        ]:
            assert f"{name}: skipped" in summary, f"Missing skipped entry for {name} in summary"

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_logs_resume_counts(
        self,
        mock_scanner,
        mock_normalizer,
        mock_dedup,
        minimal_config,
        caplog,
    ):
        """_run_ingest logs 'Resuming INGEST: N/M files already ingested'."""
        from autopilot.orchestrator import _run_ingest

        mock_file1 = MagicMock()
        mock_file1.sha256_prefix = "hash1"
        mock_file1.file_path = Path("/fake/v1.mp4")
        mock_file2 = MagicMock()
        mock_file2.sha256_prefix = "hash2"
        mock_file2.file_path = Path("/fake/v2.mp4")
        mock_scanner.scan_directory.return_value = [mock_file1, mock_file2]

        db = MagicMock()
        db.get_media.side_effect = lambda mid: {"id": mid} if mid == "hash1" else None

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            _run_ingest(config=minimal_config, db=db)

        assert any("Resuming INGEST" in r.message and "1/2" in r.message for r in caplog.records)

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_force_reprocesses_all(
        self,
        mock_scanner,
        mock_normalizer,
        mock_dedup,
        minimal_config,
    ):
        """_run_ingest with force=True reprocesses even existing media."""
        from autopilot.orchestrator import _run_ingest

        mock_file1 = MagicMock()
        mock_file1.sha256_prefix = "hash1"
        mock_file1.file_path = Path("/fake/v1.mp4")
        mock_file2 = MagicMock()
        mock_file2.sha256_prefix = "hash2"
        mock_file2.file_path = Path("/fake/v2.mp4")
        mock_scanner.scan_directory.return_value = [mock_file1, mock_file2]

        db = MagicMock()
        db.get_media.return_value = {"id": "existing"}

        _run_ingest(config=minimal_config, db=db, force=True)

        # Both should be processed despite existing in DB
        assert db.insert_media.call_count == 2
        assert mock_normalizer.normalize_audio.call_count == 2


# -- Checkpoint/resume tests for _run_analyze --------------------------------


class TestAnalyzeResume:
    """Tests for _run_analyze per-pass checkpoint/resume logic."""

    ANALYZE_PATCHES = [
        "autopilot.analyze.gpu_scheduler.GPUScheduler",
        "autopilot.analyze.faces",
        "autopilot.analyze.audio_events",
        "autopilot.analyze.embeddings",
        "autopilot.analyze.objects",
        "autopilot.analyze.scenes",
        "autopilot.analyze.asr",
    ]

    def _make_db_with_media(self, media_ids, **has_overrides):
        """Create a MagicMock db with specified media and has_* defaults."""
        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": mid, "file_path": f"/fake/{mid}.mp4", "status": "ingested"} for mid in media_ids
        ]
        # Default: nothing completed
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False
        # Apply overrides
        for k, v in has_overrides.items():
            getattr(db, k).return_value = v
        return db

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_skips_transcribed_media(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """_run_analyze skips asr.transcribe_media when has_transcript is True."""
        from autopilot.orchestrator import _run_analyze

        db = self._make_db_with_media(["m1", "m2"])
        # m1 already transcribed
        db.has_transcript.side_effect = lambda mid: mid == "m1"
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db)

        # Only m2 should get transcribed
        assert mock_asr.transcribe_media.call_count == 1

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_skips_detected_media(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """_run_analyze skips objects.detect_objects when has_detections is True."""
        from autopilot.orchestrator import _run_analyze

        db = self._make_db_with_media(["m1"])
        db.has_detections.return_value = True  # already detected
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db)

        mock_objects.detect_objects.assert_not_called()

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_skips_boundaries_faces_embeddings_audio(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """_run_analyze skips each pass independently when has_* returns True."""
        from autopilot.orchestrator import _run_analyze

        db = self._make_db_with_media(["m1"])
        db.has_boundaries.return_value = True
        db.has_faces.return_value = True
        db.has_embeddings.return_value = True
        db.has_audio_events.return_value = True
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db)

        mock_scenes.detect_shots.assert_not_called()
        mock_faces.detect_faces.assert_not_called()
        mock_embeddings.compute_embeddings.assert_not_called()
        mock_audio_events.classify_audio_events.assert_not_called()
        # transcript and detections should still be called (not skipped)
        mock_asr.transcribe_media.assert_called_once()
        mock_objects.detect_objects.assert_called_once()

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_logs_resume_counts(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
        caplog,
    ):
        """_run_analyze logs per-pass resume counts."""
        from autopilot.orchestrator import _run_analyze

        db = self._make_db_with_media(["m1", "m2", "m3"])
        # 2 of 3 already transcribed
        db.has_transcript.side_effect = lambda mid: mid in ("m1", "m2")
        mock_gpu_cls.return_value = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            _run_analyze(config=minimal_config, db=db)

        assert any("transcri" in r.message.lower() and "2/3" in r.message for r in caplog.records)

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_force_reprocesses_all(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """_run_analyze with force=True ignores has_* checks."""
        from autopilot.orchestrator import _run_analyze

        db = self._make_db_with_media(["m1"])
        # Everything already done
        db.has_transcript.return_value = True
        db.has_boundaries.return_value = True
        db.has_detections.return_value = True
        db.has_faces.return_value = True
        db.has_embeddings.return_value = True
        db.has_audio_events.return_value = True
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db, force=True)

        # All passes should run despite has_* returning True
        assert mock_asr.transcribe_media.call_count == 1
        assert mock_scenes.detect_shots.call_count == 1
        assert mock_objects.detect_objects.call_count == 1
        assert mock_faces.detect_faces.call_count == 1
        assert mock_embeddings.compute_embeddings.call_count == 1
        assert mock_audio_events.classify_audio_events.call_count == 1


# -- Checkpoint/resume tests for _run_classify --------------------------------


class TestClassifyResume:
    """Tests for _run_classify checkpoint/resume logic."""

    @patch("autopilot.organize.classify")
    @patch("autopilot.organize.cluster")
    def test_classify_skips_when_clusters_exist(
        self,
        mock_cluster,
        mock_classify,
        minimal_config,
    ):
        """_run_classify skips when activity clusters already exist and have labels."""
        from autopilot.orchestrator import _run_classify

        db = MagicMock()
        db.get_activity_clusters.return_value = [
            {"cluster_id": "ac1", "label": "Beach Day"},
            {"cluster_id": "ac2", "label": "Dinner"},
        ]

        _run_classify(config=minimal_config, db=db)

        mock_cluster.cluster_activities.assert_not_called()
        mock_classify.label_activities.assert_not_called()

    @patch("autopilot.organize.classify")
    @patch("autopilot.organize.cluster")
    def test_classify_force_clears_and_reprocesses(
        self,
        mock_cluster,
        mock_classify,
        minimal_config,
    ):
        """_run_classify with force=True re-runs even when clusters exist."""
        from autopilot.orchestrator import _run_classify

        db = MagicMock()
        db.get_activity_clusters.return_value = [
            {"cluster_id": "ac1", "label": "Beach Day"},
        ]

        _run_classify(config=minimal_config, db=db, force=True)

        mock_cluster.cluster_activities.assert_called_once()
        mock_classify.label_activities.assert_called_once()


# -- Checkpoint/resume tests for _run_script ----------------------------------


class TestScriptResume:
    """Tests for _run_script checkpoint/resume logic."""

    @patch("autopilot.plan.script")
    def test_script_skips_narrative_with_existing_script(
        self,
        mock_script,
        minimal_config,
    ):
        """_run_script skips generate_script when narrative already has a script."""
        from autopilot.orchestrator import _run_script

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        # n1 already has a script, n2 does not
        db.get_narrative_script.side_effect = lambda nid: (
            {"narrative_id": "n1", "script_json": "{}"} if nid == "n1" else None
        )
        mock_script.generate_script.return_value = {"scenes": []}

        _run_script(config=minimal_config, db=db)

        # Only n2 should get generate_script called
        assert mock_script.generate_script.call_count == 1
        mock_script.generate_script.assert_called_once_with("n2", db, minimal_config.llm)

    @patch("autopilot.plan.script")
    def test_script_logs_resume_counts(
        self,
        mock_script,
        minimal_config,
        caplog,
    ):
        """_run_script logs 'Resuming SCRIPT: N/M narratives already scripted'."""
        from autopilot.orchestrator import _run_script

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
            {"narrative_id": "n3"},
        ]
        # n1 and n2 already have scripts
        db.get_narrative_script.side_effect = lambda nid: (
            {"narrative_id": nid, "script_json": "{}"} if nid in ("n1", "n2") else None
        )
        mock_script.generate_script.return_value = {"scenes": []}

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            _run_script(config=minimal_config, db=db)

        assert any("Resuming SCRIPT" in r.message and "2/3" in r.message for r in caplog.records)

    @patch("autopilot.plan.script")
    def test_script_force_regenerates_all(
        self,
        mock_script,
        minimal_config,
    ):
        """_run_script with force=True regenerates even narratives with existing scripts."""
        from autopilot.orchestrator import _run_script

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        # Both already have scripts
        db.get_narrative_script.return_value = {"narrative_id": "n1", "script_json": "{}"}
        mock_script.generate_script.return_value = {"scenes": []}

        _run_script(config=minimal_config, db=db, force=True)

        # Both should get generate_script called
        assert mock_script.generate_script.call_count == 2


# -- Checkpoint/resume tests for _run_edl ------------------------------------


class TestEdlResume:
    """Tests for _run_edl checkpoint/resume logic."""

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_skips_narrative_with_existing_edit_plan(
        self,
        mock_edl,
        mock_validator,
        mock_otio,
        minimal_config,
    ):
        """_run_edl skips EDL generation when narrative already has an edit plan with edl_json."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_narrative_script.return_value = {"scenes": []}
        # n1 already has an edit plan with edl_json, n2 does not
        db.get_edit_plan.side_effect = lambda nid: (
            {"narrative_id": "n1", "edl_json": '{"timeline": []}'} if nid == "n1" else None
        )
        mock_edl.generate_edl.return_value = {"timeline": []}
        mock_validator.validate_edl.return_value = MagicMock(passed=True)

        _run_edl(config=minimal_config, db=db)

        # Only n2 should get generate_edl called
        assert mock_edl.generate_edl.call_count == 1

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_logs_resume_counts(
        self,
        mock_edl,
        mock_validator,
        mock_otio,
        minimal_config,
        caplog,
    ):
        """_run_edl logs 'Resuming EDL: N/M narratives already have edit plans'."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
            {"narrative_id": "n3"},
        ]
        db.get_narrative_script.return_value = {"scenes": []}
        # n1 and n2 already have edit plans
        db.get_edit_plan.side_effect = lambda nid: (
            {"narrative_id": nid, "edl_json": '{"timeline": []}'} if nid in ("n1", "n2") else None
        )
        mock_edl.generate_edl.return_value = {"timeline": []}
        mock_validator.validate_edl.return_value = MagicMock(passed=True)

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            _run_edl(config=minimal_config, db=db)

        assert any("Resuming EDL" in r.message and "2/3" in r.message for r in caplog.records)

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_force_regenerates_all(
        self,
        mock_edl,
        mock_validator,
        mock_otio,
        minimal_config,
    ):
        """_run_edl with force=True regenerates even narratives with existing edit plans."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_narrative_script.return_value = {"scenes": []}
        # Both already have edit plans
        db.get_edit_plan.return_value = {"narrative_id": "n1", "edl_json": '{"timeline": []}'}
        mock_edl.generate_edl.return_value = {"timeline": []}
        mock_validator.validate_edl.return_value = MagicMock(passed=True)

        _run_edl(config=minimal_config, db=db, force=True)

        # Both should get generate_edl called
        assert mock_edl.generate_edl.call_count == 2


# -- Checkpoint/resume tests for _run_source_assets ---------------------------


class TestSourceResume:
    """Tests for _run_source_assets checkpoint/resume logic."""

    @patch("autopilot.source.resolve")
    def test_source_skips_narrative_with_resolved_assets(
        self,
        mock_resolve,
        minimal_config,
        tmp_path,
    ):
        """_run_source_assets skips when asset_dir exists and is non-empty."""
        from autopilot.orchestrator import _run_source_assets

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }

        # Create non-empty asset dir for n1
        asset_dir_n1 = minimal_config.output_dir / "assets" / "n1"
        asset_dir_n1.mkdir(parents=True)
        (asset_dir_n1 / "clip.mp4").touch()

        mock_resolve.resolve_edl_assets.return_value = {"edl": {}, "unresolved": []}

        _run_source_assets(config=minimal_config, db=db)

        # Only n2 should get resolve_edl_assets called
        assert mock_resolve.resolve_edl_assets.call_count == 1

    @patch("autopilot.source.resolve")
    def test_source_logs_resume_counts(
        self,
        mock_resolve,
        minimal_config,
        caplog,
    ):
        """_run_source_assets logs 'Resuming SOURCE_ASSETS: N/M ...'."""
        from autopilot.orchestrator import _run_source_assets

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
            {"narrative_id": "n3"},
        ]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }

        # Create non-empty asset dirs for n1 and n2
        for nid in ("n1", "n2"):
            asset_dir = minimal_config.output_dir / "assets" / nid
            asset_dir.mkdir(parents=True)
            (asset_dir / "clip.mp4").touch()

        mock_resolve.resolve_edl_assets.return_value = {"edl": {}, "unresolved": []}

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            _run_source_assets(config=minimal_config, db=db)

        assert any(
            "Resuming SOURCE_ASSETS" in r.message and "2/3" in r.message for r in caplog.records
        )

    @patch("autopilot.source.resolve")
    def test_source_force_re_resolves_all(
        self,
        mock_resolve,
        minimal_config,
    ):
        """_run_source_assets with force=True re-resolves even with existing assets."""
        from autopilot.orchestrator import _run_source_assets

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
        }

        # Create non-empty asset dir for both
        for nid in ("n1", "n2"):
            asset_dir = minimal_config.output_dir / "assets" / nid
            asset_dir.mkdir(parents=True)
            (asset_dir / "clip.mp4").touch()

        mock_resolve.resolve_edl_assets.return_value = {"edl": {}, "unresolved": []}

        _run_source_assets(config=minimal_config, db=db, force=True)

        # Both should get resolve_edl_assets called
        assert mock_resolve.resolve_edl_assets.call_count == 2


# -- Checkpoint/resume tests for _run_render ----------------------------------


class TestRenderResume:
    """Tests for _run_render checkpoint/resume logic."""

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_skips_narrative_with_existing_render_path(
        self,
        mock_router,
        mock_validate,
        minimal_config,
    ):
        """_run_render skips when edit_plan has render_path and file exists on disk."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]

        # Create render file on disk for n1
        render_path_n1 = minimal_config.output_dir / "renders" / "n1" / "output.mp4"
        render_path_n1.parent.mkdir(parents=True)
        render_path_n1.touch()

        # n1 has render_path in edit plan pointing to existing file, n2 does not
        def get_edit_plan(nid):
            if nid == "n1":
                return {
                    "narrative_id": "n1",
                    "edl_json": '{"timeline": []}',
                    "render_path": str(render_path_n1),
                }
            return {
                "narrative_id": "n2",
                "edl_json": '{"timeline": []}',
            }

        db.get_edit_plan.side_effect = get_edit_plan

        mock_router.route_and_render.return_value = Path("/out/video.mp4")
        mock_validate.validate_render.return_value = MagicMock(passed=True, issues=[])

        _run_render(config=minimal_config, db=db)

        # Only n2 should get route_and_render called
        assert mock_router.route_and_render.call_count == 1

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_logs_resume_counts(
        self,
        mock_router,
        mock_validate,
        minimal_config,
        caplog,
    ):
        """_run_render logs 'Resuming RENDER: N/M ...'."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
            {"narrative_id": "n3"},
        ]

        # Create render files for n1 and n2
        for nid in ("n1", "n2"):
            rpath = minimal_config.output_dir / "renders" / nid / "output.mp4"
            rpath.parent.mkdir(parents=True)
            rpath.touch()

        def get_edit_plan(nid):
            if nid in ("n1", "n2"):
                return {
                    "narrative_id": nid,
                    "edl_json": '{"timeline": []}',
                    "render_path": str(minimal_config.output_dir / "renders" / nid / "output.mp4"),
                }
            return {
                "narrative_id": nid,
                "edl_json": '{"timeline": []}',
            }

        db.get_edit_plan.side_effect = get_edit_plan

        mock_router.route_and_render.return_value = Path("/out/video.mp4")
        mock_validate.validate_render.return_value = MagicMock(passed=True, issues=[])

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            _run_render(config=minimal_config, db=db)

        assert any("Resuming RENDER" in r.message and "2/3" in r.message for r in caplog.records)

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_force_re_renders_all(
        self,
        mock_router,
        mock_validate,
        minimal_config,
    ):
        """_run_render with force=True re-renders even with existing render output."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]

        # Create render files for both
        for nid in ("n1", "n2"):
            rpath = minimal_config.output_dir / "renders" / nid / "output.mp4"
            rpath.parent.mkdir(parents=True)
            rpath.touch()

        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{"timeline": []}',
            "render_path": str(minimal_config.output_dir / "renders" / "n1" / "output.mp4"),
        }

        mock_router.route_and_render.return_value = Path("/out/video.mp4")
        mock_validate.validate_render.return_value = MagicMock(passed=True, issues=[])

        _run_render(config=minimal_config, db=db, force=True)

        # Both should get route_and_render called
        assert mock_router.route_and_render.call_count == 2


# -- Checkpoint/resume tests for _run_upload ----------------------------------


class TestUploadResume:
    """Tests for _run_upload checkpoint/resume logic."""

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_skips_narrative_with_existing_upload(
        self,
        mock_youtube,
        mock_thumbnail,
        minimal_config,
    ):
        """_run_upload skips when db.get_upload returns non-None for a narrative."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "edl_json": "{}",
            "render_path": "/out/renders/output.mp4",
        }
        # n1 already uploaded, n2 not
        db.get_upload.side_effect = lambda nid: (
            {"narrative_id": "n1", "video_id": "abc"} if nid == "n1" else None
        )

        mock_youtube.upload_video.return_value = "https://youtu.be/def"
        mock_thumbnail.extract_best_thumbnail.return_value = Path("/thumb.jpg")

        _run_upload(config=minimal_config, db=db)

        # Only n2 should get upload_video called
        assert mock_youtube.upload_video.call_count == 1

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_logs_resume_counts(
        self,
        mock_youtube,
        mock_thumbnail,
        minimal_config,
        caplog,
    ):
        """_run_upload logs 'Resuming UPLOAD: N/M ...'."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
            {"narrative_id": "n3"},
        ]
        db.get_edit_plan.return_value = {
            "edl_json": "{}",
            "render_path": "/out/renders/output.mp4",
        }
        # n1 and n2 already uploaded
        db.get_upload.side_effect = lambda nid: (
            {"narrative_id": nid, "video_id": "abc"} if nid in ("n1", "n2") else None
        )

        mock_youtube.upload_video.return_value = "https://youtu.be/def"
        mock_thumbnail.extract_best_thumbnail.return_value = Path("/thumb.jpg")

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            _run_upload(config=minimal_config, db=db)

        assert any("Resuming UPLOAD" in r.message and "2/3" in r.message for r in caplog.records)

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_force_re_uploads_all(
        self,
        mock_youtube,
        mock_thumbnail,
        minimal_config,
    ):
        """_run_upload with force=True re-uploads even with existing uploads."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "edl_json": "{}",
            "render_path": "/out/renders/output.mp4",
        }
        # Both already uploaded
        db.get_upload.return_value = {"narrative_id": "n1", "video_id": "abc"}

        mock_youtube.upload_video.return_value = "https://youtu.be/def"
        mock_thumbnail.extract_best_thumbnail.return_value = Path("/thumb.jpg")

        _run_upload(config=minimal_config, db=db, force=True)

        # Both should get upload_video called
        assert mock_youtube.upload_video.call_count == 2


class TestForceFlagPropagation:
    """Tests for force flag propagation through PipelineOrchestrator."""

    def test_orchestrator_init_accepts_force(self) -> None:
        """PipelineOrchestrator.__init__ accepts a force parameter."""
        orch = PipelineOrchestrator(force=True)
        assert orch.force is True

    def test_orchestrator_init_defaults_force_false(self) -> None:
        """PipelineOrchestrator defaults force to False."""
        orch = PipelineOrchestrator()
        assert orch.force is False

    def test_orchestrator_run_passes_force_true_to_stages(self) -> None:
        """run() passes force=True to each stage when orchestrator has force=True."""
        orch = PipelineOrchestrator(force=True)
        mocks: dict[str, MagicMock] = {}
        for stage in orch.stages:
            mock_fn = MagicMock()
            mocks[stage.name] = mock_fn
            stage.func = mock_fn

        mock_config = MagicMock()
        mock_db = MagicMock()
        orch.run(config=mock_config, db=mock_db)

        for name, mock_fn in mocks.items():
            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["config"] is mock_config
            assert call_kwargs["db"] is mock_db
            assert call_kwargs["force"] is True

    def test_orchestrator_run_passes_force_false_to_stages(self) -> None:
        """run() passes force=False to each stage when orchestrator has force=False."""
        orch = PipelineOrchestrator(force=False)
        mocks: dict[str, MagicMock] = {}
        for stage in orch.stages:
            mock_fn = MagicMock()
            mocks[stage.name] = mock_fn
            stage.func = mock_fn

        mock_config = MagicMock()
        mock_db = MagicMock()
        orch.run(config=mock_config, db=mock_db)

        for name, mock_fn in mocks.items():
            mock_fn.assert_called_once()
            call_kwargs = mock_fn.call_args[1]
            assert call_kwargs["config"] is mock_config
            assert call_kwargs["db"] is mock_db
            assert call_kwargs["force"] is False


# --- Run Tracking & Event Emission tests (Task 37) ---


class TestRunTrackingInit:
    """Tests for pipeline run record creation at start of run()."""

    def test_run_creates_pipeline_run_record(self) -> None:
        """run() calls db.insert_run once with correct args."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        mock_db.insert_run.assert_called_once()
        call_kwargs = mock_db.insert_run.call_args
        run_id = call_kwargs[0][0]  # first positional arg
        assert isinstance(run_id, str)
        assert len(run_id) == 32
        assert re.fullmatch(r"[0-9a-f]{32}", run_id), f"run_id not hex: {run_id}"

        kw = call_kwargs[1]
        assert kw["status"] == "running"
        # started_at should be an ISO 8601 string
        assert isinstance(kw["started_at"], str)
        assert "T" in kw["started_at"]
        # config_snapshot should be a string
        assert isinstance(kw["config_snapshot"], str)

    def test_run_stores_run_id_on_self(self) -> None:
        """After run(), orch._run_id is a 32-char hex string."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        orch.run(config=MagicMock(), db=MagicMock())

        assert hasattr(orch, "_run_id")
        assert isinstance(orch._run_id, str)
        assert len(orch._run_id) == 32
        assert re.fullmatch(r"[0-9a-f]{32}", orch._run_id)

    def test_run_passes_budget_to_insert_run(self) -> None:
        """When budget_seconds=3600, insert_run is called with budget_remaining_seconds=3600."""
        orch = PipelineOrchestrator(budget_seconds=3600)
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        kw = mock_db.insert_run.call_args[1]
        assert kw["budget_remaining_seconds"] == 3600


class TestEmitEvent:
    """Tests for the _emit_event helper method."""

    def test_emit_event_calls_insert_event(self) -> None:
        """_emit_event calls db.insert_event with event_type and stage."""
        orch = PipelineOrchestrator()
        orch._db = MagicMock()
        orch._run_id = "a" * 32

        orch._emit_event("test_event", stage="INGEST")

        orch._db.insert_event.assert_called_once()
        call_kwargs = orch._db.insert_event.call_args[1]
        assert call_kwargs["event_type"] == "test_event"
        assert call_kwargs["stage"] == "INGEST"

    def test_emit_event_serializes_payload_to_json(self) -> None:
        """_emit_event serializes payload dict to JSON string."""
        orch = PipelineOrchestrator()
        orch._db = MagicMock()
        orch._run_id = "b" * 32

        orch._emit_event("test_event", payload={"key": "val"})

        call_kwargs = orch._db.insert_event.call_args[1]
        assert call_kwargs["payload_json"] == '{"key": "val"}'

    def test_emit_event_none_payload(self) -> None:
        """_emit_event passes payload_json=None when no payload given."""
        orch = PipelineOrchestrator()
        orch._db = MagicMock()
        orch._run_id = "c" * 32

        orch._emit_event("test_event")

        call_kwargs = orch._db.insert_event.call_args[1]
        assert call_kwargs["payload_json"] is None

    def test_emit_event_logs_at_debug_level(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """_emit_event emits a debug-level log."""
        orch = PipelineOrchestrator()
        orch._db = MagicMock()
        orch._run_id = "d" * 32

        with caplog.at_level(logging.DEBUG, logger="autopilot.orchestrator"):
            orch._emit_event("test_event", stage="INGEST")

        debug_records = [
            r for r in caplog.records if r.levelno == logging.DEBUG and "test_event" in r.message
        ]
        assert len(debug_records) >= 1


class TestStageTransitionEvents:
    """Tests for stage_started / stage_completed / stage_error events emitted during run()."""

    def test_stage_started_event_emitted_for_each_stage(self) -> None:
        """run() emits a 'stage_started' event for each of the 9 stages."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        insert_event_calls = mock_db.insert_event.call_args_list
        started_calls = [c for c in insert_event_calls if c[1].get("event_type") == "stage_started"]
        started_stages = [c[1]["stage"] for c in started_calls]
        assert started_stages == EXPECTED_STAGES

    def test_stage_completed_event_emitted_with_elapsed(self) -> None:
        """run() emits 'stage_completed' events with 'elapsed' in payload."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        insert_event_calls = mock_db.insert_event.call_args_list
        completed_calls = [
            c for c in insert_event_calls if c[1].get("event_type") == "stage_completed"
        ]
        assert len(completed_calls) == 9
        for c in completed_calls:
            payload = json.loads(c[1]["payload_json"])
            assert "elapsed" in payload
            assert isinstance(payload["elapsed"], float)

    def test_stage_error_event_emitted(self) -> None:
        """When a stage raises, a 'stage_error' event is emitted with error message."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            if stage.name == "INGEST":
                stage.func = MagicMock(side_effect=RuntimeError("ingest boom"))
            else:
                stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        insert_event_calls = mock_db.insert_event.call_args_list
        error_calls = [c for c in insert_event_calls if c[1].get("event_type") == "stage_error"]
        assert len(error_calls) >= 1
        payload = json.loads(error_calls[0][1]["payload_json"])
        assert "error" in payload
        assert "ingest boom" in payload["error"]
        assert error_calls[0][1]["stage"] == "INGEST"

    def test_current_stage_updated_via_update_run(self) -> None:
        """run() calls db.update_run with current_stage before each stage runs."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        update_run_calls = mock_db.update_run.call_args_list
        # Extract calls that set current_stage
        current_stage_calls = [c for c in update_run_calls if "current_stage" in c[1]]
        stages_set = [c[1]["current_stage"] for c in current_stage_calls]
        assert stages_set == EXPECTED_STAGES


class TestRunFinalization:
    """Tests for run finalization — status update and run_completed/run_failed events."""

    def test_run_completed_status_on_success(self) -> None:
        """When all stages pass, final update_run has status='completed' and wall_clock_seconds."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        update_calls = mock_db.update_run.call_args_list
        # The last update_run call should finalize the run
        final_calls = [
            c
            for c in update_calls
            if "status" in c[1] and c[1]["status"] in ("completed", "failed")
        ]
        assert len(final_calls) >= 1
        final = final_calls[-1][1]
        assert final["status"] == "completed"
        assert "finished_at" in final
        assert isinstance(final["finished_at"], str)
        assert "T" in final["finished_at"]
        assert "wall_clock_seconds" in final
        assert isinstance(final["wall_clock_seconds"], float)
        assert final["wall_clock_seconds"] >= 0

    def test_run_failed_status_on_error(self) -> None:
        """When a stage errors, final update_run has status='failed'."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            if stage.name == "INGEST":
                stage.func = MagicMock(side_effect=RuntimeError("boom"))
            else:
                stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        update_calls = mock_db.update_run.call_args_list
        final_calls = [
            c
            for c in update_calls
            if "status" in c[1] and c[1]["status"] in ("completed", "failed")
        ]
        assert len(final_calls) >= 1
        assert final_calls[-1][1]["status"] == "failed"

    def test_run_completed_event_emitted(self) -> None:
        """'run_completed' event emitted on successful run with duration in payload."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        insert_event_calls = mock_db.insert_event.call_args_list
        completed_calls = [
            c for c in insert_event_calls if c[1].get("event_type") == "run_completed"
        ]
        assert len(completed_calls) == 1
        payload = json.loads(completed_calls[0][1]["payload_json"])
        assert "duration" in payload
        assert isinstance(payload["duration"], float)

    def test_run_failed_event_emitted(self) -> None:
        """'run_failed' event emitted when pipeline has errors."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            if stage.name == "ANALYZE":
                stage.func = MagicMock(side_effect=ValueError("analyze fail"))
            else:
                stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        insert_event_calls = mock_db.insert_event.call_args_list
        failed_calls = [c for c in insert_event_calls if c[1].get("event_type") == "run_failed"]
        assert len(failed_calls) == 1
        payload = json.loads(failed_calls[0][1]["payload_json"])
        assert "duration" in payload


class TestBudgetRemainingTracking:
    """Tests for budget_remaining_seconds updates after each stage."""

    def test_budget_remaining_written_after_each_stage(self) -> None:
        """With budget_seconds=3600, update_run includes budget_remaining."""
        orch = PipelineOrchestrator(budget_seconds=3600)
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        update_calls = mock_db.update_run.call_args_list
        budget_calls = [c for c in update_calls if "budget_remaining_seconds" in c[1]]
        # Should have one budget update per stage (9 stages)
        assert len(budget_calls) == 9
        # Each budget_remaining value should be a float
        for c in budget_calls:
            remaining = c[1]["budget_remaining_seconds"]
            assert isinstance(remaining, float)
            # Budget remaining should be <= 3600 (decreasing)
            assert remaining <= 3600

    def test_no_budget_remaining_when_no_budget(self) -> None:
        """With budget_seconds=None, no budget_remaining updates."""
        orch = PipelineOrchestrator(budget_seconds=None)
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        update_calls = mock_db.update_run.call_args_list
        budget_calls = [c for c in update_calls if "budget_remaining_seconds" in c[1]]
        # No per-stage budget updates should occur
        assert len(budget_calls) == 0


class TestRunTrackingIntegration:
    """Integration tests using real in-memory CatalogDB."""

    def test_full_run_creates_db_records(self, catalog_db) -> None:
        """Run with real CatalogDB creates a pipeline_runs row with status='completed'."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        orch.run(config=MagicMock(), db=catalog_db)

        run_record = catalog_db.get_run(orch._run_id)
        assert run_record is not None
        assert run_record["status"] == "completed"
        assert run_record["wall_clock_seconds"] is not None
        assert float(run_record["wall_clock_seconds"]) >= 0
        assert run_record["finished_at"] is not None

    def test_full_run_creates_events(self, catalog_db) -> None:
        """Run creates stage_started + stage_completed events for each stage plus run_completed."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        orch.run(config=MagicMock(), db=catalog_db)

        events = catalog_db.get_events_since(0)
        event_types = [e["event_type"] for e in events]

        # 9 stages × 2 (started + completed) = 18, plus 1 run_completed = 19
        assert event_types.count("stage_started") == 9
        assert event_types.count("stage_completed") == 9
        assert event_types.count("run_completed") == 1

        # Verify each stage has both started and completed events
        for stage_name in EXPECTED_STAGES:
            stage_started = [
                e for e in events if e["event_type"] == "stage_started" and e["stage"] == stage_name
            ]
            stage_completed = [
                e
                for e in events
                if e["event_type"] == "stage_completed" and e["stage"] == stage_name
            ]
            assert len(stage_started) == 1, f"Missing stage_started for {stage_name}"
            assert len(stage_completed) == 1, f"Missing stage_completed for {stage_name}"

    def test_dry_run_still_creates_run_record(self, catalog_db) -> None:
        """dry_run=True still creates a pipeline_runs record."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        orch.run(config=MagicMock(), db=catalog_db, dry_run=True)

        run_record = catalog_db.get_run(orch._run_id)
        assert run_record is not None
        assert run_record["status"] is not None


class TestEmitEventResilience:
    """Tests that _emit_event swallows DB errors and never propagates them."""

    def test_emit_event_swallows_db_error(self) -> None:
        """_emit_event does not raise when db.insert_event raises RuntimeError."""
        orch = PipelineOrchestrator()
        orch._db = MagicMock()
        orch._db.insert_event.side_effect = RuntimeError("DB locked")
        orch._run_id = "a" * 32

        # Should NOT raise
        orch._emit_event("test_event", stage="X")

    def test_emit_event_logs_warning_on_db_error(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When insert_event raises, _emit_event logs a WARNING with event_type and error."""
        orch = PipelineOrchestrator()
        orch._db = MagicMock()
        orch._db.insert_event.side_effect = RuntimeError("connection lost")
        orch._run_id = "b" * 32

        with caplog.at_level(logging.WARNING, logger="autopilot.orchestrator"):
            orch._emit_event("test_event", stage="Y")

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING
            and "test_event" in r.message
            and "connection lost" in r.message
        ]
        assert len(warning_records) >= 1

    def test_emit_event_still_works_after_error(self) -> None:
        """After a failed _emit_event, a subsequent call with a working db succeeds."""
        orch = PipelineOrchestrator()
        orch._db = MagicMock()
        orch._run_id = "c" * 32

        # First call: DB raises
        orch._db.insert_event.side_effect = RuntimeError("transient")
        orch._emit_event("failing_event", stage="A")

        # Second call: DB works fine
        orch._db.insert_event.side_effect = None
        orch._emit_event("ok_event", stage="B")

        # Verify second call went through
        last_call = orch._db.insert_event.call_args
        assert last_call[1]["event_type"] == "ok_event"


class TestInsertRunResilience:
    """Tests that db.insert_run failure doesn't block the pipeline."""

    def test_insert_run_failure_does_not_block_pipeline(self) -> None:
        """When db.insert_run raises, run() still executes all stages and returns results."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        mock_db.insert_run.side_effect = RuntimeError("locked")

        results = orch.run(config=MagicMock(), db=mock_db)

        # All 9 stages should have executed
        assert len(results) == 9
        for stage_name, result in results.items():
            assert result.status == StageStatus.DONE

    def test_insert_run_failure_sets_run_id_none(self) -> None:
        """When insert_run raises, self._run_id is None."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        mock_db.insert_run.side_effect = RuntimeError("locked")

        orch.run(config=MagicMock(), db=mock_db)
        assert orch._run_id is None

    def test_insert_run_failure_skips_tracking_calls(self) -> None:
        """When insert_run raises, update_run and insert_event are never called."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        mock_db.insert_run.side_effect = RuntimeError("locked")

        orch.run(config=MagicMock(), db=mock_db)

        mock_db.update_run.assert_not_called()
        mock_db.insert_event.assert_not_called()

    def test_insert_run_failure_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When insert_run raises, a WARNING log mentioning 'run tracking' is emitted."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        mock_db.insert_run.side_effect = RuntimeError("locked")

        with caplog.at_level(logging.WARNING, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=mock_db)

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "run tracking" in r.message.lower()
        ]
        assert len(warning_records) >= 1


class TestTrackingCallResilience:
    """Tests that transient db.update_run failures don't abort the pipeline."""

    def test_update_run_failure_in_stage_loop_does_not_abort(self) -> None:
        """When db.update_run raises on current_stage update, all stages still execute."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        # update_run raises on the first call (current_stage update)
        mock_db.update_run.side_effect = RuntimeError("DB error")

        results = orch.run(config=MagicMock(), db=mock_db)

        # All 9 stages should have run
        assert len(results) == 9
        for result in results.values():
            assert result.status == StageStatus.DONE

    def test_update_run_failure_for_budget_does_not_abort(self) -> None:
        """When db.update_run raises on budget_remaining update, pipeline completes."""
        orch = PipelineOrchestrator(budget_seconds=3600)
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()

        # Only raise when budget_remaining_seconds kwarg is present
        def selective_raise(*args: Any, **kwargs: Any) -> None:
            if "budget_remaining_seconds" in kwargs:
                raise RuntimeError("DB error on budget update")

        mock_db.update_run.side_effect = selective_raise

        results = orch.run(config=MagicMock(), db=mock_db)

        assert len(results) == 9
        for result in results.values():
            assert result.status == StageStatus.DONE

    def test_finalization_update_run_failure_returns_results(self) -> None:
        """When db.update_run raises during finalization, run() still returns results."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()

        # Only raise when status='completed' kwarg is present (finalization)
        def selective_raise(*args: Any, **kwargs: Any) -> None:
            if "status" in kwargs and kwargs["status"] == "completed":
                raise RuntimeError("DB error on finalize")

        mock_db.update_run.side_effect = selective_raise

        results = orch.run(config=MagicMock(), db=mock_db)

        # Should still return all 9 results
        assert len(results) == 9
        for result in results.values():
            assert result.status == StageStatus.DONE

    def test_update_run_failure_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """WARNING log emitted for each failed db.update_run call."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        mock_db = MagicMock()
        mock_db.update_run.side_effect = RuntimeError("DB error")

        with caplog.at_level(logging.WARNING, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=mock_db)

        warning_records = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "tracking" in r.message.lower()
        ]
        assert len(warning_records) >= 1


class TestStartJob:
    """Tests for the module-level _start_job helper."""

    def test_start_job_returns_job_id_and_start_mono(self) -> None:
        """_start_job returns (job_id, start_mono) tuple with 32-char hex job_id."""
        from autopilot.orchestrator import _start_job

        mock_db = MagicMock()
        job_id, start_mono = _start_job(mock_db, "INGEST", "ingest_file")
        assert isinstance(job_id, str)
        assert len(job_id) == 32
        assert all(c in "0123456789abcdef" for c in job_id)
        assert isinstance(start_mono, float)

    def test_start_job_calls_insert_job_with_running_status(self) -> None:
        """_start_job calls db.insert_job with status='running' and correct fields."""
        from autopilot.orchestrator import _start_job

        mock_db = MagicMock()
        job_id, _ = _start_job(
            mock_db,
            "INGEST",
            "ingest_file",
            target_id="media_123",
            target_label="video.mp4",
            worker="cpu",
            run_id="abc123",
        )
        mock_db.insert_job.assert_called_once()
        call_kwargs = mock_db.insert_job.call_args
        assert call_kwargs[0][0] == job_id  # positional: job_id
        assert call_kwargs[0][1] == "INGEST"  # positional: stage
        assert call_kwargs[0][2] == "ingest_file"  # positional: job_type
        assert call_kwargs[1]["status"] == "running"
        assert call_kwargs[1]["target_id"] == "media_123"
        assert call_kwargs[1]["target_label"] == "video.mp4"
        assert call_kwargs[1]["worker"] == "cpu"
        assert call_kwargs[1]["run_id"] == "abc123"
        assert "started_at" in call_kwargs[1]

    def test_start_job_started_at_is_utc_iso(self) -> None:
        """_start_job sets started_at as a UTC ISO string."""
        from autopilot.orchestrator import _start_job

        mock_db = MagicMock()
        _start_job(mock_db, "ANALYZE", "asr")
        started_at = mock_db.insert_job.call_args[1]["started_at"]
        # Should be parseable as ISO format and contain timezone info
        assert "T" in started_at
        assert "+" in started_at or "Z" in started_at


class TestFinishJob:
    """Tests for the module-level _finish_job helper."""

    def test_finish_job_updates_with_done_status(self) -> None:
        """_finish_job calls db.update_job with status='done' by default."""
        import time

        from autopilot.orchestrator import _finish_job

        mock_db = MagicMock()
        start_mono = time.monotonic() - 1.5  # simulate 1.5s elapsed
        _finish_job(mock_db, "job_abc", start_mono)
        mock_db.update_job.assert_called_once()
        call_kwargs = mock_db.update_job.call_args[1]
        assert call_kwargs["status"] == "done"
        assert "finished_at" in call_kwargs
        assert isinstance(call_kwargs["duration_seconds"], float)
        assert call_kwargs["duration_seconds"] >= 1.0

    def test_finish_job_error_status_and_message(self) -> None:
        """_finish_job with status='error' passes error_message."""
        import time

        from autopilot.orchestrator import _finish_job

        mock_db = MagicMock()
        start_mono = time.monotonic()
        _finish_job(
            mock_db,
            "job_xyz",
            start_mono,
            status="error",
            error_message="disk full",
        )
        call_kwargs = mock_db.update_job.call_args[1]
        assert call_kwargs["status"] == "error"
        assert call_kwargs["error_message"] == "disk full"

    def test_finish_job_calculates_duration_from_monotonic(self) -> None:
        """_finish_job duration_seconds is calculated from monotonic clock."""
        import time

        from autopilot.orchestrator import _finish_job

        mock_db = MagicMock()
        start_mono = time.monotonic() - 2.0
        _finish_job(mock_db, "job_dur", start_mono)
        duration = mock_db.update_job.call_args[1]["duration_seconds"]
        assert duration >= 2.0


class TestTrackJobContextManager:
    """Tests for the _track_job context manager."""

    def test_track_job_yields_job_id(self) -> None:
        """_track_job yields a 32-char hex job_id."""
        from autopilot.orchestrator import _track_job

        mock_db = MagicMock()
        with _track_job(mock_db, "INGEST", "ingest_file") as job_id:
            assert isinstance(job_id, str)
            assert len(job_id) == 32

    def test_track_job_calls_finish_with_done_on_success(self) -> None:
        """On clean exit, _track_job calls db.update_job with status='done'."""
        from autopilot.orchestrator import _track_job

        mock_db = MagicMock()
        with _track_job(mock_db, "INGEST", "ingest_file"):
            pass  # no error

        mock_db.update_job.assert_called_once()
        call_kwargs = mock_db.update_job.call_args[1]
        assert call_kwargs["status"] == "done"

    def test_track_job_calls_finish_with_error_on_exception(self) -> None:
        """On exception, _track_job calls db.update_job with status='error' and re-raises."""
        from autopilot.orchestrator import _track_job

        mock_db = MagicMock()
        with pytest.raises(ValueError, match="test error"):
            with _track_job(mock_db, "ANALYZE", "asr") as _job_id:
                raise ValueError("test error")

        call_kwargs = mock_db.update_job.call_args[1]
        assert call_kwargs["status"] == "error"
        assert call_kwargs["error_message"] == "test error"

    def test_track_job_resilient_to_db_errors(self) -> None:
        """When db fails, _track_job doesn't mask the original exception."""
        from autopilot.orchestrator import _track_job

        mock_db = MagicMock()
        mock_db.insert_job.side_effect = RuntimeError("db connection lost")
        mock_db.update_job.side_effect = RuntimeError("db connection lost")

        # Should not raise a db error, the original ValueError should propagate
        with pytest.raises(ValueError, match="original"):
            with _track_job(mock_db, "INGEST", "ingest_file"):
                raise ValueError("original")

    def test_track_job_no_error_when_db_fails_on_success(self) -> None:
        """When db fails on success path, _track_job doesn't raise."""
        from autopilot.orchestrator import _track_job

        mock_db = MagicMock()
        mock_db.insert_job.side_effect = RuntimeError("db error")
        mock_db.update_job.side_effect = RuntimeError("db error")

        # Should not raise anything
        with _track_job(mock_db, "INGEST", "ingest_file"):
            pass  # clean exit, but db calls fail silently


class TestJobHelperResilience:
    """Tests for _start_job/_finish_job resilience and emit_fn callbacks."""

    def test_start_job_returns_job_id_when_db_raises(self) -> None:
        """When db.insert_job raises, _start_job still returns a job_id."""
        from autopilot.orchestrator import _start_job

        mock_db = MagicMock()
        mock_db.insert_job.side_effect = RuntimeError("db down")
        job_id, start_mono = _start_job(mock_db, "INGEST", "ingest_file")
        # Should still return valid job_id
        assert len(job_id) == 32
        assert isinstance(start_mono, float)

    def test_start_job_logs_warning_when_db_raises(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """_start_job logs WARNING when db.insert_job fails."""
        from autopilot.orchestrator import _start_job

        mock_db = MagicMock()
        mock_db.insert_job.side_effect = RuntimeError("db error")
        with caplog.at_level(logging.WARNING, logger="autopilot.orchestrator"):
            _start_job(mock_db, "INGEST", "ingest_file")
        assert any("insert job" in r.message.lower() for r in caplog.records)

    def test_finish_job_doesnt_crash_when_db_raises(self) -> None:
        """When db.update_job raises, _finish_job doesn't crash."""
        import time

        from autopilot.orchestrator import _finish_job

        mock_db = MagicMock()
        mock_db.update_job.side_effect = RuntimeError("db error")
        # Should not raise
        _finish_job(mock_db, "job_abc", time.monotonic())

    def test_finish_job_logs_warning_when_db_raises(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """_finish_job logs WARNING when db.update_job fails."""
        import time

        from autopilot.orchestrator import _finish_job

        mock_db = MagicMock()
        mock_db.update_job.side_effect = RuntimeError("db err")
        with caplog.at_level(logging.WARNING, logger="autopilot.orchestrator"):
            _finish_job(mock_db, "job_abc", time.monotonic())
        assert any("update job" in r.message.lower() for r in caplog.records)

    def test_start_job_calls_emit_fn(self) -> None:
        """_start_job calls emit_fn('job_started', stage=..., job_id=...)."""
        from autopilot.orchestrator import _start_job

        mock_db = MagicMock()
        mock_emit = MagicMock()
        job_id, _ = _start_job(
            mock_db,
            "INGEST",
            "ingest_file",
            emit_fn=mock_emit,
        )
        mock_emit.assert_called_once_with(
            "job_started",
            stage="INGEST",
            job_id=job_id,
        )

    def test_finish_job_calls_emit_fn_on_done(self) -> None:
        """_finish_job calls emit_fn('job_completed', ...) on done."""
        import time

        from autopilot.orchestrator import _finish_job

        mock_db = MagicMock()
        mock_emit = MagicMock()
        _finish_job(
            mock_db,
            "job_abc",
            time.monotonic(),
            emit_fn=mock_emit,
            stage="INGEST",
        )
        mock_emit.assert_called_once_with(
            "job_completed",
            stage="INGEST",
            job_id="job_abc",
        )

    def test_finish_job_calls_emit_fn_on_error(self) -> None:
        """_finish_job calls emit_fn('job_error', ...) on error."""
        import time

        from autopilot.orchestrator import _finish_job

        mock_db = MagicMock()
        mock_emit = MagicMock()
        _finish_job(
            mock_db,
            "job_abc",
            time.monotonic(),
            status="error",
            error_message="oops",
            emit_fn=mock_emit,
            stage="ANALYZE",
        )
        mock_emit.assert_called_once_with(
            "job_error",
            stage="ANALYZE",
            job_id="job_abc",
        )


class TestIngestJobTracking:
    """Tests for per-job tracking in _run_ingest."""

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_creates_ingest_file_job_per_file(
        self,
        mock_scanner,
        mock_normalizer,
        mock_dedup,
        minimal_config,
    ):
        """When run_id is provided, creates one 'ingest_file' job per file."""
        from autopilot.orchestrator import _run_ingest

        mock_file1 = MagicMock()
        mock_file1.file_path = Path("/fake/v1.mp4")
        mock_file1.sha256_prefix = "aaa111"
        mock_file2 = MagicMock()
        mock_file2.file_path = Path("/fake/v2.mp4")
        mock_file2.sha256_prefix = "bbb222"
        mock_scanner.scan_directory.return_value = [mock_file1, mock_file2]
        db = MagicMock()
        db.get_media.return_value = None

        _run_ingest(
            config=minimal_config,
            db=db,
            run_id="run_abc",
            emit_fn=MagicMock(),
        )

        # Should have insert_job calls for ingest_file (2 files) + dedup (1)
        ingest_file_calls = [c for c in db.insert_job.call_args_list if c[0][2] == "ingest_file"]
        assert len(ingest_file_calls) == 2
        # Check stage is INGEST
        for call in ingest_file_calls:
            assert call[0][1] == "INGEST"
            assert call[1]["worker"] == "cpu"

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_creates_dedup_job(
        self,
        mock_scanner,
        mock_normalizer,
        mock_dedup,
        minimal_config,
    ):
        """When run_id is provided, creates one 'dedup' job."""
        from autopilot.orchestrator import _run_ingest

        mock_scanner.scan_directory.return_value = []
        db = MagicMock()

        _run_ingest(
            config=minimal_config,
            db=db,
            run_id="run_abc",
            emit_fn=MagicMock(),
        )

        dedup_calls = [c for c in db.insert_job.call_args_list if c[0][2] == "dedup"]
        assert len(dedup_calls) == 1
        assert dedup_calls[0][0][1] == "INGEST"

    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_ingest_no_jobs_when_no_run_id(
        self,
        mock_scanner,
        mock_normalizer,
        mock_dedup,
        minimal_config,
    ):
        """When run_id=None (default), no insert_job calls made."""
        from autopilot.orchestrator import _run_ingest

        mock_file = MagicMock()
        mock_file.file_path = Path("/fake/v1.mp4")
        mock_scanner.scan_directory.return_value = [mock_file]
        db = MagicMock()
        db.get_media.return_value = None

        _run_ingest(config=minimal_config, db=db)

        db.insert_job.assert_not_called()


class TestAnalyzeJobTracking:
    """Tests for per-job tracking in _run_analyze."""

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_creates_job_per_media_per_analysis(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """Creates one job per (media x analysis_type) + face_clustering."""
        from autopilot.orchestrator import _run_analyze

        media1 = {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"}
        db = MagicMock()
        db.list_all_media.return_value = [media1]
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(
            config=minimal_config,
            db=db,
            run_id="run_xyz",
            emit_fn=MagicMock(),
        )

        # 6 analysis types for 1 media + 1 face_clustering = 7 jobs
        assert db.insert_job.call_count == 7

        # Check analysis job types present
        job_types = [c[0][2] for c in db.insert_job.call_args_list]
        for expected in ["asr", "scenes", "objects", "faces", "embeddings", "audio_events"]:
            assert expected in job_types
        assert "face_clustering" in job_types

        # All should have stage='ANALYZE'
        for call in db.insert_job.call_args_list:
            assert call[0][1] == "ANALYZE"

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_gpu_worker_for_analysis_jobs(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """Analysis jobs have worker='gpu', face_clustering has worker='cpu'."""
        from autopilot.orchestrator import _run_analyze

        media1 = {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"}
        db = MagicMock()
        db.list_all_media.return_value = [media1]
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(
            config=minimal_config,
            db=db,
            run_id="run_xyz",
            emit_fn=MagicMock(),
        )

        for call in db.insert_job.call_args_list:
            job_type = call[0][2]
            if job_type == "face_clustering":
                assert call[1]["worker"] == "cpu"
            else:
                assert call[1]["worker"] == "gpu"

    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    def test_analyze_no_jobs_when_no_run_id(
        self,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_gpu_cls,
        minimal_config,
    ):
        """When run_id=None (default), no insert_job calls made."""
        from autopilot.orchestrator import _run_analyze

        db = MagicMock()
        db.list_all_media.return_value = [
            {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"},
        ]
        db.has_transcript.return_value = False
        db.has_boundaries.return_value = False
        db.has_detections.return_value = False
        db.has_faces.return_value = False
        db.has_embeddings.return_value = False
        db.has_audio_events.return_value = False
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db)

        db.insert_job.assert_not_called()


class TestClassifyJobTracking:
    """Tests for per-job tracking in _run_classify."""

    @patch("autopilot.organize.classify")
    @patch("autopilot.organize.cluster")
    def test_classify_creates_two_jobs(
        self,
        mock_cluster,
        mock_classify,
        minimal_config,
    ):
        """Creates 'cluster_activities' and 'label_activities' jobs."""
        from autopilot.orchestrator import _run_classify

        db = MagicMock()
        db.get_activity_clusters.return_value = []

        _run_classify(
            config=minimal_config,
            db=db,
            run_id="run_c",
            emit_fn=MagicMock(),
        )

        assert db.insert_job.call_count == 2
        job_types = [c[0][2] for c in db.insert_job.call_args_list]
        assert "cluster_activities" in job_types
        assert "label_activities" in job_types
        for call in db.insert_job.call_args_list:
            assert call[0][1] == "CLASSIFY"
            assert call[1]["worker"] == "cpu"

    @patch("autopilot.organize.classify")
    @patch("autopilot.organize.cluster")
    def test_classify_no_jobs_when_no_run_id(
        self,
        mock_cluster,
        mock_classify,
        minimal_config,
    ):
        """When run_id=None, no insert_job calls."""
        from autopilot.orchestrator import _run_classify

        db = MagicMock()
        db.get_activity_clusters.return_value = []

        _run_classify(config=minimal_config, db=db)

        db.insert_job.assert_not_called()


class TestNarrateJobTracking:
    """Tests for per-job tracking in _run_narrate."""

    @patch("autopilot.organize.narratives")
    def test_narrate_creates_two_jobs(
        self,
        mock_narratives,
        minimal_config,
    ):
        """Creates 'build_storyboard' and 'propose_narratives' jobs."""
        from autopilot.orchestrator import _run_narrate

        mock_narratives.build_master_storyboard.return_value = MagicMock()
        mock_narratives.propose_narratives.return_value = []
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()
        db.list_narratives.return_value = []  # no checkpoint hit

        _run_narrate(
            config=minimal_config,
            db=db,
            run_id="run_n",
            emit_fn=MagicMock(),
        )

        assert db.insert_job.call_count == 2
        job_types = [c[0][2] for c in db.insert_job.call_args_list]
        assert "build_storyboard" in job_types
        assert "propose_narratives" in job_types
        for call in db.insert_job.call_args_list:
            assert call[0][1] == "NARRATE"
            assert call[1]["worker"] == "cpu"

    @patch("autopilot.organize.narratives")
    def test_narrate_no_jobs_when_no_run_id(
        self,
        mock_narratives,
        minimal_config,
    ):
        """When run_id=None, no insert_job calls."""
        from autopilot.orchestrator import _run_narrate

        mock_narratives.build_master_storyboard.return_value = MagicMock()
        mock_narratives.propose_narratives.return_value = []
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()
        db.list_narratives.return_value = []  # no checkpoint hit

        _run_narrate(config=minimal_config, db=db)

        db.insert_job.assert_not_called()


class TestScriptJobTracking:
    """Tests for per-job tracking in _run_script."""

    @patch("autopilot.plan.script")
    def test_script_creates_job_per_narrative(
        self,
        mock_script,
        minimal_config,
    ):
        """Creates one 'generate_script' job per narrative."""
        from autopilot.orchestrator import _run_script

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"},
            {"narrative_id": "n2"},
        ]
        db.get_narrative_script.return_value = None

        _run_script(
            config=minimal_config,
            db=db,
            run_id="run_s",
            emit_fn=MagicMock(),
        )

        assert db.insert_job.call_count == 2
        for call in db.insert_job.call_args_list:
            assert call[0][1] == "SCRIPT"
            assert call[0][2] == "generate_script"
            assert call[1]["worker"] == "cpu"

        target_ids = [c[1]["target_id"] for c in db.insert_job.call_args_list]
        assert "n1" in target_ids
        assert "n2" in target_ids

    @patch("autopilot.plan.script")
    def test_script_no_jobs_when_no_run_id(
        self,
        mock_script,
        minimal_config,
    ):
        """When run_id=None, no insert_job calls."""
        from autopilot.orchestrator import _run_script

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_narrative_script.return_value = None

        _run_script(config=minimal_config, db=db)

        db.insert_job.assert_not_called()


class TestEdlJobTracking:
    """Tests for per-job tracking in _run_edl."""

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_creates_three_jobs_per_narrative(
        self,
        mock_edl,
        mock_validator,
        mock_otio,
        minimal_config,
    ):
        """Creates 'generate_edl', 'validate_edl', 'otio_export' per narrative."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = None
        db.get_narrative_script.return_value = "some script"
        mock_edl.generate_edl.return_value = {"cuts": []}
        mock_validator.validate_edl.return_value = MagicMock(passed=True)

        _run_edl(
            config=minimal_config,
            db=db,
            run_id="run_e",
            emit_fn=MagicMock(),
        )

        assert db.insert_job.call_count == 3
        job_types = [c[0][2] for c in db.insert_job.call_args_list]
        assert "generate_edl" in job_types
        assert "validate_edl" in job_types
        assert "otio_export" in job_types
        for call in db.insert_job.call_args_list:
            assert call[0][1] == "EDL"
            assert call[1]["worker"] == "cpu"

    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    def test_edl_no_jobs_when_no_run_id(
        self,
        mock_edl,
        mock_validator,
        mock_otio,
        minimal_config,
    ):
        """When run_id=None, no insert_job calls."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = None
        db.get_narrative_script.return_value = "some script"
        mock_edl.generate_edl.return_value = {"cuts": []}
        mock_validator.validate_edl.return_value = MagicMock(passed=True)

        _run_edl(config=minimal_config, db=db)

        db.insert_job.assert_not_called()


class TestSourceAssetsJobTracking:
    """Tests for per-job tracking in _run_source_assets."""

    @patch("autopilot.source.resolve")
    def test_source_creates_job_per_narrative(
        self,
        mock_resolve,
        minimal_config,
    ):
        """Creates one 'resolve_assets' job per narrative."""
        from autopilot.orchestrator import _run_source_assets

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {"edl_json": '{"cuts": []}'}

        _run_source_assets(
            config=minimal_config,
            db=db,
            run_id="run_sa",
            emit_fn=MagicMock(),
        )

        job_calls = [c for c in db.insert_job.call_args_list if c[0][2] == "resolve_assets"]
        assert len(job_calls) == 1
        assert job_calls[0][0][1] == "SOURCE_ASSETS"
        assert job_calls[0][1]["worker"] == "cpu"
        assert job_calls[0][1]["target_id"] == "n1"

    @patch("autopilot.source.resolve")
    def test_source_no_jobs_when_no_run_id(
        self,
        mock_resolve,
        minimal_config,
    ):
        """When run_id=None, no insert_job calls."""
        from autopilot.orchestrator import _run_source_assets

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {"edl_json": '{"cuts": []}'}

        _run_source_assets(config=minimal_config, db=db)

        db.insert_job.assert_not_called()


class TestRenderJobTracking:
    """Tests for per-job tracking in _run_render."""

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_creates_two_jobs_per_narrative(
        self,
        mock_router,
        mock_rv,
        minimal_config,
    ):
        """Creates 'render' (gpu) and 'validate_render' (cpu) per narrative."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {"edl_json": '{"cuts": []}'}
        mock_router.route_and_render.return_value = Path("/out/n1.mp4")
        mock_rv.validate_render.return_value = MagicMock(issues=[])

        _run_render(
            config=minimal_config,
            db=db,
            run_id="run_r",
            emit_fn=MagicMock(),
        )

        assert db.insert_job.call_count == 2
        job_types = [c[0][2] for c in db.insert_job.call_args_list]
        assert "render" in job_types
        assert "validate_render" in job_types
        for call in db.insert_job.call_args_list:
            assert call[0][1] == "RENDER"
            assert call[1]["target_id"] == "n1"

        # Check workers
        workers = {c[0][2]: c[1]["worker"] for c in db.insert_job.call_args_list}
        assert workers["render"] == "gpu"
        assert workers["validate_render"] == "cpu"

    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    def test_render_no_jobs_when_no_run_id(
        self,
        mock_router,
        mock_rv,
        minimal_config,
    ):
        """When run_id=None, no insert_job calls."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {"edl_json": '{"cuts": []}'}
        mock_router.route_and_render.return_value = Path("/out/n1.mp4")
        mock_rv.validate_render.return_value = MagicMock(issues=[])

        _run_render(config=minimal_config, db=db)

        db.insert_job.assert_not_called()


class TestUploadJobTracking:
    """Tests for per-job tracking in _run_upload."""

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_creates_two_jobs_per_narrative(
        self,
        mock_yt,
        mock_thumb,
        minimal_config,
    ):
        """Creates 'upload_video' and 'extract_thumbnail' per narrative."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_upload.return_value = None
        db.get_edit_plan.return_value = {"render_path": "/out/n1.mp4"}

        _run_upload(
            config=minimal_config,
            db=db,
            run_id="run_u",
            emit_fn=MagicMock(),
        )

        assert db.insert_job.call_count == 2
        job_types = [c[0][2] for c in db.insert_job.call_args_list]
        assert "upload_video" in job_types
        assert "extract_thumbnail" in job_types
        for call in db.insert_job.call_args_list:
            assert call[0][1] == "UPLOAD"
            assert call[1]["worker"] == "cpu"
            assert call[1]["target_id"] == "n1"

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    def test_upload_no_jobs_when_no_run_id(
        self,
        mock_yt,
        mock_thumb,
        minimal_config,
    ):
        """When run_id=None, no insert_job calls."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_upload.return_value = None
        db.get_edit_plan.return_value = {"render_path": "/out/n1.mp4"}

        _run_upload(config=minimal_config, db=db)

        db.insert_job.assert_not_called()


class TestOrchestratorPassesJobKwargs:
    """Test that PipelineOrchestrator.run() passes run_id and emit_fn to stages."""

    def test_run_passes_run_id_to_stage_funcs(self) -> None:
        """Each stage.func is called with run_id=self._run_id."""
        orch = PipelineOrchestrator()
        mock_funcs = []
        for stage in orch.stages:
            mf = MagicMock()
            stage.func = mf
            mock_funcs.append((stage.name, mf))

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        for stage_name, mf in mock_funcs:
            mf.assert_called_once()
            call_kwargs = mf.call_args[1]
            assert "run_id" in call_kwargs, f"{stage_name} missing run_id"
            assert call_kwargs["run_id"] == orch._run_id

    def test_run_passes_emit_fn_to_stage_funcs(self) -> None:
        """Each stage.func is called with emit_fn=self._emit_event."""
        orch = PipelineOrchestrator()
        mock_funcs = []
        for stage in orch.stages:
            mf = MagicMock()
            stage.func = mf
            mock_funcs.append((stage.name, mf))

        mock_db = MagicMock()
        orch.run(config=MagicMock(), db=mock_db)

        for stage_name, mf in mock_funcs:
            call_kwargs = mf.call_args[1]
            assert "emit_fn" in call_kwargs, f"{stage_name} missing emit_fn"
            assert call_kwargs["emit_fn"] == orch._emit_event

    def test_run_passes_none_run_id_when_insert_run_fails(self) -> None:
        """When insert_run fails, stages get run_id=None."""
        orch = PipelineOrchestrator()
        mock_funcs = []
        for stage in orch.stages:
            mf = MagicMock()
            stage.func = mf
            mock_funcs.append(mf)

        mock_db = MagicMock()
        mock_db.insert_run.side_effect = RuntimeError("db error")
        orch.run(config=MagicMock(), db=mock_db)

        for mf in mock_funcs:
            assert mf.call_args[1]["run_id"] is None


class TestJobTrackingIntegration:
    """Integration tests: run pipeline with real CatalogDB and stub stages that
    create jobs via _track_job.  Verify pipeline_jobs + pipeline_events records."""

    def test_jobs_created_in_db_with_correct_run_id(self, catalog_db) -> None:
        """Stub stages create jobs via _track_job; verify pipeline_jobs records."""
        from autopilot.orchestrator import _track_job

        orch = PipelineOrchestrator()

        # Replace all stages with no-ops except INGEST which creates 2 jobs
        def fake_ingest(*, config, db, force, run_id=None, emit_fn=None):
            with _track_job(
                db,
                "INGEST",
                "ingest_file",
                target_id="media_001",
                worker="cpu",
                run_id=run_id,
                emit_fn=emit_fn,
            ):
                pass  # simulate work
            with _track_job(db, "INGEST", "dedup", worker="cpu", run_id=run_id, emit_fn=emit_fn):
                pass

        for stage in orch.stages:
            if stage.name == "INGEST":
                stage.func = fake_ingest
            else:
                stage.func = MagicMock()

        orch.run(config=MagicMock(), db=catalog_db)

        jobs = catalog_db.list_jobs(run_id=orch._run_id)
        assert len(jobs) == 2

        ingest_file_jobs = [j for j in jobs if j["job_type"] == "ingest_file"]
        assert len(ingest_file_jobs) == 1
        j = ingest_file_jobs[0]
        assert j["stage"] == "INGEST"
        assert j["status"] == "done"
        assert j["target_id"] == "media_001"
        assert j["worker"] == "cpu"
        assert j["run_id"] == orch._run_id
        assert float(j["duration_seconds"]) >= 0

        dedup_jobs = [j for j in jobs if j["job_type"] == "dedup"]
        assert len(dedup_jobs) == 1
        assert dedup_jobs[0]["status"] == "done"

    def test_job_events_appear_in_pipeline_events(self, catalog_db) -> None:
        """job_started and job_completed events appear in pipeline_events."""
        from autopilot.orchestrator import _track_job

        orch = PipelineOrchestrator()

        def fake_analyze(*, config, db, force, run_id=None, emit_fn=None):
            with _track_job(
                db, "ANALYZE", "asr", target_id="m1", worker="gpu", run_id=run_id, emit_fn=emit_fn
            ):
                pass

        for stage in orch.stages:
            if stage.name == "ANALYZE":
                stage.func = fake_analyze
            else:
                stage.func = MagicMock()

        orch.run(config=MagicMock(), db=catalog_db)

        events = catalog_db.get_events_since(0)
        event_types = [e["event_type"] for e in events]

        assert "job_started" in event_types
        assert "job_completed" in event_types

        # Find the job_started event for ANALYZE
        job_started_events = [
            e for e in events if e["event_type"] == "job_started" and e["stage"] == "ANALYZE"
        ]
        assert len(job_started_events) >= 1

    def test_job_error_recorded_on_stage_failure(self, catalog_db) -> None:
        """A failing job within a stage records status='error' in pipeline_jobs."""
        from autopilot.orchestrator import _track_job

        orch = PipelineOrchestrator()

        def fake_script(*, config, db, force, run_id=None, emit_fn=None):
            with _track_job(
                db,
                "SCRIPT",
                "generate_script",
                target_id="n1",
                worker="cpu",
                run_id=run_id,
                emit_fn=emit_fn,
            ):
                raise ValueError("script generation failed")

        for stage in orch.stages:
            if stage.name == "SCRIPT":
                stage.func = fake_script
            else:
                stage.func = MagicMock()

        orch.run(config=MagicMock(), db=catalog_db)

        jobs = catalog_db.list_jobs(run_id=orch._run_id, stage="SCRIPT")
        assert len(jobs) == 1
        assert jobs[0]["status"] == "error"
        assert "script generation failed" in jobs[0]["error_message"]

        # job_error event should also be emitted
        events = catalog_db.get_events_since(0)
        job_error_events = [
            e for e in events if e["event_type"] == "job_error" and e["stage"] == "SCRIPT"
        ]
        assert len(job_error_events) >= 1


# ── Gate system tests ────────────────────────────────────────────────────


class TestGateStageNameMapping:
    """Tests for PipelineOrchestrator._gate_stage_name() static method."""

    def test_ingest_maps_to_lowercase(self) -> None:
        assert PipelineOrchestrator._gate_stage_name("INGEST") == "ingest"

    def test_analyze_maps_to_lowercase(self) -> None:
        assert PipelineOrchestrator._gate_stage_name("ANALYZE") == "analyze"

    def test_source_assets_maps_to_source(self) -> None:
        """SOURCE_ASSETS must map to 'source' (the DB gate name)."""
        assert PipelineOrchestrator._gate_stage_name("SOURCE_ASSETS") == "source"

    def test_all_nine_stages_map(self) -> None:
        """Every orchestrator stage should map to a known DB gate name."""
        expected = {
            "INGEST": "ingest",
            "ANALYZE": "analyze",
            "CLASSIFY": "classify",
            "NARRATE": "narrate",
            "SCRIPT": "script",
            "EDL": "edl",
            "SOURCE_ASSETS": "source",
            "RENDER": "render",
            "UPLOAD": "upload",
        }
        for stage, gate in expected.items():
            assert PipelineOrchestrator._gate_stage_name(stage) == gate

    def test_callable_without_instance(self) -> None:
        """_gate_stage_name is a static method — no instance needed."""
        result = PipelineOrchestrator._gate_stage_name("INGEST")
        assert result == "ingest"


class TestCheckGateAuto:
    """Tests for _check_gate() in 'auto' mode (default)."""

    def test_gate_auto_returns_approved(self, catalog_db) -> None:
        """Auto mode should return 'approved'."""
        catalog_db.init_default_gates()
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"
        result = orch._check_gate("INGEST")
        assert result == "approved"

    def test_gate_auto_updates_status_to_approved(self, catalog_db) -> None:
        """Auto mode updates the gate status to 'approved' with decided_by='system'."""
        catalog_db.init_default_gates()
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"
        orch._check_gate("INGEST")
        gate = catalog_db.get_gate("ingest")
        assert gate["status"] == "approved"
        assert gate["decided_by"] == "system"

    def test_gate_auto_emits_gate_passed_event(self, catalog_db) -> None:
        """Auto mode emits a 'gate_passed' event."""
        catalog_db.init_default_gates()
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"
        orch._check_gate("INGEST")
        events = catalog_db.get_events_since(0)
        gate_events = [e for e in events if e["event_type"] == "gate_passed"]
        assert len(gate_events) == 1
        assert gate_events[0]["stage"] == "INGEST"

    def test_gate_auto_fetches_by_mapped_name(self, catalog_db) -> None:
        """Gate is fetched using the mapped name (SOURCE_ASSETS -> source)."""
        catalog_db.init_default_gates()
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"
        orch._check_gate("SOURCE_ASSETS")
        gate = catalog_db.get_gate("source")
        assert gate["status"] == "approved"


class TestCheckGateNotify:
    """Tests for _check_gate() in 'notify' mode."""

    def test_gate_notify_returns_approved(self, catalog_db) -> None:
        """Notify mode should return 'approved'."""
        catalog_db.init_default_gates()
        catalog_db.update_gate("ingest", mode="notify")
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"
        result = orch._check_gate("INGEST")
        assert result == "approved"

    def test_gate_notify_updates_status(self, catalog_db) -> None:
        """Notify mode updates the gate to approved with decided_by='system'."""
        catalog_db.init_default_gates()
        catalog_db.update_gate("analyze", mode="notify")
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"
        orch._check_gate("ANALYZE")
        gate = catalog_db.get_gate("analyze")
        assert gate["status"] == "approved"
        assert gate["decided_by"] == "system"

    def test_gate_notify_emits_gate_passed(self, catalog_db) -> None:
        """Notify mode emits a 'gate_passed' event."""
        catalog_db.init_default_gates()
        catalog_db.update_gate("classify", mode="notify")
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"
        orch._check_gate("CLASSIFY")
        events = catalog_db.get_events_since(0)
        gate_events = [e for e in events if e["event_type"] == "gate_passed"]
        assert len(gate_events) == 1
        assert gate_events[0]["stage"] == "CLASSIFY"


class TestCheckGatePause:
    """Tests for _check_gate() in 'pause' mode."""

    def test_gate_pause_sets_waiting_and_polls(self, catalog_db) -> None:
        """Pause mode sets status='waiting', emits 'gate_waiting', then polls."""
        catalog_db.init_default_gates()
        catalog_db.update_gate("ingest", mode="pause")
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"

        # Simulate external approval: after first get_gate returns 'waiting',
        # update gate to 'approved' so the poll loop exits.
        call_count = 0
        original_get_gate = catalog_db.get_gate

        def fake_get_gate(stage):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                # Simulate external approval
                catalog_db.update_gate(stage, status="approved", decided_by="human")
            return original_get_gate(stage)

        catalog_db.get_gate = fake_get_gate

        with patch("autopilot.orchestrator.time.sleep"):
            result = orch._check_gate("INGEST")

        assert result == "approved"
        # gate_waiting event emitted
        events = catalog_db.get_events_since(0)
        waiting_events = [e for e in events if e["event_type"] == "gate_waiting"]
        assert len(waiting_events) == 1
        assert waiting_events[0]["stage"] == "INGEST"
        # gate_approved event emitted
        approved_events = [e for e in events if e["event_type"] == "gate_approved"]
        assert len(approved_events) == 1

    def test_gate_pause_skipped_status(self, catalog_db) -> None:
        """Pause mode returns 'skipped' when gate status is set to 'skipped'."""
        catalog_db.init_default_gates()
        catalog_db.update_gate("narrate", mode="pause")
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"

        call_count = 0
        original_get_gate = catalog_db.get_gate

        def fake_get_gate(stage):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                catalog_db.update_gate(stage, status="skipped")
            return original_get_gate(stage)

        catalog_db.get_gate = fake_get_gate

        with patch("autopilot.orchestrator.time.sleep"):
            result = orch._check_gate("NARRATE")

        assert result == "skipped"
        events = catalog_db.get_events_since(0)
        skipped_events = [e for e in events if e["event_type"] == "gate_skipped"]
        assert len(skipped_events) == 1


class TestCheckGateTimeout:
    """Tests for _check_gate() pause mode with timeout."""

    def test_gate_pause_timeout_auto_approves(self, catalog_db) -> None:
        """When timeout_hours is exceeded, gate is auto-approved with notes."""
        catalog_db.init_default_gates()
        catalog_db.update_gate("ingest", mode="pause", timeout_hours=0.0001)
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"

        # Mock time.monotonic to simulate timeout: first call returns 0, second returns large value
        mono_values = iter([0.0, 100.0, 200.0])

        with (
            patch("autopilot.orchestrator.time.sleep"),
            patch("autopilot.orchestrator.time.monotonic", side_effect=mono_values),
        ):
            result = orch._check_gate("INGEST")

        assert result == "approved"

        # Gate should be updated with timeout notes
        gate = catalog_db.get_gate("ingest")
        assert gate["status"] == "approved"
        assert gate["decided_by"] == "system"
        assert "timeout" in gate["notes"]

        # gate_approved event with reason=timeout
        events = catalog_db.get_events_since(0)
        approved_events = [e for e in events if e["event_type"] == "gate_approved"]
        assert len(approved_events) == 1
        payload = json.loads(approved_events[0]["payload_json"])
        assert payload["reason"] == "timeout"


class TestCheckGateShutdown:
    """Tests for _check_gate() pause mode with shutdown."""

    def setup_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def teardown_method(self) -> None:
        from autopilot.orchestrator import _reset_shutdown

        _reset_shutdown()

    def test_gate_pause_returns_skipped_on_shutdown(self, catalog_db) -> None:
        """When shutdown is requested, _check_gate returns 'skipped' immediately."""
        from autopilot.orchestrator import request_shutdown

        catalog_db.init_default_gates()
        catalog_db.update_gate("ingest", mode="pause")
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"

        # Request shutdown before calling _check_gate
        request_shutdown()

        result = orch._check_gate("INGEST")
        assert result == "skipped"


class TestCheckGateNotFound:
    """Tests for _check_gate() when gate row is missing."""

    def test_gate_not_found_returns_approved(self, catalog_db) -> None:
        """When gate row doesn't exist, return 'approved' as graceful fallback."""
        # Don't call init_default_gates — no gate rows
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"
        result = orch._check_gate("INGEST")
        assert result == "approved"

    def test_gate_not_found_logs_warning(self, catalog_db, caplog) -> None:
        """When gate row doesn't exist, a warning is logged."""
        orch = PipelineOrchestrator()
        orch._db = catalog_db
        orch._run_id = "test-run-id"
        with caplog.at_level(logging.WARNING):
            orch._check_gate("INGEST")
        assert any("Gate not found" in r.message for r in caplog.records)

    def test_gate_check_resilience_on_db_error(self, caplog) -> None:
        """When DB raises, _check_gate returns 'approved' and logs warning."""
        orch = PipelineOrchestrator()
        mock_db = MagicMock()
        mock_db.get_gate.side_effect = RuntimeError("db connection lost")
        orch._db = mock_db
        orch._run_id = "test-run-id"
        with caplog.at_level(logging.WARNING):
            result = orch._check_gate("INGEST")
        assert result == "approved"
        assert any("Gate check failed" in r.message for r in caplog.records)


class TestGateInitInRun:
    """Tests for gate initialization in run()."""

    def test_run_calls_init_default_gates(self, catalog_db) -> None:
        """run() calls db.init_default_gates()."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        # Spy on init_default_gates
        original = catalog_db.init_default_gates
        call_count = 0

        def spy():
            nonlocal call_count
            call_count += 1
            return original()

        catalog_db.init_default_gates = spy

        orch.run(config=MagicMock(), db=catalog_db)
        assert call_count == 1

    def test_run_resets_gate_statuses_to_idle(self) -> None:
        """run() resets all gate statuses to 'idle' at start."""
        mock_db = MagicMock()
        mock_db.get_all_gates.return_value = [
            {"stage": s}
            for s in (
                "ingest",
                "analyze",
                "classify",
                "narrate",
                "script",
                "edl",
                "source",
                "render",
                "upload",
            )
        ]
        mock_db.get_gate.return_value = {"mode": "auto", "status": "idle", "timeout_hours": None}

        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        orch.run(config=MagicMock(), db=mock_db)

        # Verify update_gate was called with status='idle' for each stage
        idle_calls = [
            c
            for c in mock_db.update_gate.call_args_list
            if c.kwargs.get("status") == "idle"
            or (len(c.args) >= 1 and any(kw == "idle" for kw in c.kwargs.values()))
        ]
        assert len(idle_calls) == 9

    def test_run_proceeds_when_init_gates_raises(self, catalog_db) -> None:
        """Pipeline still runs if init_default_gates() raises."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        # Make init_default_gates raise
        catalog_db.init_default_gates = MagicMock(side_effect=RuntimeError("db error"))

        # Should still complete
        results = orch.run(config=MagicMock(), db=catalog_db)
        assert len(results) == 9


class TestGateIntegrationInRun:
    """Tests for _check_gate integration in run() stage loop."""

    def test_check_gate_called_before_each_stage(self) -> None:
        """_check_gate is called once per stage before execution."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        gate_calls: list[str] = []

        def spy_check(stage_name):
            gate_calls.append(stage_name)
            return "approved"

        orch._check_gate = spy_check

        mock_db = MagicMock()
        mock_db.get_all_gates.return_value = [
            {"stage": s}
            for s in (
                "ingest",
                "analyze",
                "classify",
                "narrate",
                "script",
                "edl",
                "source",
                "render",
                "upload",
            )
        ]
        mock_db.get_gate.return_value = {"mode": "auto", "status": "idle", "timeout_hours": None}
        orch.run(config=MagicMock(), db=mock_db)

        assert gate_calls == [
            "INGEST",
            "ANALYZE",
            "CLASSIFY",
            "NARRATE",
            "SCRIPT",
            "EDL",
            "SOURCE_ASSETS",
            "RENDER",
            "UPLOAD",
        ]

    def test_gate_skipped_stage_not_executed(self) -> None:
        """When _check_gate returns 'skipped', stage func is NOT called."""
        orch = PipelineOrchestrator()
        mock_funcs = {}
        for stage in orch.stages:
            stage.func = MagicMock()
            mock_funcs[stage.name] = stage.func

        def selective_gate(stage_name):
            if stage_name == "RENDER":
                return "skipped"
            return "approved"

        orch._check_gate = selective_gate

        mock_db = MagicMock()
        mock_db.get_all_gates.return_value = [
            {"stage": s}
            for s in (
                "ingest",
                "analyze",
                "classify",
                "narrate",
                "script",
                "edl",
                "source",
                "render",
                "upload",
            )
        ]
        mock_db.get_gate.return_value = {"mode": "auto", "status": "idle", "timeout_hours": None}
        results = orch.run(config=MagicMock(), db=mock_db)

        # RENDER stage func should NOT be called
        mock_funcs["RENDER"].assert_not_called()
        # Result should be SKIPPED
        assert results["RENDER"].status == StageStatus.SKIPPED

    def test_gate_skipped_does_not_cascade_to_dependents(self) -> None:
        """Gate-skipped stages do NOT propagate skip to downstream stages."""
        orch = PipelineOrchestrator()
        mock_funcs = {}
        for stage in orch.stages:
            stage.func = MagicMock()
            mock_funcs[stage.name] = stage.func

        def selective_gate(stage_name):
            if stage_name == "SOURCE_ASSETS":
                return "skipped"
            return "approved"

        orch._check_gate = selective_gate

        mock_db = MagicMock()
        mock_db.get_all_gates.return_value = [
            {"stage": s}
            for s in (
                "ingest",
                "analyze",
                "classify",
                "narrate",
                "script",
                "edl",
                "source",
                "render",
                "upload",
            )
        ]
        mock_db.get_gate.return_value = {"mode": "auto", "status": "idle", "timeout_hours": None}
        results = orch.run(config=MagicMock(), db=mock_db)

        # SOURCE_ASSETS skipped via gate
        assert results["SOURCE_ASSETS"].status == StageStatus.SKIPPED
        # RENDER depends on SOURCE_ASSETS but should still run (gate skip != error skip)
        mock_funcs["RENDER"].assert_called_once()
        assert results["RENDER"].status == StageStatus.DONE


class TestGateBackwardsCompat:
    """Tests for backwards compatibility of gate system with human_review_fn."""

    @patch("autopilot.organize.narratives")
    def test_human_review_fn_still_invoked_with_auto_gate(self, mock_narratives) -> None:
        """With human_review_fn and NARRATE gate mode='auto', callback is still used."""
        narr = MagicMock()
        narr.narrative_id = "n1"
        mock_narratives.build_master_storyboard.return_value = "sb"
        mock_narratives.propose_narratives.return_value = [narr]
        mock_narratives.format_for_review.return_value = "review text"

        review_fn = MagicMock(return_value=["n1"])
        orch = PipelineOrchestrator(human_review_fn=review_fn)

        # Replace all non-NARRATE stages with mocks
        for stage in orch.stages:
            if stage.name != "NARRATE":
                stage.func = MagicMock()

        mock_db = MagicMock()
        mock_db.get_all_gates.return_value = [
            {"stage": s}
            for s in (
                "ingest",
                "analyze",
                "classify",
                "narrate",
                "script",
                "edl",
                "source",
                "render",
                "upload",
            )
        ]
        mock_db.get_gate.return_value = {"mode": "auto", "status": "idle", "timeout_hours": None}
        mock_db.list_narratives.return_value = []  # no checkpoint hit

        results = orch.run(config=MagicMock(), db=mock_db)

        # The human_review_fn callback should still have been invoked
        review_fn.assert_called_once()
        assert results["NARRATE"].status == StageStatus.DONE

    def test_existing_narrate_test_still_passes(self) -> None:
        """The existing test_narrate_calls_human_review_callback pattern still works.

        This is a meta-test confirming the gate system doesn't break
        _run_narrate's internal human_review_fn logic.
        """
        from autopilot.orchestrator import _run_narrate

        with patch("autopilot.organize.narratives") as mock_narratives:
            narr = MagicMock()
            narr.narrative_id = "n1"
            mock_narratives.build_master_storyboard.return_value = "sb"
            mock_narratives.propose_narratives.return_value = [narr]
            mock_narratives.format_for_review.return_value = "review text"
            db = MagicMock()
            db.list_narratives.return_value = []  # no checkpoint hit

            review_fn = MagicMock(return_value=["n1"])
            _run_narrate(config=MagicMock(), db=db, human_review_fn=review_fn)

            review_fn.assert_called_once_with("review text", [narr])


class TestDryRunSkipsGateCheck:
    """REVIEW FIX: dry_run=True must NOT call _check_gate() for any stage.

    If _check_gate runs before dry_run guard, pause-mode gates would block
    dry_run indefinitely via the 2s polling loop.
    """

    def test_dry_run_does_not_call_check_gate(self) -> None:
        """dry_run=True should never invoke _check_gate()."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        gate_calls: list[str] = []

        def spy_check(stage_name):
            gate_calls.append(stage_name)
            raise AssertionError(
                f"_check_gate should NOT be called in dry_run, but was called for {stage_name}"
            )

        orch._check_gate = spy_check

        mock_db = MagicMock()
        mock_db.get_all_gates.return_value = [
            {"stage": s}
            for s in (
                "ingest",
                "analyze",
                "classify",
                "narrate",
                "script",
                "edl",
                "source",
                "render",
                "upload",
            )
        ]
        mock_db.get_gate.return_value = {
            "mode": "auto",
            "status": "idle",
            "timeout_hours": None,
        }

        # This should complete without calling _check_gate
        results = orch.run(config=MagicMock(), db=mock_db, dry_run=True)

        # All stages should be SKIPPED (dry_run behavior)
        assert len(results) == 9
        for name, result in results.items():
            assert result.status == StageStatus.SKIPPED, f"{name} should be SKIPPED in dry_run"
        # _check_gate should never have been called
        assert gate_calls == [], f"_check_gate was called for stages: {gate_calls}"


class TestGateResetUsesPublicAPI:
    """REVIEW FIX: run() must use db.get_all_gates() (public API) for gate
    reset, not db._PIPELINE_STAGES (private attribute).
    """

    def test_gate_reset_calls_get_all_gates(self) -> None:
        """run() uses db.get_all_gates() to enumerate gates for reset."""
        mock_db = MagicMock()
        mock_db.get_all_gates.return_value = [
            {"stage": s}
            for s in (
                "ingest",
                "analyze",
                "classify",
                "narrate",
                "script",
                "edl",
                "source",
                "render",
                "upload",
            )
        ]
        mock_db.get_gate.return_value = {
            "mode": "auto",
            "status": "idle",
            "timeout_hours": None,
        }

        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        orch.run(config=MagicMock(), db=mock_db)

        # get_all_gates must have been called (public API)
        mock_db.get_all_gates.assert_called()

    def test_gate_reset_does_not_access_private_pipeline_stages(self) -> None:
        """run() must NOT access db._PIPELINE_STAGES (private attribute)."""
        mock_db = MagicMock()
        mock_db.get_all_gates.return_value = [
            {"stage": s}
            for s in (
                "ingest",
                "analyze",
                "classify",
                "narrate",
                "script",
                "edl",
                "source",
                "render",
                "upload",
            )
        ]
        mock_db.get_gate.return_value = {
            "mode": "auto",
            "status": "idle",
            "timeout_hours": None,
        }

        # Remove _PIPELINE_STAGES to ensure it's not accessed
        # MagicMock auto-creates attributes, so we use a spec-limited mock
        # or just verify via PropertyMock
        access_log: list[str] = []
        original_getattr = type(mock_db).__getattr__

        def tracking_getattr(self_mock, name):
            if name == "_PIPELINE_STAGES":
                access_log.append(name)
            return original_getattr(self_mock, name)

        with patch.object(type(mock_db), "__getattr__", tracking_getattr):
            orch = PipelineOrchestrator()
            for stage in orch.stages:
                stage.func = MagicMock()
            orch.run(config=MagicMock(), db=mock_db)

        assert "_PIPELINE_STAGES" not in access_log, (
            "run() accessed db._PIPELINE_STAGES — should use db.get_all_gates() instead"
        )

    def test_update_gate_called_for_each_gate_from_get_all_gates(self) -> None:
        """update_gate is called with status='idle' for each gate returned
        by get_all_gates()."""
        mock_db = MagicMock()
        gate_stages = [
            "ingest",
            "analyze",
            "classify",
            "narrate",
            "script",
            "edl",
            "source",
            "render",
            "upload",
        ]
        mock_db.get_all_gates.return_value = [{"stage": s} for s in gate_stages]
        mock_db.get_gate.return_value = {
            "mode": "auto",
            "status": "idle",
            "timeout_hours": None,
        }

        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        orch.run(config=MagicMock(), db=mock_db)

        # Verify update_gate(stage, status='idle') called for each
        idle_calls = [
            c
            for c in mock_db.update_gate.call_args_list
            if len(c.args) >= 1 and c.kwargs.get("status") == "idle"
        ]
        idle_stages = [c.args[0] for c in idle_calls]
        assert sorted(idle_stages) == sorted(gate_stages)
