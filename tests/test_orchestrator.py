"""Tests for the pipeline orchestrator (autopilot.orchestrator)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autopilot.orchestrator import PipelineOrchestrator, StageDefinition, StageStatus

EXPECTED_STAGES = [
    "INGEST", "ANALYZE", "CLASSIFY", "NARRATE", "SCRIPT",
    "EDL", "SOURCE_ASSETS", "RENDER", "UPLOAD",
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
            assert stage.dependencies == EXPECTED_DEPS[stage.name], (
                f"{stage.name} deps mismatch"
            )

    def test_stages_have_callable_functions(self) -> None:
        """Every stage has a callable func attribute."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            assert callable(stage.func), f"{stage.name} func is not callable"

    def test_stages_have_positive_estimated_seconds(self) -> None:
        """Every stage has estimated_seconds >= 0."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            assert stage.estimated_seconds >= 0, (
                f"{stage.name} has negative estimated_seconds"
            )


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
            assert func is not _stage_stub, (
                f"{stage.name} still uses _stage_stub"
            )

    def test_stage_functions_accept_config_and_db_kwargs(self) -> None:
        """Each stage function's signature accepts config and db kwargs."""
        import inspect

        orch = PipelineOrchestrator()
        for stage in orch.stages:
            func = getattr(stage.func, "func", stage.func)
            sig = inspect.signature(func)
            param_names = set(sig.parameters.keys())
            assert "config" in param_names, (
                f"{stage.name} func missing 'config' parameter"
            )
            assert "db" in param_names, (
                f"{stage.name} func missing 'db' parameter"
            )


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
            mock_fn.assert_called_once_with(config=mock_config, db=mock_db)

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
            assert f"[RUNNING] {stage.name}" in log_text, (
                f"Missing RUNNING log for {stage.name}"
            )
            assert f"[DONE] {stage.name}" in log_text, (
                f"Missing DONE log for {stage.name}"
            )

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
        orch._stage_map["ANALYZE"].func = MagicMock(
            side_effect=RuntimeError("analyze failed")
        )

        results = orch.run(config=MagicMock(), db=MagicMock())

        assert results["ANALYZE"].status == StageStatus.ERROR

    def test_run_skips_dependents_on_error(self) -> None:
        """When ANALYZE errors, CLASSIFY and all downstream are SKIPPED."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()
        orch._stage_map["ANALYZE"].func = MagicMock(
            side_effect=RuntimeError("analyze failed")
        )

        results = orch.run(config=MagicMock(), db=MagicMock())

        assert results["INGEST"].status == StageStatus.DONE
        assert results["ANALYZE"].status == StageStatus.ERROR
        # All stages that depend (transitively) on ANALYZE should be SKIPPED
        for name in ["CLASSIFY", "NARRATE", "SCRIPT", "EDL", "SOURCE_ASSETS", "RENDER", "UPLOAD"]:
            assert results[name].status == StageStatus.SKIPPED, (
                f"{name} should be SKIPPED"
            )

    def test_run_reports_error_message(self) -> None:
        """Error message is captured in the stage result."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()
        orch._stage_map["INGEST"].func = MagicMock(
            side_effect=RuntimeError("ingest boom")
        )

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

    def test_dry_run_prints_execution_plan(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """dry_run=True logs DRY-RUN prefix for each stage."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock(), dry_run=True)

        log_text = caplog.text
        for stage in orch.stages:
            assert f"[DRY-RUN] {stage.name}" in log_text, (
                f"Missing DRY-RUN log for {stage.name}"
            )

    def test_dry_run_shows_estimated_time(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
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

    def test_run_warns_when_over_budget(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
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

    @patch("autopilot.orchestrator.dedup")
    @patch("autopilot.orchestrator.normalizer")
    @patch("autopilot.orchestrator.scanner")
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

    @patch("autopilot.orchestrator.dedup")
    @patch("autopilot.orchestrator.normalizer")
    @patch("autopilot.orchestrator.scanner")
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

        _run_ingest(config=minimal_config, db=db)

        assert db.insert_media.call_count == 2

    @patch("autopilot.orchestrator.dedup")
    @patch("autopilot.orchestrator.normalizer")
    @patch("autopilot.orchestrator.scanner")
    def test_ingest_normalizes_audio(
        self, mock_scanner, mock_normalizer, mock_dedup, minimal_config
    ):
        """_run_ingest calls normalize_audio for each media file."""
        from autopilot.orchestrator import _run_ingest

        mock_file = MagicMock()
        mock_file.file_path = Path("/fake/video.mp4")
        mock_scanner.scan_directory.return_value = [mock_file]
        db = MagicMock()

        _run_ingest(config=minimal_config, db=db)

        mock_normalizer.normalize_audio.assert_called_once()

    @patch("autopilot.orchestrator.dedup")
    @patch("autopilot.orchestrator.normalizer")
    @patch("autopilot.orchestrator.scanner")
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

    @patch("autopilot.orchestrator.GPUScheduler")
    @patch("autopilot.orchestrator.faces")
    @patch("autopilot.orchestrator.audio_events")
    @patch("autopilot.orchestrator.embeddings")
    @patch("autopilot.orchestrator.objects")
    @patch("autopilot.orchestrator.scenes")
    @patch("autopilot.orchestrator.asr")
    def test_analyze_creates_gpu_scheduler(
        self, mock_asr, mock_scenes, mock_objects, mock_embeddings,
        mock_audio_events, mock_faces, mock_gpu_cls, minimal_config,
    ):
        """_run_analyze creates a GPUScheduler with config.processing settings."""
        from autopilot.orchestrator import _run_analyze

        db = MagicMock()
        db.list_all_media.return_value = []

        _run_analyze(config=minimal_config, db=db)

        mock_gpu_cls.assert_called_once()

    @patch("autopilot.orchestrator.GPUScheduler")
    @patch("autopilot.orchestrator.faces")
    @patch("autopilot.orchestrator.audio_events")
    @patch("autopilot.orchestrator.embeddings")
    @patch("autopilot.orchestrator.objects")
    @patch("autopilot.orchestrator.scenes")
    @patch("autopilot.orchestrator.asr")
    def test_analyze_runs_all_analysis_per_media(
        self, mock_asr, mock_scenes, mock_objects, mock_embeddings,
        mock_audio_events, mock_faces, mock_gpu_cls, minimal_config,
    ):
        """_run_analyze calls all 6 analysis functions for each media in DB."""
        from autopilot.orchestrator import _run_analyze

        media1 = {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"}
        media2 = {"id": "m2", "file_path": "/fake/v2.mp4", "status": "ingested"}
        db = MagicMock()
        db.list_all_media.return_value = [media1, media2]
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db)

        assert mock_asr.transcribe_media.call_count == 2
        assert mock_scenes.detect_shots.call_count == 2
        assert mock_objects.detect_objects.call_count == 2
        assert mock_faces.detect_faces.call_count == 2
        assert mock_embeddings.compute_embeddings.call_count == 2
        assert mock_audio_events.classify_audio_events.call_count == 2

    @patch("autopilot.orchestrator.GPUScheduler")
    @patch("autopilot.orchestrator.faces")
    @patch("autopilot.orchestrator.audio_events")
    @patch("autopilot.orchestrator.embeddings")
    @patch("autopilot.orchestrator.objects")
    @patch("autopilot.orchestrator.scenes")
    @patch("autopilot.orchestrator.asr")
    def test_analyze_calls_cluster_faces_after_analysis(
        self, mock_asr, mock_scenes, mock_objects, mock_embeddings,
        mock_audio_events, mock_faces, mock_gpu_cls, minimal_config,
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

    @patch("autopilot.orchestrator.GPUScheduler")
    @patch("autopilot.orchestrator.faces")
    @patch("autopilot.orchestrator.audio_events")
    @patch("autopilot.orchestrator.embeddings")
    @patch("autopilot.orchestrator.objects")
    @patch("autopilot.orchestrator.scenes")
    @patch("autopilot.orchestrator.asr")
    def test_analyze_skips_duplicate_media(
        self, mock_asr, mock_scenes, mock_objects, mock_embeddings,
        mock_audio_events, mock_faces, mock_gpu_cls, minimal_config,
    ):
        """_run_analyze skips media with status='duplicate'."""
        from autopilot.orchestrator import _run_analyze

        media = [
            {"id": "m1", "file_path": "/fake/v1.mp4", "status": "ingested"},
            {"id": "m2", "file_path": "/fake/v2.mp4", "status": "duplicate"},
        ]
        db = MagicMock()
        db.list_all_media.return_value = media
        mock_gpu_cls.return_value = MagicMock()

        _run_analyze(config=minimal_config, db=db)

        # Only m1 should be analyzed, not m2 (duplicate)
        assert mock_asr.transcribe_media.call_count == 1


class TestClassifyStage:
    """Tests for the real _run_classify stage function."""

    @patch("autopilot.orchestrator.classify")
    @patch("autopilot.orchestrator.cluster")
    def test_classify_calls_cluster_activities(
        self, mock_cluster, mock_classify, minimal_config
    ):
        """_run_classify calls cluster.cluster_activities with db."""
        from autopilot.orchestrator import _run_classify

        db = MagicMock()
        _run_classify(config=minimal_config, db=db)

        mock_cluster.cluster_activities.assert_called_once_with(db)

    @patch("autopilot.orchestrator.classify")
    @patch("autopilot.orchestrator.cluster")
    def test_classify_calls_label_activities(
        self, mock_cluster, mock_classify, minimal_config
    ):
        """_run_classify calls classify.label_activities with db and config.llm."""
        from autopilot.orchestrator import _run_classify

        db = MagicMock()
        _run_classify(config=minimal_config, db=db)

        mock_classify.label_activities.assert_called_once_with(db, minimal_config.llm)


class TestNarrateStage:
    """Tests for the real _run_narrate stage function."""

    @patch("autopilot.orchestrator.narratives")
    def test_narrate_builds_storyboard(self, mock_narratives, minimal_config):
        """_run_narrate calls build_master_storyboard with db."""
        from autopilot.orchestrator import _run_narrate

        mock_narratives.build_master_storyboard.return_value = "storyboard text"
        mock_narratives.propose_narratives.return_value = []
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()

        _run_narrate(config=minimal_config, db=db)

        mock_narratives.build_master_storyboard.assert_called_once_with(db)

    @patch("autopilot.orchestrator.narratives")
    def test_narrate_proposes_narratives(self, mock_narratives, minimal_config):
        """_run_narrate calls propose_narratives with storyboard, db, config."""
        from autopilot.orchestrator import _run_narrate

        mock_narratives.build_master_storyboard.return_value = "storyboard"
        mock_narratives.propose_narratives.return_value = []
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()

        _run_narrate(config=minimal_config, db=db)

        mock_narratives.propose_narratives.assert_called_once_with(
            "storyboard", db, minimal_config
        )

    @patch("autopilot.orchestrator.narratives")
    def test_narrate_calls_human_review_callback(self, mock_narratives, minimal_config):
        """_run_narrate invokes human_review_fn with formatted text and narratives."""
        from autopilot.orchestrator import _run_narrate

        narr = MagicMock()
        narr.narrative_id = "n1"
        mock_narratives.build_master_storyboard.return_value = "sb"
        mock_narratives.propose_narratives.return_value = [narr]
        mock_narratives.format_for_review.return_value = "review text"
        db = MagicMock()

        review_fn = MagicMock(return_value=["n1"])
        _run_narrate(config=minimal_config, db=db, human_review_fn=review_fn)

        review_fn.assert_called_once_with("review text", [narr])

    @patch("autopilot.orchestrator.narratives")
    def test_narrate_auto_approves_when_no_callback(self, mock_narratives, minimal_config):
        """Without human_review_fn all narratives get status='approved'."""
        from autopilot.orchestrator import _run_narrate

        narr = MagicMock()
        narr.narrative_id = "n1"
        mock_narratives.build_master_storyboard.return_value = "sb"
        mock_narratives.propose_narratives.return_value = [narr]
        mock_narratives.format_for_review.return_value = ""
        db = MagicMock()

        _run_narrate(config=minimal_config, db=db)

        db.update_narrative_status.assert_any_call("n1", "approved")

    @patch("autopilot.orchestrator.narratives")
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

        # Only approve n1, reject n2
        review_fn = MagicMock(return_value=["n1"])
        _run_narrate(config=minimal_config, db=db, human_review_fn=review_fn)

        db.update_narrative_status.assert_any_call("n1", "approved")
        db.update_narrative_status.assert_any_call("n2", "rejected")


class TestScriptStage:
    """Tests for the real _run_script stage function."""

    @patch("autopilot.orchestrator.script")
    def test_script_generates_per_approved_narrative(
        self, mock_script, minimal_config
    ):
        """_run_script calls generate_script once per approved narrative."""
        from autopilot.orchestrator import _run_script

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"}, {"narrative_id": "n2"},
        ]
        mock_script.generate_script.return_value = {"scenes": []}

        _run_script(config=minimal_config, db=db)

        assert mock_script.generate_script.call_count == 2

    @patch("autopilot.orchestrator.script")
    def test_script_continues_on_per_narrative_error(
        self, mock_script, minimal_config
    ):
        """First narrative raises ScriptError, second still gets generated."""
        from autopilot.orchestrator import _run_script
        from autopilot.plan.script import ScriptError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"}, {"narrative_id": "n2"},
        ]
        mock_script.generate_script.side_effect = [
            ScriptError("failed"), {"scenes": []},
        ]

        _run_script(config=minimal_config, db=db)

        assert mock_script.generate_script.call_count == 2

    @patch("autopilot.orchestrator.script")
    def test_script_raises_if_all_narratives_fail(
        self, mock_script, minimal_config
    ):
        """If every narrative fails, stage itself raises RuntimeError."""
        from autopilot.orchestrator import _run_script
        from autopilot.plan.script import ScriptError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"}, {"narrative_id": "n2"},
        ]
        mock_script.generate_script.side_effect = ScriptError("fail")

        with pytest.raises(RuntimeError, match="All narratives failed"):
            _run_script(config=minimal_config, db=db)


class TestEdlStage:
    """Tests for the real _run_edl stage function."""

    @patch("autopilot.orchestrator.otio_export")
    @patch("autopilot.orchestrator.validator")
    @patch("autopilot.orchestrator.edl_mod")
    def test_edl_generates_validates_exports_per_narrative(
        self, mock_edl, mock_validator, mock_otio, minimal_config
    ):
        """_run_edl calls generate_edl, validate_edl, export_otio for each narrative."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_narrative_script.return_value = {"scenes": []}
        mock_edl.generate_edl.return_value = {"timeline": []}
        mock_validator.validate_edl.return_value = MagicMock(passed=True)
        mock_otio.export_otio.return_value = Path("/out/timeline.otio")

        _run_edl(config=minimal_config, db=db)

        mock_edl.generate_edl.assert_called_once()
        mock_validator.validate_edl.assert_called_once()
        mock_otio.export_otio.assert_called_once()

    @patch("autopilot.orchestrator.otio_export")
    @patch("autopilot.orchestrator.validator")
    @patch("autopilot.orchestrator.edl_mod")
    def test_edl_stores_validation_result(
        self, mock_edl, mock_validator, mock_otio, minimal_config
    ):
        """_run_edl stores validation result via db.upsert_edit_plan."""
        from autopilot.orchestrator import _run_edl

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_narrative_script.return_value = {"scenes": []}
        mock_edl.generate_edl.return_value = {"timeline": []}
        val_result = MagicMock(passed=True)
        mock_validator.validate_edl.return_value = val_result
        mock_otio.export_otio.return_value = Path("/out/timeline.otio")

        _run_edl(config=minimal_config, db=db)

        db.upsert_edit_plan.assert_called_once()

    @patch("autopilot.orchestrator.otio_export")
    @patch("autopilot.orchestrator.validator")
    @patch("autopilot.orchestrator.edl_mod")
    def test_edl_continues_on_failure(
        self, mock_edl, mock_validator, mock_otio, minimal_config
    ):
        """One narrative fails EDL, others still processed."""
        from autopilot.orchestrator import _run_edl
        from autopilot.plan.edl import EdlError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"}, {"narrative_id": "n2"},
        ]
        db.get_narrative_script.return_value = {"scenes": []}
        mock_edl.generate_edl.side_effect = [
            EdlError("fail"), {"timeline": []},
        ]
        mock_validator.validate_edl.return_value = MagicMock(passed=True)
        mock_otio.export_otio.return_value = Path("/out/timeline.otio")

        _run_edl(config=minimal_config, db=db)

        # Second narrative should still be processed
        assert mock_edl.generate_edl.call_count == 2


class TestSourceStage:
    """Tests for the real _run_source_assets stage function."""

    @patch("autopilot.orchestrator.resolve")
    def test_source_resolves_assets_per_narrative(
        self, mock_resolve, minimal_config
    ):
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
        assert call_kwargs[1].get("narrative_id") == "n1" or (
            len(call_kwargs[0]) >= 1  # positional args present
        )


class TestRenderStage:
    """Tests for the real _run_render stage function."""

    @patch("autopilot.orchestrator.render_validate")
    @patch("autopilot.orchestrator.router")
    def test_render_routes_and_validates_per_narrative(
        self, mock_router, mock_validate, minimal_config
    ):
        """_run_render calls route_and_render and validate_render per narrative."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1", "edl_json": '{"timeline": []}',
        }
        mock_router.route_and_render.return_value = Path("/out/video.mp4")
        mock_validate.validate_render.return_value = MagicMock(
            passed=True, issues=[]
        )

        _run_render(config=minimal_config, db=db)

        mock_router.route_and_render.assert_called_once()
        mock_validate.validate_render.assert_called_once()

    @patch("autopilot.orchestrator.render_validate")
    @patch("autopilot.orchestrator.router")
    def test_render_logs_validation_warnings(
        self, mock_router, mock_validate, minimal_config, caplog
    ):
        """Validation issues are logged."""
        from autopilot.orchestrator import _run_render

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1", "edl_json": '{"timeline": []}',
        }
        mock_router.route_and_render.return_value = Path("/out/video.mp4")
        issue = MagicMock()
        issue.severity = "warning"
        issue.message = "Low bitrate"
        mock_validate.validate_render.return_value = MagicMock(
            passed=True, issues=[issue]
        )

        with caplog.at_level(logging.WARNING, logger="autopilot.orchestrator"):
            _run_render(config=minimal_config, db=db)

        assert any("Low bitrate" in r.message for r in caplog.records)

    @patch("autopilot.orchestrator.render_validate")
    @patch("autopilot.orchestrator.router")
    def test_render_continues_on_failure(
        self, mock_router, mock_validate, minimal_config
    ):
        """One narrative's render fails, others still processed."""
        from autopilot.orchestrator import _run_render
        from autopilot.render.router import RoutingError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"}, {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1", "edl_json": '{"timeline": []}',
        }
        mock_router.route_and_render.side_effect = [
            RoutingError("fail"), Path("/out/video.mp4"),
        ]
        mock_validate.validate_render.return_value = MagicMock(
            passed=True, issues=[]
        )

        _run_render(config=minimal_config, db=db)

        assert mock_router.route_and_render.call_count == 2


class TestUploadStage:
    """Tests for the real _run_upload stage function."""

    @patch("autopilot.orchestrator.thumbnail")
    @patch("autopilot.orchestrator.youtube")
    def test_upload_uploads_and_thumbnails_per_narrative(
        self, mock_youtube, mock_thumbnail, minimal_config
    ):
        """_run_upload calls upload_video and extract_best_thumbnail per narrative."""
        from autopilot.orchestrator import _run_upload

        db = MagicMock()
        db.list_narratives.return_value = [{"narrative_id": "n1"}]
        db.get_edit_plan.return_value = {
            "narrative_id": "n1",
            "edl_json": '{}',
        }
        # Simulate a render output path convention
        render_dir = minimal_config.output_dir / "renders" / "n1"
        render_dir.mkdir(parents=True)
        video_file = render_dir / "output.mp4"
        video_file.write_text("fake")

        mock_youtube.upload_video.return_value = "https://youtu.be/abc"
        mock_thumbnail.extract_best_thumbnail.return_value = Path("/thumb.jpg")

        _run_upload(config=minimal_config, db=db)

        mock_youtube.upload_video.assert_called_once()
        mock_thumbnail.extract_best_thumbnail.assert_called_once()

    @patch("autopilot.orchestrator.thumbnail")
    @patch("autopilot.orchestrator.youtube")
    def test_upload_continues_on_failure(
        self, mock_youtube, mock_thumbnail, minimal_config
    ):
        """One narrative fails upload, others still uploaded."""
        from autopilot.orchestrator import _run_upload
        from autopilot.upload.youtube import UploadError

        db = MagicMock()
        db.list_narratives.return_value = [
            {"narrative_id": "n1"}, {"narrative_id": "n2"},
        ]
        db.get_edit_plan.return_value = {"edl_json": '{}'}

        for nid in ["n1", "n2"]:
            render_dir = minimal_config.output_dir / "renders" / nid
            render_dir.mkdir(parents=True)
            (render_dir / "output.mp4").write_text("fake")

        mock_youtube.upload_video.side_effect = [
            UploadError("fail"), "https://youtu.be/def",
        ]
        mock_thumbnail.extract_best_thumbnail.return_value = Path("/thumb.jpg")

        _run_upload(config=minimal_config, db=db)

        assert mock_youtube.upload_video.call_count == 2


class TestRealStageRegistration:
    """Tests for PipelineOrchestrator registering real stage functions."""

    def test_orchestrator_registers_real_functions_not_stubs(self):
        """Each stage's func is not a stub."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            func = getattr(stage.func, "func", stage.func)
            name = getattr(func, "__name__", "")
            assert "stub" not in name.lower(), (
                f"{stage.name} still uses a stub function"
            )

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

    def test_run_logs_cumulative_elapsed_per_stage(
        self, caplog: pytest.LogCaptureFixture
    ):
        """After each stage, log shows cumulative time and remaining budget."""
        orch = PipelineOrchestrator(budget_seconds=3600)
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        progress_lines = [
            r.message for r in caplog.records if "[PROGRESS]" in r.message
        ]
        # Should have one progress line per stage
        assert len(progress_lines) == 9
        # Each should contain budget info
        for line in progress_lines:
            assert "budget" in line.lower() or "%" in line

    def test_run_logs_summary_table_at_end(
        self, caplog: pytest.LogCaptureFixture
    ):
        """Final log message includes all stage names with their elapsed times."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            stage.func = MagicMock()

        with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
            orch.run(config=MagicMock(), db=MagicMock())

        # Look for summary that mentions all stage names
        all_text = " ".join(r.message for r in caplog.records)
        for stage_name in ["INGEST", "ANALYZE", "CLASSIFY", "NARRATE",
                           "SCRIPT", "EDL", "SOURCE_ASSETS", "RENDER", "UPLOAD"]:
            assert stage_name in all_text
