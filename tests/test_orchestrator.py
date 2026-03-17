"""Tests for the pipeline orchestrator (autopilot.orchestrator)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, call, patch

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
        func = lambda config, db: None
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
        stage = StageDefinition(name="TEST", func=lambda c, d: None)
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


class TestStageStubs:
    """Tests for stage stub functions."""

    def test_stub_functions_log_not_implemented(self, caplog: pytest.LogCaptureFixture) -> None:
        """Each stub logs 'not yet implemented' with the stage name."""
        orch = PipelineOrchestrator()
        for stage in orch.stages:
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="autopilot.orchestrator"):
                stage.func(config=MagicMock(), db=MagicMock())
            assert any("not yet implemented" in r.message for r in caplog.records), (
                f"{stage.name} stub did not log 'not yet implemented'"
            )
            assert any(stage.name in r.message for r in caplog.records), (
                f"{stage.name} stub did not log stage name"
            )

    def test_stub_functions_accept_config_and_db(self) -> None:
        """Each stub accepts config and db keyword arguments without error."""
        orch = PipelineOrchestrator()
        mock_config = MagicMock()
        mock_db = MagicMock()
        for stage in orch.stages:
            # Should not raise
            stage.func(config=mock_config, db=mock_db)


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
