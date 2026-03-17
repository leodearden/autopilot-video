"""Tests for the pipeline orchestrator (autopilot.orchestrator)."""

from __future__ import annotations

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
