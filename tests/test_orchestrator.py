"""Tests for the pipeline orchestrator (autopilot.orchestrator)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from autopilot.orchestrator import StageDefinition, StageStatus


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
