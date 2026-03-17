"""Pipeline orchestrator — DAG-based stage execution for autopilot-video."""

from __future__ import annotations

import enum
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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


def _stage_stub(name: str) -> Callable[..., Any]:
    """Create a stub function for an unimplemented stage."""

    def _stub(**kwargs: Any) -> None:
        logger.info("Stage %s not yet implemented", name)

    _stub.__name__ = f"_stage_{name.lower()}"
    _stub.__qualname__ = f"_stage_{name.lower()}"
    return _stub


class PipelineOrchestrator:
    """Manages a DAG of pipeline stages and executes them in topological order."""

    def __init__(self, budget_seconds: float | None = None) -> None:
        self.budget_seconds = budget_seconds
        self.stages: list[StageDefinition] = [
            StageDefinition(
                name="INGEST",
                func=_stage_stub("INGEST"),
                dependencies=[],
                estimated_seconds=600,
            ),
            StageDefinition(
                name="ANALYZE",
                func=_stage_stub("ANALYZE"),
                dependencies=["INGEST"],
                estimated_seconds=1800,
            ),
            StageDefinition(
                name="CLASSIFY",
                func=_stage_stub("CLASSIFY"),
                dependencies=["ANALYZE"],
                estimated_seconds=900,
            ),
            StageDefinition(
                name="NARRATE",
                func=_stage_stub("NARRATE"),
                dependencies=["CLASSIFY"],
                estimated_seconds=300,
            ),
            StageDefinition(
                name="SCRIPT",
                func=_stage_stub("SCRIPT"),
                dependencies=["NARRATE"],
                estimated_seconds=300,
            ),
            StageDefinition(
                name="EDL",
                func=_stage_stub("EDL"),
                dependencies=["SCRIPT"],
                estimated_seconds=120,
            ),
            StageDefinition(
                name="SOURCE_ASSETS",
                func=_stage_stub("SOURCE_ASSETS"),
                dependencies=["EDL"],
                estimated_seconds=1200,
            ),
            StageDefinition(
                name="RENDER",
                func=_stage_stub("RENDER"),
                dependencies=["EDL", "SOURCE_ASSETS"],
                estimated_seconds=3600,
            ),
            StageDefinition(
                name="UPLOAD",
                func=_stage_stub("UPLOAD"),
                dependencies=["RENDER"],
                estimated_seconds=600,
            ),
        ]
        self._stage_map: dict[str, StageDefinition] = {
            s.name: s for s in self.stages
        }
