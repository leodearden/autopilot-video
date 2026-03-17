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
