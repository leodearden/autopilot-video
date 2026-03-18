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

        total_elapsed = time.monotonic() - pipeline_start
        logger.info("Pipeline complete (%.1fs)", total_elapsed)

        # Check budget
        if self.budget_seconds is not None and total_elapsed > self.budget_seconds:
            logger.warning(
                "Pipeline exceeded budget: %.1fs elapsed vs %.1fs budget",
                total_elapsed,
                self.budget_seconds,
            )

        return results
