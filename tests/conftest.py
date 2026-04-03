"""Shared test fixtures and helpers for autopilot-video."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import pytest

if TYPE_CHECKING:
    from autopilot.db import CatalogDB


# ---------------------------------------------------------------------------
# Ensure the project's .venv site-packages is on sys.path so that tests
# can find project dependencies (fastapi, starlette, etc.) even when pytest
# is invoked from an external Python interpreter (e.g. the orchestrator).
#
# In a git worktree the .venv lives in the main repo, not in the worktree
# checkout.  We detect this by reading the .git file (worktrees have a file,
# not a directory) and resolving back to the main repo root.
# ---------------------------------------------------------------------------
def _resolve_project_root(worktree_root: Path) -> Path:
    """Resolve the main project root from a git worktree checkout.

    In a git worktree the ``.git`` entry is a file (not a directory) containing
    ``gitdir: <path>``.  This function reads that file and walks up from the
    gitdir to find the main repo root.

    Falls back to *worktree_root* unchanged when:
    - ``.git`` is a directory (normal, non-worktree repo)
    - ``.git`` is missing or unreadable (OSError)
    - ``.git`` content doesn't start with ``gitdir:``
    """
    dot_git = worktree_root / ".git"
    if not dot_git.is_file():
        return worktree_root
    try:
        gitdir_line = dot_git.read_text().strip()
    except OSError:
        return worktree_root
    if not gitdir_line.startswith("gitdir:"):
        return worktree_root
    gitdir_rel = Path(gitdir_line.split(":", 1)[1].strip())
    # Resolve relative paths against worktree_root (not CWD)
    if gitdir_rel.is_absolute():
        gitdir = gitdir_rel.resolve()
    else:
        gitdir = (worktree_root / gitdir_rel).resolve()
    # gitdir is e.g. /repo/.git/worktrees/45 — main repo is three parents up
    return gitdir.parent.parent.parent


_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _resolve_project_root(_WORKTREE_ROOT)

_PY_VER = f"python{sys.version_info.major}.{sys.version_info.minor}"
_VENV_SP = _PROJECT_ROOT / ".venv" / "lib" / _PY_VER / "site-packages"
if _VENV_SP.is_dir() and str(_VENV_SP) not in sys.path:
    sys.path.insert(0, str(_VENV_SP))


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def prompts_dir(project_root: Path) -> Path:
    """Return the prompts directory."""
    return project_root / "autopilot" / "prompts"


@pytest.fixture
def catalog_db() -> Generator:
    """Create an in-memory CatalogDB instance for testing.

    Yields a fresh CatalogDB connected to ':memory:' with autocommit enabled
    so that CRUD calls are immediately visible without explicit commit or
    context manager usage.  This convenience mode is for unit tests only —
    production code should use ``with db:`` blocks for transactional control.
    """
    from autopilot.db import CatalogDB

    db = CatalogDB(":memory:")
    db.conn.isolation_level = None  # autocommit for test convenience
    yield db
    db.close()


@pytest.fixture
def minimal_config(tmp_path: Path):
    """Create an AutopilotConfig with tmp_path-based directories."""
    from autopilot.config import AutopilotConfig

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return AutopilotConfig(input_dir=input_dir, output_dir=output_dir)


@pytest.fixture
def mock_gpu_scheduler():
    """Provide a no-op GPUScheduler-like context manager.

    Returns a MagicMock that supports the GPUScheduler interface:
    - .model(name) returns a context manager yielding a MagicMock
    - .register(name, spec) is a no-op
    - .device returns 0
    """
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    scheduler = MagicMock()
    scheduler.device = 0

    @contextmanager
    def _model(name: str):
        yield MagicMock()

    scheduler.model = _model
    return scheduler


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PIPELINE_STAGES = (
    "ingest", "analyze", "classify", "narrate", "script",
    "edl", "source", "render", "upload",
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def parse_sse_body(text: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts.

    Each event has 'id', 'event', and 'data' keys extracted from
    the SSE wire format.  Multiple ``data:`` lines within a single
    event are concatenated with newlines per the SSE specification.
    """
    events: list[dict] = []
    current: dict = {}
    for line in text.splitlines():
        if line.startswith("id:"):
            current["id"] = line[3:].strip()
        elif line.startswith("event:"):
            current["event"] = line[6:].strip()
        elif line.startswith("data:"):
            chunk = line[5:].strip()
            if "data" in current:
                current["data"] += "\n" + chunk
            else:
                current["data"] = chunk
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


# Keep the old underscore-prefixed name as an alias for back-compat within
# test files that already import it.
_parse_sse_body = parse_sse_body


def seed_narrative(
    db: CatalogDB,
    narrative_id: str = "n-1",
    *,
    commit: bool = True,
    **overrides: object,
) -> None:
    """Insert a narrative with sensible defaults, overridable via kwargs.

    Parameters
    ----------
    db:
        CatalogDB instance (must already have the narratives table).
    narrative_id:
        Primary key for the narrative row.
    commit:
        If *True* (default), call ``db.conn.commit()`` after the insert.
        Pass *False* when the caller manages its own transaction.
    **overrides:
        Any column keyword accepted by ``db.insert_narrative()``.
    """
    defaults: dict[str, object] = {
        "title": "Morning Walk",
        "description": "A walk in the park",
        "proposed_duration_seconds": 120.0,
        "activity_cluster_ids_json": '["c-1","c-2"]',
        "arc_notes": "peaceful start",
        "emotional_journey": "calm → happy",
        "status": "proposed",
    }
    defaults.update(overrides)
    db.insert_narrative(narrative_id, **defaults)  # type: ignore[arg-type]
    if commit:
        db.conn.commit()


_seed_narrative = seed_narrative


def seed_cluster(
    db: CatalogDB,
    cluster_id: str = "c-1",
    *,
    commit: bool = True,
    **overrides: object,
) -> None:
    """Insert an activity cluster with sensible defaults, overridable via kwargs.

    Parameters
    ----------
    db:
        CatalogDB instance.
    cluster_id:
        Primary key for the cluster row.
    commit:
        If *True* (default), call ``db.conn.commit()`` after the insert.
    **overrides:
        Any column keyword accepted by ``db.insert_activity_cluster()``.
    """
    defaults: dict[str, object] = {
        "label": "Morning Activity",
        "description": "Walking the dog",
        "time_start": "2025-01-01T08:00:00",
        "time_end": "2025-01-01T09:00:00",
        "location_label": "Park",
        "gps_center_lat": 37.7749,
        "gps_center_lon": -122.4194,
        "clip_ids_json": '["clip-1","clip-2"]',
    }
    defaults.update(overrides)
    db.insert_activity_cluster(cluster_id, **defaults)  # type: ignore[arg-type]
    if commit:
        db.conn.commit()


_seed_cluster = seed_cluster


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def extract_json_blocks(text: str) -> list:
    """Extract and parse all fenced ```json code blocks from markdown text.

    Finds all occurrences of ```json ... ``` in the given text,
    parses each as JSON, and returns a list of parsed objects.
    """
    pattern = r"```json\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for match in matches:
        parsed = json.loads(match.strip())
        results.append(parsed)
    return results
