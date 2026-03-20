"""Shared test fixtures and helpers for autopilot-video."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Generator

import pytest


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
