"""End-to-end integration tests for the full web console.

These tests verify cross-cutting workflows and data consistency across
multiple web console views. Individual endpoint tests live in their
respective test files (test_dashboard.py, test_gates.py, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app

PIPELINE_STAGES = (
    "ingest", "analyze", "classify", "narrate", "script",
    "edl", "source", "render", "upload",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_db_path(tmp_path: Path) -> str:
    """Return the path for a test catalog DB file."""
    return str(tmp_path / "catalog.db")


@pytest.fixture
def e2e_db(e2e_db_path: str) -> CatalogDB:
    """Create a CatalogDB backed by a real file for E2E tests."""
    db = CatalogDB(e2e_db_path)
    db.conn.isolation_level = None  # autocommit
    return db


@pytest.fixture
def e2e_app(e2e_db_path: str) -> FastAPI:
    """Create a FastAPI app pointing at the E2E DB."""
    return create_app(e2e_db_path)


@pytest.fixture
def e2e_client(e2e_app: FastAPI) -> TestClient:
    """Create a TestClient for E2E tests."""
    return TestClient(e2e_app)


# ---------------------------------------------------------------------------
# SSE parsing helper
# ---------------------------------------------------------------------------


def _parse_sse_body(text: str) -> list[dict]:
    """Parse SSE response text into a list of event dicts.

    Each event has 'id', 'event', and 'data' keys extracted from
    the SSE wire format.
    """
    events: list[dict] = []
    current: dict = {}
    for line in text.splitlines():
        if line.startswith("id:"):
            current["id"] = line[3:].strip()
        elif line.startswith("event:"):
            current["event"] = line[6:].strip()
        elif line.startswith("data:"):
            current["data"] = line[5:].strip()
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events
