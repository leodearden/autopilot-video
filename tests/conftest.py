"""Shared test fixtures for autopilot-video."""

from __future__ import annotations

import pathlib
from typing import Generator

import pytest


@pytest.fixture
def project_root() -> pathlib.Path:
    """Return the project root directory."""
    return pathlib.Path(__file__).resolve().parent.parent


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
