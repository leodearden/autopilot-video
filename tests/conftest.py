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

    Yields a fresh CatalogDB connected to ':memory:' and closes it after test.
    """
    from autopilot.db import CatalogDB

    db = CatalogDB(":memory:")
    yield db
    db.close()
