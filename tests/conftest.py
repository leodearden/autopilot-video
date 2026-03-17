"""Shared test fixtures for autopilot-video."""

import pathlib

import pytest


@pytest.fixture
def project_root() -> pathlib.Path:
    """Return the project root directory."""
    return pathlib.Path(__file__).resolve().parent.parent
