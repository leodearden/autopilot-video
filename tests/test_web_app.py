"""Tests for the FastAPI web application skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_create_app_returns_fastapi_instance(self, tmp_path: Path) -> None:
        """create_app(db_path) returns a FastAPI instance."""
        from autopilot.web.app import create_app

        db_path = str(tmp_path / "catalog.db")
        app = create_app(db_path)
        assert isinstance(app, FastAPI)
