"""Tests for the CLI entry point (autopilot.cli)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import click
import pytest
from click.testing import CliRunner

from autopilot.cli import main


class TestMainGroup:
    """Tests for the Click main group."""

    def test_main_group_exists(self) -> None:
        """main is a Click Group."""
        assert isinstance(main, click.Group)

    def test_main_help(self) -> None:
        """--help exits 0 and output contains 'autopilot'."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "autopilot" in result.output.lower()

    def test_main_has_config_option(self) -> None:
        """--config option exists in main.params."""
        param_names = [p.name for p in main.params]
        assert "config" in param_names
