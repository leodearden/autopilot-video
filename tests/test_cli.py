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


EXPECTED_SUBCOMMANDS = ["ingest", "analyze", "plan", "edit", "render", "upload", "run"]
COMMON_OPTIONS = ["input_dir", "output_dir", "verbose", "dry_run"]


class TestSubcommands:
    """Tests for CLI subcommand registration."""

    def test_all_subcommands_registered(self) -> None:
        """main group has all expected subcommands."""
        registered = list(main.commands.keys())
        for cmd_name in EXPECTED_SUBCOMMANDS:
            assert cmd_name in registered, f"Missing subcommand: {cmd_name}"

    def test_each_subcommand_help(self) -> None:
        """Each subcommand responds to --help with exit code 0."""
        runner = CliRunner()
        for cmd_name in EXPECTED_SUBCOMMANDS:
            result = runner.invoke(main, [cmd_name, "--help"])
            assert result.exit_code == 0, (
                f"{cmd_name} --help failed: {result.output}"
            )

    def test_subcommands_have_common_options(self) -> None:
        """Each subcommand has --input-dir, --output-dir, --verbose, --dry-run."""
        for cmd_name in EXPECTED_SUBCOMMANDS:
            cmd = main.commands[cmd_name]
            param_names = [p.name for p in cmd.params]
            for opt in COMMON_OPTIONS:
                assert opt in param_names, (
                    f"{cmd_name} missing option: {opt}"
                )
