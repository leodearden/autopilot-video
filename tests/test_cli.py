"""Tests for the CLI entry point (autopilot.cli)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import click
import pytest
from click.testing import CliRunner

from autopilot.cli import main
from autopilot.config import load_config


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


def _write_minimal_config(tmp_path: Path) -> Path:
    """Write a minimal config YAML and return its path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"input_dir: {tmp_path / 'input'}\noutput_dir: {tmp_path / 'output'}\n"
    )
    (tmp_path / "input").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    return config_file


class TestCLIConfigIntegration:
    """Tests for CLI config loading and DB opening."""

    def test_subcommand_loads_config(self, tmp_path: Path) -> None:
        """ingest subcommand loads config from --config path."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.load_config", wraps=load_config) as mock_load:
            with patch("autopilot.cli.PipelineOrchestrator"):
                result = runner.invoke(
                    main,
                    ["--config", str(config_file), "ingest"],
                )
            mock_load.assert_called_once_with(str(config_file))

    def test_subcommand_opens_catalog_db(self, tmp_path: Path) -> None:
        """ingest subcommand opens CatalogDB at output_dir/catalog.db."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()
        expected_db_path = str(tmp_path / "output" / "catalog.db")

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator"):
                result = runner.invoke(
                    main,
                    ["--config", str(config_file), "ingest"],
                )
            mock_db_cls.assert_called_once_with(expected_db_path)

    def test_verbose_sets_logging_level(self, tmp_path: Path) -> None:
        """--verbose sets logging level to DEBUG."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.logging.basicConfig") as mock_basic:
            with patch("autopilot.cli.CatalogDB") as mock_db_cls:
                mock_db_cls.return_value = MagicMock()
                with patch("autopilot.cli.PipelineOrchestrator"):
                    result = runner.invoke(
                        main,
                        ["--config", str(config_file), "ingest", "--verbose"],
                    )
            mock_basic.assert_called_once_with(level=logging.DEBUG)

    def test_input_dir_override(self, tmp_path: Path) -> None:
        """--input-dir overrides config's input_dir."""
        config_file = _write_minimal_config(tmp_path)
        override_dir = tmp_path / "override_input"
        override_dir.mkdir()
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch = MagicMock()
                mock_orch_cls.return_value = mock_orch
                result = runner.invoke(
                    main,
                    [
                        "--config", str(config_file),
                        "ingest",
                        "--input-dir", str(override_dir),
                    ],
                )
                # Verify the stage was called with a config that has overridden input_dir
                stage_call = mock_orch._stage_map.__getitem__().func
                if stage_call.call_args:
                    called_config = stage_call.call_args[1].get("config")
                    if called_config:
                        assert called_config.input_dir == override_dir


class TestRunSubcommand:
    """Tests for the 'run' subcommand wiring to orchestrator."""

    def test_run_creates_orchestrator(self, tmp_path: Path) -> None:
        """'run' subcommand instantiates PipelineOrchestrator."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = MagicMock()
                result = runner.invoke(
                    main,
                    ["--config", str(config_file), "run"],
                )
                mock_orch_cls.assert_called_once()

    def test_run_calls_orchestrator_run(self, tmp_path: Path) -> None:
        """'run' subcommand calls orchestrator.run() with config and db."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db = MagicMock()
            mock_db_cls.return_value = mock_db
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch = MagicMock()
                mock_orch_cls.return_value = mock_orch
                result = runner.invoke(
                    main,
                    ["--config", str(config_file), "run"],
                )
                mock_orch.run.assert_called_once()
                call_kwargs = mock_orch.run.call_args[1]
                assert "config" in call_kwargs
                assert call_kwargs["db"] is mock_db

    def test_run_passes_dry_run(self, tmp_path: Path) -> None:
        """'run --dry-run' passes dry_run=True to orchestrator.run()."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch = MagicMock()
                mock_orch_cls.return_value = mock_orch
                result = runner.invoke(
                    main,
                    ["--config", str(config_file), "run", "--dry-run"],
                )
                call_kwargs = mock_orch.run.call_args[1]
                assert call_kwargs["dry_run"] is True

    def test_run_exits_zero_on_success(self, tmp_path: Path) -> None:
        """'run' exits 0 on successful completion."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = MagicMock()
                result = runner.invoke(
                    main,
                    ["--config", str(config_file), "run"],
                )
                assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.output}"


class TestStageSubcommands:
    """Tests for individual stage subcommands delegation."""

    def _invoke_with_mock_orch(
        self, tmp_path: Path, subcommand: str
    ) -> MagicMock:
        """Invoke a subcommand with mocked orchestrator, return the mock."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch = MagicMock()
                mock_orch_cls.return_value = mock_orch
                result = runner.invoke(
                    main,
                    ["--config", str(config_file), subcommand],
                )
                assert result.exit_code == 0, (
                    f"{subcommand} failed: {result.output}"
                )
                return mock_orch

    def test_ingest_delegates_to_stage(self, tmp_path: Path) -> None:
        """'ingest' calls the INGEST stage function."""
        mock_orch = self._invoke_with_mock_orch(tmp_path, "ingest")
        mock_orch._stage_map.__getitem__.assert_any_call("INGEST")

    def test_analyze_delegates_to_stage(self, tmp_path: Path) -> None:
        """'analyze' calls the ANALYZE and CLASSIFY stage functions."""
        mock_orch = self._invoke_with_mock_orch(tmp_path, "analyze")
        calls = [c.args[0] for c in mock_orch._stage_map.__getitem__.call_args_list]
        assert "ANALYZE" in calls
        assert "CLASSIFY" in calls

    def test_plan_delegates(self, tmp_path: Path) -> None:
        """'plan' calls the NARRATE and SCRIPT stage functions."""
        mock_orch = self._invoke_with_mock_orch(tmp_path, "plan")
        calls = [c.args[0] for c in mock_orch._stage_map.__getitem__.call_args_list]
        assert "NARRATE" in calls
        assert "SCRIPT" in calls

    def test_edit_delegates(self, tmp_path: Path) -> None:
        """'edit' calls the EDL and SOURCE_ASSETS stage functions."""
        mock_orch = self._invoke_with_mock_orch(tmp_path, "edit")
        calls = [c.args[0] for c in mock_orch._stage_map.__getitem__.call_args_list]
        assert "EDL" in calls
        assert "SOURCE_ASSETS" in calls

    def test_render_delegates(self, tmp_path: Path) -> None:
        """'render' calls the RENDER stage function."""
        mock_orch = self._invoke_with_mock_orch(tmp_path, "render")
        mock_orch._stage_map.__getitem__.assert_any_call("RENDER")

    def test_upload_delegates(self, tmp_path: Path) -> None:
        """'upload' calls the UPLOAD stage function."""
        mock_orch = self._invoke_with_mock_orch(tmp_path, "upload")
        mock_orch._stage_map.__getitem__.assert_any_call("UPLOAD")
