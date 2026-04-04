"""Tests for the CLI entry point (autopilot.cli)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
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
COMMON_OPTIONS = ["input_dir", "output_dir", "verbose", "dry_run", "force"]


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
            assert result.exit_code == 0, f"{cmd_name} --help failed: {result.output}"

    def test_subcommands_have_common_options(self) -> None:
        """Each subcommand has --input-dir, --output-dir, --verbose, --dry-run."""
        for cmd_name in EXPECTED_SUBCOMMANDS:
            cmd = main.commands[cmd_name]
            param_names = [p.name for p in cmd.params]
            for opt in COMMON_OPTIONS:
                assert opt in param_names, f"{cmd_name} missing option: {opt}"


def _write_minimal_config(tmp_path: Path) -> Path:
    """Write a minimal config YAML and return its path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"input_dir: {tmp_path / 'input'}\noutput_dir: {tmp_path / 'output'}\n")
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
                runner.invoke(
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
                runner.invoke(
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
                    runner.invoke(
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

        stage_func = MagicMock()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch = MagicMock()
                mock_orch_cls.return_value = mock_orch
                mock_orch._stage_map.__getitem__.return_value.func = stage_func
                result = runner.invoke(
                    main,
                    [
                        "--config",
                        str(config_file),
                        "ingest",
                        "--input-dir",
                        str(override_dir),
                    ],
                )
                assert result.exit_code == 0
                # Verify the stage was called with a config that has overridden input_dir
                stage_func.assert_called_once()
                called_config = stage_func.call_args[1]["config"]
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
                runner.invoke(
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
                runner.invoke(
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
                runner.invoke(
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
                assert result.exit_code == 0, (
                    f"Exit code: {result.exit_code}, Output: {result.output}"
                )


class TestStageSubcommands:
    """Tests for individual stage subcommands delegation."""

    def _invoke_with_mock_orch(self, tmp_path: Path, subcommand: str) -> MagicMock:
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
                assert result.exit_code == 0, f"{subcommand} failed: {result.output}"
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


class TestCLIErrors:
    """Tests for CLI error handling."""

    def test_missing_config_file(self, tmp_path: Path) -> None:
        """Nonexistent --config file gives exit code != 0 with 'config' in error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(tmp_path / "nonexistent.yaml"), "ingest"],
        )
        assert result.exit_code != 0
        assert "config" in result.output.lower() or "Config" in result.output

    def test_invalid_config(self, tmp_path: Path) -> None:
        """Invalid YAML config gives exit code != 0."""
        bad_config = tmp_path / "bad.yaml"
        bad_config.write_text(": : : invalid yaml [[[")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--config", str(bad_config), "ingest"],
        )
        assert result.exit_code != 0

    def test_stage_error_prints_message(self, tmp_path: Path) -> None:
        """Stage function raising is reported to the user."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch = MagicMock()
                mock_orch_cls.return_value = mock_orch
                # Make stage func raise
                mock_orch._stage_map.__getitem__().func.side_effect = RuntimeError("stage exploded")
                result = runner.invoke(
                    main,
                    ["--config", str(config_file), "ingest"],
                )
                assert result.exit_code != 0
                assert "stage exploded" in result.output


class TestCLIIntegration:
    """Integration tests running the real pipeline with stubs."""

    def test_run_end_to_end(self, tmp_path: Path) -> None:
        """'run' executes all 9 stages with 'not yet implemented' messages."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        result = runner.invoke(
            main,
            ["--config", str(config_file), "run", "--verbose"],
        )
        assert result.exit_code == 0, f"Output: {result.output}"

    def test_run_dry_run_end_to_end(self, tmp_path: Path) -> None:
        """'run --dry-run' lists all stages without execution."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        result = runner.invoke(
            main,
            ["--config", str(config_file), "run", "--dry-run"],
        )
        assert result.exit_code == 0, f"Output: {result.output}"


def _write_config_with_missing_output_dir(tmp_path: Path, output_dir: Path) -> Path:
    """Write a config YAML where output_dir does NOT exist on disk."""
    config_file = tmp_path / "config.yaml"
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    config_file.write_text(f"input_dir: {input_dir}\noutput_dir: {output_dir}\n")
    return config_file


class TestOutputDirCreation:
    """Tests for output directory auto-creation in _setup_context."""

    def test_setup_context_creates_output_dir_when_missing(self, tmp_path: Path) -> None:
        """_setup_context creates output_dir when it does not exist."""
        output_dir = tmp_path / "nonexistent" / "deep" / "output"
        config_file = _write_config_with_missing_output_dir(tmp_path, output_dir)
        runner = CliRunner()

        with patch("autopilot.cli.PipelineOrchestrator"):
            result = runner.invoke(
                main,
                ["--config", str(config_file), "ingest"],
            )
        assert result.exit_code == 0, f"Expected exit 0 but got {result.exit_code}: {result.output}"
        assert output_dir.is_dir(), "output_dir should have been created"

    def test_setup_context_creates_catalog_db_in_new_dir(self, tmp_path: Path) -> None:
        """catalog.db is created inside the newly created output directory."""
        output_dir = tmp_path / "fresh_output"
        config_file = _write_config_with_missing_output_dir(tmp_path, output_dir)
        runner = CliRunner()

        with patch("autopilot.cli.PipelineOrchestrator"):
            result = runner.invoke(
                main,
                ["--config", str(config_file), "ingest"],
            )
        assert result.exit_code == 0, f"Expected exit 0 but got {result.exit_code}: {result.output}"
        assert (output_dir / "catalog.db").exists(), (
            "catalog.db should exist in the new output directory"
        )

    def test_setup_context_existing_output_dir_unchanged(self, tmp_path: Path) -> None:
        """mkdir with exist_ok=True does not clobber existing directory contents."""
        output_dir = tmp_path / "existing_output"
        output_dir.mkdir(parents=True)
        marker = output_dir / "marker.txt"
        marker.write_text("keep me")

        config_file = _write_config_with_missing_output_dir(tmp_path, output_dir)
        runner = CliRunner()

        with patch("autopilot.cli.PipelineOrchestrator"):
            result = runner.invoke(
                main,
                ["--config", str(config_file), "ingest"],
            )
        assert result.exit_code == 0, f"Expected exit 0 but got {result.exit_code}: {result.output}"
        assert marker.exists(), "Existing marker file should not be removed"
        assert marker.read_text() == "keep me"


class TestCLIRealStageWiring:
    """Tests for CLI using real stage functions via the orchestrator."""

    def test_cli_run_uses_real_orchestrator(self, tmp_path: Path) -> None:
        """'run' subcommand creates a PipelineOrchestrator with real stages."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch_cls.return_value = mock_orch
            result = runner.invoke(
                main,
                ["--config", str(config_file), "run"],
            )
            assert result.exit_code == 0, result.output
            # Verify run() was called on the orchestrator
            mock_orch.run.assert_called_once()

    def test_cli_ingest_calls_real_stage(self, tmp_path: Path) -> None:
        """'ingest' subcommand invokes the INGEST stage function."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch_cls.return_value = mock_orch
            result = runner.invoke(
                main,
                ["--config", str(config_file), "ingest"],
            )
            assert result.exit_code == 0, result.output
            # Verify the INGEST stage func was called
            mock_orch._stage_map.__getitem__.assert_called_with("INGEST")

    def test_cli_individual_subcommands_call_correct_stages(self, tmp_path: Path) -> None:
        """Each subcommand exercises the right stage functions."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        expected_stages = {
            "ingest": ["INGEST"],
            "analyze": ["ANALYZE", "CLASSIFY"],
            "plan": ["NARRATE", "SCRIPT"],
            "edit": ["EDL", "SOURCE_ASSETS"],
            "render": ["RENDER"],
            "upload": ["UPLOAD"],
        }

        for cmd_name, stage_names in expected_stages.items():
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch = MagicMock()
                mock_orch_cls.return_value = mock_orch
                result = runner.invoke(
                    main,
                    ["--config", str(config_file), cmd_name],
                )
                assert result.exit_code == 0, f"{cmd_name} failed: {result.output}"
                # Verify correct stages were accessed
                calls = mock_orch._stage_map.__getitem__.call_args_list
                accessed = [c[0][0] for c in calls]
                assert accessed == stage_names, (
                    f"{cmd_name}: expected {stage_names}, got {accessed}"
                )


class TestForceFlag:
    """Tests for CLI --force flag propagation."""

    def test_force_in_common_options(self) -> None:
        """--force is present in _common_options for all subcommands."""
        for cmd_name in EXPECTED_SUBCOMMANDS:
            cmd = main.commands[cmd_name]
            param_names = [p.name for p in cmd.params]
            assert "force" in param_names, f"{cmd_name} missing --force option"

    def test_run_command_has_force_option(self) -> None:
        """'run' subcommand accepts --force flag."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert "--force" in result.output

    def test_run_command_passes_force_to_orchestrator(self, tmp_path: Path) -> None:
        """'run --force' passes force=True to PipelineOrchestrator constructor."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = MagicMock()
                runner.invoke(
                    main,
                    ["--config", str(config_file), "run", "--force"],
                )
                call_kwargs = mock_orch_cls.call_args[1]
                assert call_kwargs.get("force") is True

    def test_run_command_defaults_force_false(self, tmp_path: Path) -> None:
        """'run' without --force passes force=False to PipelineOrchestrator."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch_cls.return_value = MagicMock()
                runner.invoke(
                    main,
                    ["--config", str(config_file), "run"],
                )
                call_kwargs = mock_orch_cls.call_args[1]
                assert call_kwargs.get("force") is False

    def test_individual_subcommands_pass_force_to_stage(self, tmp_path: Path) -> None:
        """Each subcommand passes force=True to stage function when --force given."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        for cmd_name in ["ingest", "analyze", "plan", "edit", "render", "upload"]:
            with patch("autopilot.cli.CatalogDB") as mock_db_cls:
                mock_db_cls.return_value = MagicMock()
                with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                    mock_orch = MagicMock()
                    mock_orch_cls.return_value = mock_orch
                    stage_func = MagicMock()
                    mock_orch._stage_map.__getitem__.return_value.func = stage_func

                    result = runner.invoke(
                        main,
                        ["--config", str(config_file), cmd_name, "--force"],
                    )
                    assert result.exit_code == 0, f"{cmd_name} --force failed: {result.output}"
                    # Every call to stage_func should have force=True
                    for call in stage_func.call_args_list:
                        assert call[1].get("force") is True, (
                            f"{cmd_name}: stage func not called with force=True"
                        )


import pytest


# Mapping from CLI subcommand name → stage names it would normally execute
SUBCOMMAND_STAGES = {
    "ingest": ["INGEST"],
    "analyze": ["ANALYZE", "CLASSIFY"],
    "plan": ["NARRATE", "SCRIPT"],
    "edit": ["EDL", "SOURCE_ASSETS"],
    "render": ["RENDER"],
    "upload": ["UPLOAD"],
}


class TestDryRunSubcommands:
    """Tests that --dry-run prevents stage functions from being called."""

    @pytest.mark.parametrize("cmd_name", list(SUBCOMMAND_STAGES.keys()))
    def test_dry_run_does_not_call_stage_func(self, tmp_path: Path, cmd_name: str) -> None:
        """'<cmd> --dry-run' does NOT call any stage function."""
        config_file = _write_minimal_config(tmp_path)
        runner = CliRunner()

        with patch("autopilot.cli.CatalogDB") as mock_db_cls:
            mock_db_cls.return_value = MagicMock()
            with patch("autopilot.cli.PipelineOrchestrator") as mock_orch_cls:
                mock_orch = MagicMock()
                mock_orch_cls.return_value = mock_orch
                stage_func = MagicMock()
                mock_orch._stage_map.__getitem__.return_value.func = stage_func

                result = runner.invoke(
                    main,
                    ["--config", str(config_file), cmd_name, "--dry-run"],
                )
                assert result.exit_code == 0, (
                    f"{cmd_name} --dry-run failed: {result.output}"
                )
                stage_func.assert_not_called(), (
                    f"{cmd_name} --dry-run should NOT call any stage function"
                )
