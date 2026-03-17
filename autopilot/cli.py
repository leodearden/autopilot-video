"""Click-based CLI entry point for autopilot-video."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click

from autopilot.config import AutopilotConfig, ConfigError, load_config
from autopilot.db import CatalogDB
from autopilot.orchestrator import PipelineOrchestrator

__all__ = ["main"]

logger = logging.getLogger(__name__)


@click.group(name="autopilot")
@click.option(
    "--config",
    type=click.Path(),
    default="config.yaml",
    help="Path to config YAML.",
)
@click.pass_context
def main(ctx: click.Context, config: str) -> None:
    """Autopilot Video — automated video editing pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


def _common_options(f: Any) -> Any:
    """Decorator that adds common options shared by all subcommands."""
    f = click.option(
        "--input-dir", type=click.Path(), default=None, help="Override input directory."
    )(f)
    f = click.option(
        "--output-dir", type=click.Path(), default=None, help="Override output directory."
    )(f)
    f = click.option("--verbose", is_flag=True, help="Enable debug logging.")(f)
    f = click.option("--dry-run", is_flag=True, help="Show what would run without executing.")(f)
    return f


def _setup_context(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
) -> tuple[AutopilotConfig, CatalogDB]:
    """Load config and open DB, applying CLI overrides.

    Returns:
        (config, db) tuple ready for stage functions.

    Raises:
        click.ClickException: On config errors.
    """
    config_path = ctx.obj["config_path"]

    try:
        config = load_config(config_path)
    except ConfigError as e:
        raise click.ClickException(str(e)) from e

    if input_dir is not None:
        config.input_dir = Path(input_dir)
    if output_dir is not None:
        config.output_dir = Path(output_dir)

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = config.output_dir / "catalog.db"
    db = CatalogDB(str(db_path))
    return config, db


@main.command()
@_common_options
@click.pass_context
def ingest(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Run the ingest stage."""
    config, db = _setup_context(ctx, input_dir, output_dir, verbose)
    try:
        orch = PipelineOrchestrator()
        orch._stage_map["INGEST"].func(config=config, db=db)
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        db.close()


@main.command()
@_common_options
@click.pass_context
def analyze(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Run the analyze and classify stages."""
    config, db = _setup_context(ctx, input_dir, output_dir, verbose)
    try:
        orch = PipelineOrchestrator()
        orch._stage_map["ANALYZE"].func(config=config, db=db)
        orch._stage_map["CLASSIFY"].func(config=config, db=db)
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        db.close()


@main.command()
@_common_options
@click.pass_context
def plan(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Run the narrate and script stages."""
    config, db = _setup_context(ctx, input_dir, output_dir, verbose)
    try:
        orch = PipelineOrchestrator()
        orch._stage_map["NARRATE"].func(config=config, db=db)
        orch._stage_map["SCRIPT"].func(config=config, db=db)
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        db.close()


@main.command()
@_common_options
@click.pass_context
def edit(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Run the EDL and source_assets stages."""
    config, db = _setup_context(ctx, input_dir, output_dir, verbose)
    try:
        orch = PipelineOrchestrator()
        orch._stage_map["EDL"].func(config=config, db=db)
        orch._stage_map["SOURCE_ASSETS"].func(config=config, db=db)
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        db.close()


@main.command()
@_common_options
@click.pass_context
def render(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Run the render stage."""
    config, db = _setup_context(ctx, input_dir, output_dir, verbose)
    try:
        orch = PipelineOrchestrator()
        orch._stage_map["RENDER"].func(config=config, db=db)
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        db.close()


@main.command()
@_common_options
@click.pass_context
def upload(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Run the upload stage."""
    config, db = _setup_context(ctx, input_dir, output_dir, verbose)
    try:
        orch = PipelineOrchestrator()
        orch._stage_map["UPLOAD"].func(config=config, db=db)
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        db.close()


@main.command()
@_common_options
@click.pass_context
def run(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
    dry_run: bool,
) -> None:
    """Run the full pipeline (all stages)."""
    config, db = _setup_context(ctx, input_dir, output_dir, verbose)
    try:
        orch = PipelineOrchestrator(
            budget_seconds=config.processing.max_wall_clock_hours * 3600,
        )
        orch.run(config=config, db=db, dry_run=dry_run)
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        db.close()
