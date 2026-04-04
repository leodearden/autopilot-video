"""Click-based CLI entry point for autopilot-video."""

from __future__ import annotations

import logging
import signal
from pathlib import Path
from typing import Any

import click

from autopilot.config import AutopilotConfig, ConfigError, load_config
from autopilot.db import CatalogDB
from autopilot.orchestrator import PipelineOrchestrator, request_shutdown

__all__ = ["main"]

logger = logging.getLogger(__name__)


@click.group(name="autopilot")
@click.option(
    "--config",
    type=click.Path(),
    default="config.yaml",
    help="Path to config YAML.",
)
@click.option(
    "--api-fallback",
    is_flag=True,
    default=False,
    help="Use Anthropic SDK instead of Claude CLI for LLM calls.",
)
@click.pass_context
def main(ctx: click.Context, config: str, api_fallback: bool) -> None:
    """Autopilot Video — automated video editing pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config
    ctx.obj["api_fallback"] = api_fallback


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
    f = click.option(
        "--force",
        is_flag=True,
        help="Bypass checkpoint/resume: reprocess all items.",
    )(f)
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
    if ctx.obj.get("api_fallback"):
        config.llm.use_api = True

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    db_path = config.output_dir / "catalog.db"
    db = CatalogDB(str(db_path))
    return config, db


def _handle_dry_run(dry_run: bool, stages: str) -> bool:
    """Check dry-run flag and echo the planned stages if active.

    Returns:
        True if dry-run is active (caller should return immediately),
        False otherwise.
    """
    if dry_run:
        click.echo(f"[DRY-RUN] Would execute: {stages}")
        return True
    return False


def _cli_human_review(formatted_text: str, narratives: list) -> list[str]:
    """Interactive CLI prompt for human review of proposed narratives.

    Displays the formatted review text and prompts the user to select
    which narratives to approve. Returns a list of approved narrative IDs.
    """
    click.echo("\n" + "=" * 60)
    click.echo("NARRATIVE REVIEW")
    click.echo("=" * 60)
    click.echo(formatted_text)
    click.echo("=" * 60)

    all_ids = [n.narrative_id for n in narratives]
    click.echo(f"\nNarrative IDs: {', '.join(all_ids)}")
    response = click.prompt(
        "Enter narrative IDs to approve (comma-separated, or 'all')",
        default="all",
    )

    if response.strip().lower() == "all":
        return all_ids

    return [nid.strip() for nid in response.split(",") if nid.strip()]


@main.command()
@_common_options
@click.pass_context
def ingest(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
    dry_run: bool,
    force: bool,
) -> None:
    """Run the ingest stage."""
    if _handle_dry_run(dry_run, "INGEST"):
        return
    db = None
    try:
        config, db = _setup_context(ctx, input_dir, output_dir, verbose)
        orch = PipelineOrchestrator()
        orch._stage_map["INGEST"].func(config=config, db=db, force=force)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        if db is not None:
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
    force: bool,
) -> None:
    """Run the analyze and classify stages."""
    if _handle_dry_run(dry_run, "ANALYZE, CLASSIFY"):
        return
    db = None
    try:
        config, db = _setup_context(ctx, input_dir, output_dir, verbose)
        orch = PipelineOrchestrator()
        orch._stage_map["ANALYZE"].func(config=config, db=db, force=force)
        orch._stage_map["CLASSIFY"].func(config=config, db=db, force=force)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        if db is not None:
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
    force: bool,
) -> None:
    """Run the narrate and script stages."""
    if _handle_dry_run(dry_run, "NARRATE, SCRIPT"):
        return
    db = None
    try:
        config, db = _setup_context(ctx, input_dir, output_dir, verbose)
        orch = PipelineOrchestrator()
        orch._stage_map["NARRATE"].func(config=config, db=db, force=force)
        orch._stage_map["SCRIPT"].func(config=config, db=db, force=force)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        if db is not None:
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
    force: bool,
) -> None:
    """Run the EDL and source_assets stages."""
    if _handle_dry_run(dry_run, "EDL, SOURCE_ASSETS"):
        return
    db = None
    try:
        config, db = _setup_context(ctx, input_dir, output_dir, verbose)
        orch = PipelineOrchestrator()
        orch._stage_map["EDL"].func(config=config, db=db, force=force)
        orch._stage_map["SOURCE_ASSETS"].func(config=config, db=db, force=force)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        if db is not None:
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
    force: bool,
) -> None:
    """Run the render stage."""
    if _handle_dry_run(dry_run, "RENDER"):
        return
    db = None
    try:
        config, db = _setup_context(ctx, input_dir, output_dir, verbose)
        orch = PipelineOrchestrator()
        orch._stage_map["RENDER"].func(config=config, db=db, force=force)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        if db is not None:
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
    force: bool,
) -> None:
    """Run the upload stage."""
    if _handle_dry_run(dry_run, "UPLOAD"):
        return
    db = None
    try:
        config, db = _setup_context(ctx, input_dir, output_dir, verbose)
        orch = PipelineOrchestrator()
        orch._stage_map["UPLOAD"].func(config=config, db=db, force=force)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        if db is not None:
            db.close()


@main.command()
@click.option("--host", default="127.0.0.1", help="Bind host address.")
@click.option("--port", default=8080, type=int, help="Bind port number.")
@click.option(
    "--output-dir",
    type=click.Path(),
    required=True,
    help="Project output directory (contains catalog.db).",
)
def serve(host: str, port: int, output_dir: str) -> None:
    """Start the web management console."""
    import uvicorn

    from autopilot.web.app import create_app

    db_path = str(Path(output_dir) / "catalog.db")
    app = create_app(db_path)
    uvicorn.run(app, host=host, port=port)


@main.command()
@_common_options
@click.pass_context
def run(
    ctx: click.Context,
    input_dir: str | None,
    output_dir: str | None,
    verbose: bool,
    dry_run: bool,
    force: bool,
) -> None:
    """Run the full pipeline (all stages)."""

    def _shutdown_handler(signum: int, frame: Any) -> None:
        logger.info("Shutdown requested, finishing current work...")
        request_shutdown()

    db = None
    try:
        config, db = _setup_context(ctx, input_dir, output_dir, verbose)

        # Register signal handlers for graceful shutdown
        prev_sigint = signal.signal(signal.SIGINT, _shutdown_handler)
        prev_sigterm = signal.signal(signal.SIGTERM, _shutdown_handler)

        orch = PipelineOrchestrator(
            budget_seconds=config.processing.max_wall_clock_hours * 3600,
            human_review_fn=_cli_human_review,
            force=force,
        )
        orch.run(config=config, db=db, dry_run=dry_run)

        # Restore original signal handlers
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(str(e)) from e
    finally:
        if db is not None:
            db.close()
