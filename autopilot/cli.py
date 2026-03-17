"""Click-based CLI entry point for autopilot-video."""

from __future__ import annotations

import logging
import sys
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
