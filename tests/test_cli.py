"""Tests for the CLI entry point (autopilot.cli)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from click.testing import CliRunner
