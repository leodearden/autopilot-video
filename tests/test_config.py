"""Tests for autopilot.config — typed configuration dataclasses and loader."""

import dataclasses
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Step 1: Public API — all names importable and correct types
# ---------------------------------------------------------------------------


def test_all_dataclasses_importable() -> None:
    """All config dataclasses can be imported from autopilot.config."""
    from autopilot.config import (
        AutopilotConfig,
        CameraConfig,
        CreatorConfig,
        LLMConfig,
        ModelConfig,
        OutputConfig,
        ProcessingConfig,
        YouTubeConfig,
    )

    for cls in (
        CreatorConfig,
        CameraConfig,
        OutputConfig,
        ModelConfig,
        LLMConfig,
        YouTubeConfig,
        ProcessingConfig,
        AutopilotConfig,
    ):
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} must be a dataclass"


def test_config_error_is_exception() -> None:
    """ConfigError is an Exception subclass."""
    from autopilot.config import ConfigError

    assert issubclass(ConfigError, Exception)


def test_load_config_importable() -> None:
    """load_config is a callable."""
    from autopilot.config import load_config

    assert callable(load_config)
