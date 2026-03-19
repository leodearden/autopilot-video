"""Tests for shot boundary detection (autopilot.analyze.scenes)."""

from __future__ import annotations

import inspect
import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPublicAPI:
    """Tests for module-level public API imports."""

    def test_shot_detection_error_importable(self) -> None:
        """ShotDetectionError and detect_shots are importable from autopilot.analyze.scenes."""
        from autopilot.analyze.scenes import ShotDetectionError, detect_shots

        assert ShotDetectionError is not None
        assert detect_shots is not None

    def test_shot_detection_error_is_exception(self) -> None:
        """ShotDetectionError is a subclass of Exception with message."""
        from autopilot.analyze.scenes import ShotDetectionError

        assert issubclass(ShotDetectionError, Exception)
        err = ShotDetectionError("test error")
        assert str(err) == "test error"

    def test_detect_shots_signature(self) -> None:
        """detect_shots has correct signature: media_id, video_path, db, scheduler (no config)."""
        from autopilot.analyze.scenes import detect_shots

        sig = inspect.signature(detect_shots)
        param_names = list(sig.parameters.keys())
        assert param_names == ["media_id", "video_path", "db", "scheduler"]
        # With __future__ annotations, return type is stringified
        assert sig.return_annotation in (None, "None")

    def test_all_exports(self) -> None:
        """__all__ exports ShotDetectionError and detect_shots."""
        from autopilot.analyze import scenes

        assert set(scenes.__all__) == {"ShotDetectionError", "detect_shots"}

    def test_module_constants(self) -> None:
        """Module constants are accessible with correct values."""
        from autopilot.analyze.scenes import (
            PYSCENEDETECT_THRESHOLD,
            TRANSNETV2_INPUT_HEIGHT,
            TRANSNETV2_INPUT_WIDTH,
        )

        assert TRANSNETV2_INPUT_WIDTH == 48
        assert TRANSNETV2_INPUT_HEIGHT == 27
        assert PYSCENEDETECT_THRESHOLD == 27.0
