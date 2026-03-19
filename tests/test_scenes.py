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


class TestIdempotency:
    """Tests for idempotency guard — skip when boundaries already exist."""

    def test_skips_when_boundaries_exist(self, catalog_db, tmp_path) -> None:
        """When boundaries already exist for media_id, detect_shots returns early."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")

        # Insert media + existing boundaries
        catalog_db.insert_media("vid1", str(video_file))
        catalog_db.upsert_boundaries("vid1", "[]", "transnetv2")

        scheduler = MagicMock()

        # Should return without calling scheduler
        detect_shots("vid1", video_file, catalog_db, scheduler)

        scheduler.model.assert_not_called()

    def test_proceeds_when_no_boundaries(self, catalog_db, tmp_path) -> None:
        """When no boundaries exist, detect_shots proceeds past the guard."""
        from autopilot.analyze.scenes import detect_shots

        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake")

        catalog_db.insert_media("vid2", str(video_file))

        scheduler = MagicMock()

        # Should NOT return early — will raise because no real model/video
        # but the point is it gets past the idempotency check
        with pytest.raises(Exception):
            detect_shots("vid2", video_file, catalog_db, scheduler)


class TestInputValidation:
    """Tests for video path validation."""

    def test_nonexistent_path_raises(self, catalog_db, tmp_path) -> None:
        """Nonexistent video_path raises ShotDetectionError matching 'not found'."""
        from autopilot.analyze.scenes import ShotDetectionError, detect_shots

        bad_path = tmp_path / "does_not_exist.mp4"
        catalog_db.insert_media("vid3", str(bad_path))
        scheduler = MagicMock()

        with pytest.raises(ShotDetectionError, match="not found"):
            detect_shots("vid3", bad_path, catalog_db, scheduler)

    def test_validation_before_heavy_imports(self, catalog_db, tmp_path) -> None:
        """Validation happens before heavy imports — scheduler not called on bad path."""
        from autopilot.analyze.scenes import ShotDetectionError, detect_shots

        bad_path = tmp_path / "nope.mp4"
        catalog_db.insert_media("vid4", str(bad_path))
        scheduler = MagicMock()

        with pytest.raises(ShotDetectionError):
            detect_shots("vid4", bad_path, catalog_db, scheduler)

        scheduler.model.assert_not_called()
