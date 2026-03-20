"""Tests for EDL validation (autopilot.plan.validator)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest


# -- Step 1: Public API surface tests -----------------------------------------


class TestValidatorPublicAPI:
    """Verify ValidationResult, validate_edl, and EdlValidationError surface."""

    def test_validation_result_importable(self):
        """ValidationResult is importable from validator module."""
        from autopilot.plan.validator import ValidationResult

        assert ValidationResult is not None

    def test_validation_result_has_fields(self):
        """ValidationResult has passed, errors, and warnings fields."""
        from autopilot.plan.validator import ValidationResult

        result = ValidationResult(passed=True, errors=[], warnings=[])
        assert result.passed is True
        assert result.errors == []
        assert result.warnings == []

    def test_validation_result_with_errors(self):
        """ValidationResult can hold errors and warnings."""
        from autopilot.plan.validator import ValidationResult

        result = ValidationResult(
            passed=False,
            errors=["Overlap on track 1"],
            warnings=["Duration slightly short"],
        )
        assert result.passed is False
        assert len(result.errors) == 1
        assert len(result.warnings) == 1

    def test_edl_validation_error_importable(self):
        """EdlValidationError is importable and is an Exception subclass."""
        from autopilot.plan.validator import EdlValidationError

        assert issubclass(EdlValidationError, Exception)
        err = EdlValidationError("test")
        assert str(err) == "test"

    def test_validate_edl_importable_and_callable(self):
        """validate_edl is importable and callable."""
        from autopilot.plan.validator import validate_edl

        assert callable(validate_edl)

    def test_validate_edl_signature(self):
        """validate_edl accepts edl dict and db, returns ValidationResult."""
        from autopilot.plan.validator import validate_edl

        sig = inspect.signature(validate_edl)
        params = list(sig.parameters.keys())
        assert "edl" in params
        assert "db" in params

    def test_all_exports(self):
        """__all__ includes EdlValidationError, ValidationResult, validate_edl."""
        from autopilot.plan import validator

        assert "EdlValidationError" in validator.__all__
        assert "ValidationResult" in validator.__all__
        assert "validate_edl" in validator.__all__


# -- Step 3: Overlap detection tests ------------------------------------------


def _mock_db():
    """Create a mock CatalogDB that returns media for known clip_ids."""
    db = MagicMock()
    db.get_media.return_value = {"id": "v1", "duration_seconds": 120.0}
    return db


class TestOverlapDetection:
    """Tests for validate_edl overlap detection on same track."""

    def test_non_overlapping_clips_pass(self):
        """Non-overlapping clips on the same track produce no errors."""
        from autopilot.plan.validator import validate_edl

        edl = {
            "target_duration_seconds": 20,
            "clips": [
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            "transitions": [],
            "audio_settings": [],
            "crop_modes": [],
            "titles": [],
            "music": [],
            "voiceovers": [],
            "broll_requests": [],
        }
        result = validate_edl(edl, _mock_db())
        # No overlap errors
        overlap_errors = [e for e in result.errors if "overlap" in e.lower()]
        assert len(overlap_errors) == 0

    def test_overlapping_clips_same_track_fail(self):
        """Overlapping clips on the same track produce an error."""
        from autopilot.plan.validator import validate_edl

        edl = {
            "target_duration_seconds": 15,
            "clips": [
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:05.000",
                    "out_timecode": "00:00:15.000",
                    "track": 1,
                },
            ],
            "transitions": [],
            "audio_settings": [],
            "crop_modes": [],
            "titles": [],
            "music": [],
            "voiceovers": [],
            "broll_requests": [],
        }
        result = validate_edl(edl, _mock_db())
        assert result.passed is False
        overlap_errors = [e for e in result.errors if "overlap" in e.lower()]
        assert len(overlap_errors) >= 1

    def test_overlapping_clips_different_tracks_pass(self):
        """Overlapping clips on different tracks produce no overlap errors."""
        from autopilot.plan.validator import validate_edl

        edl = {
            "target_duration_seconds": 10,
            "clips": [
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:05.000",
                    "out_timecode": "00:00:15.000",
                    "track": 2,
                },
            ],
            "transitions": [],
            "audio_settings": [],
            "crop_modes": [],
            "titles": [],
            "music": [],
            "voiceovers": [],
            "broll_requests": [],
        }
        result = validate_edl(edl, _mock_db())
        overlap_errors = [e for e in result.errors if "overlap" in e.lower()]
        assert len(overlap_errors) == 0
