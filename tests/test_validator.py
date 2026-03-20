"""Tests for EDL validation (autopilot.plan.validator)."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

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


# -- Step 5: Duration check tests ---------------------------------------------


def _make_edl_with_duration(clip_out: str, target: float) -> dict:
    """Helper to make a simple EDL with one clip and a target duration."""
    return {
        "target_duration_seconds": target,
        "clips": [
            {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": clip_out,
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


class TestDurationCheck:
    """Tests for validate_edl total duration check within ±10%."""

    def test_within_tolerance_passes(self):
        """Duration within ±10% of target produces no errors."""
        from autopilot.plan.validator import validate_edl

        # 10s clip, target 10s -> exactly on target
        edl = _make_edl_with_duration("00:00:10.000", 10.0)
        result = validate_edl(edl, _mock_db())
        duration_errors = [e for e in result.errors if "duration" in e.lower()]
        assert len(duration_errors) == 0

    def test_over_10_percent_above_fails(self):
        """Duration more than 10% above target produces an error."""
        from autopilot.plan.validator import validate_edl

        # 12s clip, target 10s -> 20% over
        edl = _make_edl_with_duration("00:00:12.000", 10.0)
        result = validate_edl(edl, _mock_db())
        assert result.passed is False
        duration_errors = [e for e in result.errors if "duration" in e.lower()]
        assert len(duration_errors) >= 1

    def test_over_10_percent_below_fails(self):
        """Duration more than 10% below target produces an error."""
        from autopilot.plan.validator import validate_edl

        # 8s clip, target 10s -> 20% under
        edl = _make_edl_with_duration("00:00:08.000", 10.0)
        result = validate_edl(edl, _mock_db())
        assert result.passed is False
        duration_errors = [e for e in result.errors if "duration" in e.lower()]
        assert len(duration_errors) >= 1

    def test_exactly_at_boundary_passes(self):
        """Duration exactly at ±10% boundary passes."""
        from autopilot.plan.validator import validate_edl

        # 11s clip, target 10s -> exactly 10% over (boundary)
        edl = _make_edl_with_duration("00:00:11.000", 10.0)
        result = validate_edl(edl, _mock_db())
        duration_errors = [e for e in result.errors if "duration" in e.lower()]
        assert len(duration_errors) == 0

        # 9s clip, target 10s -> exactly 10% under (boundary)
        edl = _make_edl_with_duration("00:00:09.000", 10.0)
        result = validate_edl(edl, _mock_db())
        duration_errors = [e for e in result.errors if "duration" in e.lower()]
        assert len(duration_errors) == 0


# -- Step 7: Clip ID existence tests ------------------------------------------


class TestClipIdExistence:
    """Tests for validate_edl clip_id existence check."""

    def test_all_valid_clip_ids_pass(self):
        """All valid clip_ids produce no clip-existence errors."""
        from autopilot.plan.validator import validate_edl

        db = MagicMock()
        db.get_media.return_value = {"id": "v1", "duration_seconds": 120.0}

        edl = _make_edl_with_duration("00:00:10.000", 10.0)
        result = validate_edl(edl, db)
        clip_errors = [e for e in result.errors if "clip" in e.lower() and "not found" in e.lower()]
        assert len(clip_errors) == 0

    def test_unknown_clip_id_fails(self):
        """Unknown clip_id produces an error listing the missing ID."""
        from autopilot.plan.validator import validate_edl

        db = MagicMock()
        db.get_media.return_value = None  # clip doesn't exist

        edl = _make_edl_with_duration("00:00:10.000", 10.0)
        result = validate_edl(edl, db)
        assert result.passed is False
        clip_errors = [e for e in result.errors if "v1" in e]
        assert len(clip_errors) >= 1


# -- Step 9: In/out timecode bounds tests --------------------------------------


def _make_bounded_edl(in_tc: str, out_tc: str, clip_duration: float) -> tuple[dict, MagicMock]:
    """Create an EDL and mock DB with a clip of given duration."""
    db = MagicMock()
    db.get_media.return_value = {"id": "v1", "duration_seconds": clip_duration}
    edl = {
        "target_duration_seconds": 1000,  # generous to avoid duration errors
        "clips": [
            {
                "clip_id": "v1",
                "in_timecode": in_tc,
                "out_timecode": out_tc,
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
    return edl, db


class TestTimeccodeBounds:
    """Tests for validate_edl in/out point bounds checking."""

    def test_valid_in_out_within_duration_passes(self):
        """In/out timecodes within clip duration produce no bounds errors."""
        from autopilot.plan.validator import validate_edl

        # Clip is 60s, use 0-30s
        edl, db = _make_bounded_edl("00:00:00.000", "00:00:30.000", 60.0)
        result = validate_edl(edl, db)
        bounds_errors = [e for e in result.errors if "bound" in e.lower() or "exceed" in e.lower()]
        assert len(bounds_errors) == 0

    def test_out_timecode_exceeding_clip_duration_fails(self):
        """Out timecode beyond clip duration produces an error."""
        from autopilot.plan.validator import validate_edl

        # Clip is 30s, out is 45s
        edl, db = _make_bounded_edl("00:00:00.000", "00:00:45.000", 30.0)
        result = validate_edl(edl, db)
        assert result.passed is False
        bounds_errors = [
            e for e in result.errors
            if "bound" in e.lower() or "exceed" in e.lower() or "beyond" in e.lower()
        ]
        assert len(bounds_errors) >= 1

    def test_in_timecode_after_out_timecode_fails(self):
        """In timecode after out timecode produces an error."""
        from autopilot.plan.validator import validate_edl

        # in=20s, out=10s -> invalid
        edl, db = _make_bounded_edl("00:00:20.000", "00:00:10.000", 60.0)
        result = validate_edl(edl, db)
        assert result.passed is False
        order_errors = [
            e for e in result.errors
            if "before" in e.lower() or "after" in e.lower() or "order" in e.lower()
                or "in_timecode" in e.lower()
        ]
        assert len(order_errors) >= 1


# -- Step 11: Audio level tests -----------------------------------------------


def _make_audio_edl(level_db: float) -> dict:
    """Create a minimal EDL with one audio_settings entry."""
    return {
        "target_duration_seconds": 10,
        "clips": [
            {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            },
        ],
        "transitions": [],
        "audio_settings": [
            {"clip_id": "v1", "level_db": level_db},
        ],
        "crop_modes": [],
        "titles": [],
        "music": [],
        "voiceovers": [],
        "broll_requests": [],
    }


class TestAudioLevelCheck:
    """Tests for validate_edl audio level broadcast-safe check."""

    def test_levels_in_range_pass(self):
        """Audio levels in [-24, 0] dB produce no audio errors."""
        from autopilot.plan.validator import validate_edl

        edl = _make_audio_edl(-12.0)
        result = validate_edl(edl, _mock_db())
        audio_errors = [e for e in result.errors if "audio" in e.lower() or "level" in e.lower()]
        assert len(audio_errors) == 0

    def test_level_below_minus_24_generates_warning(self):
        """Audio level below -24 dB generates a warning."""
        from autopilot.plan.validator import validate_edl

        edl = _make_audio_edl(-30.0)
        result = validate_edl(edl, _mock_db())
        audio_warnings = [
            w for w in result.warnings
            if "audio" in w.lower() or "level" in w.lower()
        ]
        assert len(audio_warnings) >= 1

    def test_level_above_0_generates_error(self):
        """Audio level above 0 dB generates an error."""
        from autopilot.plan.validator import validate_edl

        edl = _make_audio_edl(6.0)
        result = validate_edl(edl, _mock_db())
        assert result.passed is False
        audio_errors = [e for e in result.errors if "audio" in e.lower() or "level" in e.lower()]
        assert len(audio_errors) >= 1


# -- Step 25: Malformed timecodes and missing keys tests ----------------------


def _make_edl_with_clips(clips: list[dict]) -> dict:
    """Create a full EDL structure with given clips list."""
    return {
        "target_duration_seconds": 1000,
        "clips": clips,
        "transitions": [],
        "audio_settings": [],
        "crop_modes": [],
        "titles": [],
        "music": [],
        "voiceovers": [],
        "broll_requests": [],
    }


class TestMalformedTimecodes:
    """Tests that validate_edl handles malformed timecodes and missing keys
    gracefully — always returning ValidationResult, never raising."""

    def test_malformed_timecode_string_returns_error_not_raises(self):
        """Clip with malformed timecode (e.g. 'not-a-timecode') returns
        ValidationResult with passed=False and descriptive error, never raises."""
        from autopilot.plan.validator import validate_edl

        edl = _make_edl_with_clips([
            {
                "clip_id": "v1",
                "in_timecode": "not-a-timecode",
                "out_timecode": "00:00:10.000",
                "track": 1,
            },
        ])
        # Must NOT raise ValueError
        result = validate_edl(edl, _mock_db())
        assert result.passed is False
        assert any("timecode" in e.lower() or "invalid" in e.lower() for e in result.errors)

    def test_missing_in_timecode_key_returns_error_not_raises(self):
        """Clip with missing in_timecode key returns ValidationResult with
        error, never raises KeyError."""
        from autopilot.plan.validator import validate_edl

        edl = _make_edl_with_clips([
            {
                "clip_id": "v1",
                # no in_timecode key
                "out_timecode": "00:00:10.000",
                "track": 1,
            },
        ])
        # Must NOT raise KeyError
        result = validate_edl(edl, _mock_db())
        assert result.passed is False
        assert any("timecode" in e.lower() or "missing" in e.lower() for e in result.errors)

    def test_missing_out_timecode_key_returns_error_not_raises(self):
        """Clip with missing out_timecode key returns ValidationResult with
        error, never raises KeyError."""
        from autopilot.plan.validator import validate_edl

        edl = _make_edl_with_clips([
            {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                # no out_timecode key
                "track": 1,
            },
        ])
        # Must NOT raise KeyError
        result = validate_edl(edl, _mock_db())
        assert result.passed is False
        assert any("timecode" in e.lower() or "missing" in e.lower() for e in result.errors)

    def test_non_standard_timecode_format_returns_error(self):
        """Clip with non-standard format (e.g. '10s') returns error in result."""
        from autopilot.plan.validator import validate_edl

        edl = _make_edl_with_clips([
            {
                "clip_id": "v1",
                "in_timecode": "10s",
                "out_timecode": "20s",
                "track": 1,
            },
        ])
        # Must NOT raise
        result = validate_edl(edl, _mock_db())
        assert result.passed is False
        assert len(result.errors) >= 1

    def test_mix_of_valid_and_malformed_clips(self):
        """EDL with mix of valid and malformed clips collects errors for bad
        clips but still validates the good ones."""
        from autopilot.plan.validator import validate_edl

        edl = _make_edl_with_clips([
            {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            },
            {
                "clip_id": "v2",
                "in_timecode": "garbage",
                "out_timecode": "00:00:20.000",
                "track": 1,
            },
        ])
        # Must NOT raise
        result = validate_edl(edl, _mock_db())
        assert result.passed is False
        # Should have error about v2's bad timecode
        malformed_errors = [
            e for e in result.errors
            if "v2" in e or "timecode" in e.lower() or "invalid" in e.lower()
        ]
        assert len(malformed_errors) >= 1


# -- Step 27: Non-numeric catalog duration tests ------------------------------


class TestMalformedCatalogData:
    """Tests that validate_edl handles non-numeric duration_seconds from
    catalog data gracefully — always returning ValidationResult, never raising."""

    def test_duration_seconds_none_returns_error_not_raises(self):
        """db.get_media returns {'duration_seconds': None} — validate_edl
        returns ValidationResult with error, never raises TypeError."""
        from autopilot.plan.validator import validate_edl

        db = MagicMock()
        db.get_media.return_value = {"id": "v1", "duration_seconds": None}

        edl, _ = _make_bounded_edl("00:00:00.000", "00:00:10.000", 60.0)
        # Override db with our broken-duration mock
        result = validate_edl(edl, db)
        # Should not raise TypeError; should have an error about non-numeric duration
        assert result.passed is False
        duration_errors = [
            e for e in result.errors
            if "duration" in e.lower() or "non-numeric" in e.lower()
        ]
        assert len(duration_errors) >= 1

    def test_duration_seconds_bad_string_returns_error_not_raises(self):
        """db.get_media returns {'duration_seconds': 'bad'} — returns error,
        never raises ValueError."""
        from autopilot.plan.validator import validate_edl

        db = MagicMock()
        db.get_media.return_value = {"id": "v1", "duration_seconds": "bad"}

        edl, _ = _make_bounded_edl("00:00:00.000", "00:00:10.000", 60.0)
        result = validate_edl(edl, db)
        assert result.passed is False
        duration_errors = [
            e for e in result.errors
            if "duration" in e.lower() or "non-numeric" in e.lower()
        ]
        assert len(duration_errors) >= 1

    def test_duration_seconds_missing_key_handles_gracefully(self):
        """db.get_media returns {} (missing duration_seconds key) — returns
        error or gracefully handles, never raises."""
        from autopilot.plan.validator import validate_edl

        db = MagicMock()
        db.get_media.return_value = {"id": "v1"}  # no duration_seconds

        edl, _ = _make_bounded_edl("00:00:00.000", "00:00:10.000", 60.0)
        result = validate_edl(edl, db)
        # Should not raise KeyError.
        # The default of 0 from get("duration_seconds", 0) is valid for float(),
        # so out_timecode 10s > duration 0s will trigger a "exceeds" error.
        assert result.passed is False
