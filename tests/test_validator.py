"""Tests for EDL validation (autopilot.plan.validator)."""

from __future__ import annotations

import inspect

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
