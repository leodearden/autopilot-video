"""Unit tests for CatalogDB._validate_update_kwargs helper."""

from __future__ import annotations

import pytest


class TestValidateUpdateKwargs:
    """Direct tests for the _validate_update_kwargs private helper."""

    def test_passes_silently_when_all_keys_allowed(self, catalog_db):
        """No exception when all kwargs keys are in the allowed set."""
        allowed = frozenset({"status", "name", "count"})
        # Should not raise
        catalog_db._validate_update_kwargs(allowed, {"status": "done", "name": "x"}, "test")

    def test_raises_valueerror_for_disallowed_keys(self, catalog_db):
        """Raises ValueError when kwargs contain keys not in the allowed set."""
        allowed = frozenset({"status", "name"})
        with pytest.raises(ValueError):
            catalog_db._validate_update_kwargs(allowed, {"status": "ok", "evil": "bad"}, "thing")

    def test_error_message_includes_entity_name(self, catalog_db):
        """The ValueError message contains the entity name."""
        allowed = frozenset({"status"})
        with pytest.raises(ValueError, match="widget"):
            catalog_db._validate_update_kwargs(allowed, {"bad_col": 1}, "widget")

    def test_error_message_includes_sorted_bad_keys(self, catalog_db):
        """The ValueError message lists bad keys in sorted order."""
        allowed = frozenset({"status"})
        with pytest.raises(ValueError, match=r"\['alpha', 'beta'\]"):
            catalog_db._validate_update_kwargs(allowed, {"beta": 1, "alpha": 2}, "entity")

    def test_passes_with_empty_kwargs(self, catalog_db):
        """Empty kwargs should pass validation (no bad keys)."""
        allowed = frozenset({"status"})
        # Should not raise
        catalog_db._validate_update_kwargs(allowed, {}, "entity")
