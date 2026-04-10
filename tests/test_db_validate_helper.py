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


class TestExecuteKwargsUpdate:
    """Direct tests for the _execute_kwargs_update private helper."""

    def test_returns_zero_on_empty_kwargs(self, catalog_db):
        """Empty kwargs returns 0 without touching the database."""
        result = catalog_db._execute_kwargs_update(
            "activity_clusters", "cluster_id", "nonexistent",
            frozenset({"label"}), "cluster", {},
        )
        assert result == 0

    def test_raises_valueerror_on_disallowed_columns(self, catalog_db):
        """Delegates to _validate_update_kwargs; bad columns raise ValueError."""
        with pytest.raises(ValueError, match=r"Disallowed column\(s\) for widget update"):
            catalog_db._execute_kwargs_update(
                "activity_clusters", "cluster_id", "ac1",
                frozenset({"label"}), "widget", {"evil_col": "bad"},
            )

    def test_returns_zero_when_pk_not_found(self, catalog_db):
        """Returns 0 when the primary-key value is not found (docstring promise)."""
        result = catalog_db._execute_kwargs_update(
            "activity_clusters", "cluster_id", "does_not_exist",
            frozenset({"label"}), "cluster", {"label": "X"},
        )
        assert result == 0

    def test_executes_update_and_returns_rowcount(self, catalog_db):
        """Builds SET clause, executes UPDATE, returns row count.

        Non-updated columns remain unchanged.
        """
        catalog_db.insert_activity_cluster(
            cluster_id="ac1", label="Original", location_label="Park"
        )
        rowcount = catalog_db._execute_kwargs_update(
            "activity_clusters", "cluster_id", "ac1",
            frozenset({"label", "description"}), "cluster",
            {"label": "Updated", "description": "New desc"},
        )
        assert rowcount == 1
        row = catalog_db.get_activity_cluster("ac1")
        assert row["label"] == "Updated"
        assert row["description"] == "New desc"
        assert row["location_label"] == "Park"

    def test_single_column_update(self, catalog_db):
        """Handles single-column update correctly (no trailing comma bug)."""
        catalog_db.insert_activity_cluster(cluster_id="ac2", label="Before")
        rowcount = catalog_db._execute_kwargs_update(
            "activity_clusters", "cluster_id", "ac2",
            frozenset({"label"}), "cluster", {"label": "After"},
        )
        assert rowcount == 1
        row = catalog_db.get_activity_cluster("ac2")
        assert row["label"] == "After"

    def test_raises_valueerror_on_non_identifier_table(self, catalog_db):
        """SQL-injection-style table name raises ValueError."""
        with pytest.raises(ValueError, match=r"Invalid table name"):
            catalog_db._execute_kwargs_update(
                "users; DROP TABLE users; --",
                "cluster_id", "ac1",
                frozenset({"label"}), "cluster", {"label": "x"},
            )


class TestUpdateMethodsDelegateValidation:
    """Verify each public update method delegates to _validate_update_kwargs."""

    def test_update_activity_cluster_rejects_disallowed_columns(self, catalog_db):
        """update_activity_cluster raises ValueError with 'cluster' entity."""
        catalog_db.insert_activity_cluster(cluster_id="ac1", label="Test")
        with pytest.raises(ValueError, match=r"Disallowed column\(s\) for cluster update"):
            catalog_db.update_activity_cluster("ac1", not_a_column="bad")

    def test_update_narrative_rejects_disallowed_columns(self, catalog_db):
        """update_narrative raises ValueError with 'narrative' entity."""
        catalog_db.insert_narrative(narrative_id="n1", title="Test")
        with pytest.raises(ValueError, match=r"Disallowed column\(s\) for narrative update"):
            catalog_db.update_narrative("n1", not_a_column="bad")

    def test_update_gate_rejects_disallowed_columns(self, catalog_db):
        """update_gate raises ValueError with 'gate' entity."""
        catalog_db.init_default_gates()
        with pytest.raises(ValueError, match=r"Disallowed column\(s\) for gate update"):
            catalog_db.update_gate("ingest", not_a_column="bad")

    def test_update_job_rejects_disallowed_columns(self, catalog_db):
        """update_job raises ValueError with 'job' entity."""
        catalog_db.insert_job("j1", "ingest", "media_import")
        with pytest.raises(ValueError, match=r"Disallowed column\(s\) for job update"):
            catalog_db.update_job("j1", not_a_column="bad")

    def test_update_run_rejects_disallowed_columns(self, catalog_db):
        """update_run raises ValueError with 'run' entity."""
        catalog_db.insert_run("r1", started_at="2026-01-01T00:00:00")
        with pytest.raises(ValueError, match=r"Disallowed column\(s\) for run update"):
            catalog_db.update_run("r1", not_a_column="bad")
