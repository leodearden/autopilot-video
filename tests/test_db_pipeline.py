"""Tests for pipeline control tables in autopilot.db CatalogDB."""

from __future__ import annotations


# -- Schema tests for pipeline tables ----------------------------------------


class TestPipelineSchema:
    """Verify pipeline control tables exist with correct columns and indexes."""

    def test_pipeline_gates_columns(self, catalog_db):
        """pipeline_gates table has expected columns."""
        cur = catalog_db.conn.execute("PRAGMA table_info(pipeline_gates)")
        columns = [row[1] for row in cur.fetchall()]
        assert columns == [
            "stage",
            "mode",
            "status",
            "decided_at",
            "decided_by",
            "notes",
            "timeout_hours",
        ]

    def test_pipeline_jobs_columns(self, catalog_db):
        """pipeline_jobs table has expected columns."""
        cur = catalog_db.conn.execute("PRAGMA table_info(pipeline_jobs)")
        columns = [row[1] for row in cur.fetchall()]
        assert columns == [
            "job_id",
            "stage",
            "job_type",
            "target_id",
            "target_label",
            "status",
            "started_at",
            "finished_at",
            "duration_seconds",
            "progress_pct",
            "error_message",
            "worker",
            "run_id",
        ]

    def test_pipeline_events_columns(self, catalog_db):
        """pipeline_events table has expected columns."""
        cur = catalog_db.conn.execute("PRAGMA table_info(pipeline_events)")
        columns = [row[1] for row in cur.fetchall()]
        assert columns == [
            "event_id",
            "event_type",
            "stage",
            "job_id",
            "payload_json",
            "created_at",
        ]

    def test_pipeline_runs_columns(self, catalog_db):
        """pipeline_runs table has expected columns."""
        cur = catalog_db.conn.execute("PRAGMA table_info(pipeline_runs)")
        columns = [row[1] for row in cur.fetchall()]
        assert columns == [
            "run_id",
            "started_at",
            "finished_at",
            "config_snapshot",
            "current_stage",
            "status",
            "wall_clock_seconds",
            "budget_remaining_seconds",
        ]

    def test_pipeline_jobs_stage_status_index(self, catalog_db):
        """Index exists on pipeline_jobs(stage, status)."""
        cur = catalog_db.conn.execute(
            "PRAGMA index_info(idx_pipeline_jobs_stage_status)"
        )
        cols = [row[2] for row in cur.fetchall()]
        assert cols == ["stage", "status"]

    def test_pipeline_jobs_run_id_index(self, catalog_db):
        """Index exists on pipeline_jobs(run_id)."""
        cur = catalog_db.conn.execute(
            "PRAGMA index_info(idx_pipeline_jobs_run_id)"
        )
        cols = [row[2] for row in cur.fetchall()]
        assert cols == ["run_id"]

    def test_pipeline_events_event_type_index(self, catalog_db):
        """Index exists on pipeline_events(event_type)."""
        cur = catalog_db.conn.execute(
            "PRAGMA index_info(idx_pipeline_events_event_type)"
        )
        cols = [row[2] for row in cur.fetchall()]
        assert cols == ["event_type"]

    def test_pipeline_events_created_at_index(self, catalog_db):
        """Index exists on pipeline_events(created_at)."""
        cur = catalog_db.conn.execute(
            "PRAGMA index_info(idx_pipeline_events_created_at)"
        )
        cols = [row[2] for row in cur.fetchall()]
        assert cols == ["created_at"]
