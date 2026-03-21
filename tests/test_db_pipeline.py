"""Tests for pipeline control tables in autopilot.db CatalogDB."""

from __future__ import annotations

KNOWN_STAGES = [
    "ingest",
    "analyze",
    "classify",
    "narrate",
    "script",
    "edl",
    "source",
    "render",
    "upload",
]

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


# -- Gates CRUD tests -------------------------------------------------------


class TestGatesCRUD:
    """Tests for pipeline gates CRUD methods."""

    def test_init_default_gates_populates_stages(self, catalog_db):
        """init_default_gates() inserts rows for all known pipeline stages."""
        catalog_db.init_default_gates()
        cur = catalog_db.conn.execute(
            "SELECT stage FROM pipeline_gates ORDER BY stage"
        )
        stages = [row[0] for row in cur.fetchall()]
        assert stages == sorted(KNOWN_STAGES)

    def test_init_default_gates_is_idempotent(self, catalog_db):
        """Calling init_default_gates() twice doesn't duplicate or overwrite."""
        catalog_db.init_default_gates()
        # Modify a gate
        catalog_db.update_gate("ingest", mode="manual")
        # Call again
        catalog_db.init_default_gates()
        gate = catalog_db.get_gate("ingest")
        assert gate is not None
        assert gate["mode"] == "manual"  # preserved, not overwritten
        cur = catalog_db.conn.execute("SELECT count(*) FROM pipeline_gates")
        assert cur.fetchone()[0] == len(KNOWN_STAGES)

    def test_get_gate_returns_dict(self, catalog_db):
        """get_gate() returns a dict with all expected fields."""
        catalog_db.init_default_gates()
        gate = catalog_db.get_gate("ingest")
        assert gate is not None
        assert isinstance(gate, dict)
        assert gate["stage"] == "ingest"
        assert gate["mode"] == "auto"
        assert gate["status"] == "idle"
        assert gate["decided_by"] == "system"

    def test_get_gate_not_found_returns_none(self, catalog_db):
        """get_gate() returns None for a nonexistent stage."""
        result = catalog_db.get_gate("nonexistent")
        assert result is None

    def test_get_all_gates_returns_list(self, catalog_db):
        """get_all_gates() returns a list of dicts."""
        catalog_db.init_default_gates()
        gates = catalog_db.get_all_gates()
        assert isinstance(gates, list)
        assert len(gates) == len(KNOWN_STAGES)
        assert all(isinstance(g, dict) for g in gates)

    def test_update_gate_modifies_fields(self, catalog_db):
        """update_gate() changes specified fields."""
        catalog_db.init_default_gates()
        catalog_db.update_gate(
            "analyze", mode="manual", status="approved", notes="LGTM"
        )
        gate = catalog_db.get_gate("analyze")
        assert gate is not None
        assert gate["mode"] == "manual"
        assert gate["status"] == "approved"
        assert gate["notes"] == "LGTM"

    def test_update_gate_noop_on_empty_kwargs(self, catalog_db):
        """update_gate() with no kwargs does nothing."""
        catalog_db.init_default_gates()
        catalog_db.update_gate("ingest")  # no kwargs
        gate = catalog_db.get_gate("ingest")
        assert gate is not None
        assert gate["mode"] == "auto"  # unchanged
