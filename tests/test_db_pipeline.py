"""Tests for pipeline control tables in autopilot.db CatalogDB."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict
from unittest.mock import patch

import pytest

from autopilot.db import CatalogDB

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


# -- Specs for parametrized update-column-validation tests --------------------
#
# Each entry maps a short key to the callables and fixture data needed to
# exercise one of the three update_* methods.  The lambdas accept a CatalogDB
# as first arg so the parametrize decorator only carries a string key.


class _UpdateSpec(TypedDict):
    setup: Callable[..., object]
    update: Callable[..., object]
    get: Callable[..., dict | None]
    valid_col: str
    valid_val: str
    default_val: str
    invalid_col: str


_UPDATE_SPECS: dict[str, _UpdateSpec] = {
    "gate": {
        "setup": lambda db: db.init_default_gates(),
        "update": lambda db, **kw: db.update_gate("ingest", **kw),
        "get": lambda db: db.get_gate("ingest"),
        "valid_col": "mode",
        "valid_val": "manual",
        "default_val": "auto",
        "invalid_col": "hacked",
    },
    "job": {
        "setup": lambda db: db.insert_job("j-val", "ingest", "media_import"),
        "update": lambda db, **kw: db.update_job("j-val", **kw),
        "get": lambda db: db.get_job("j-val"),
        "valid_col": "status",
        "valid_val": "done",
        "default_val": "pending",
        "invalid_col": "injected",
    },
    "run": {
        "setup": lambda db: db.insert_run(
            "r-val", started_at="2026-01-01T00:00:00"
        ),
        "update": lambda db, **kw: db.update_run("r-val", **kw),
        "get": lambda db: db.get_run("r-val"),
        "valid_col": "status",
        "valid_val": "completed",
        "default_val": "running",
        "invalid_col": "corrupted",
    },
}

# -- Meta-tests: lock structural/type refactors in place ----------------------


def test_update_spec_is_typeddict_with_expected_fields() -> None:
    """_UpdateSpec is a TypedDict with exactly the seven expected fields."""
    import sys
    import typing

    module = sys.modules[__name__]

    # (a) _UpdateSpec exists in the module namespace
    assert hasattr(module, "_UpdateSpec"), "_UpdateSpec not found in module namespace"
    _UpdateSpec = getattr(module, "_UpdateSpec")  # noqa: N806

    # (b) it is a TypedDict
    assert typing.is_typeddict(_UpdateSpec), "_UpdateSpec is not a TypedDict"

    # (c) the keys of get_type_hints equal the expected set
    hints = typing.get_type_hints(_UpdateSpec)
    expected_keys = {"setup", "update", "get", "valid_col", "valid_val", "default_val", "invalid_col"}
    assert set(hints.keys()) == expected_keys, f"Keys mismatch: {set(hints.keys())}"

    # (d) str-typed fields are exactly str
    assert hints["valid_col"] is str, f"valid_col: expected str, got {hints['valid_col']}"
    assert hints["valid_val"] is str, f"valid_val: expected str, got {hints['valid_val']}"
    assert hints["default_val"] is str, f"default_val: expected str, got {hints['default_val']}"
    assert hints["invalid_col"] is str, f"invalid_col: expected str, got {hints['invalid_col']}"

    # (e) _UPDATE_SPECS annotation mentions _UpdateSpec (covers the annotation change)
    ann = module.__annotations__.get("_UPDATE_SPECS", "")
    assert "_UpdateSpec" in str(ann), (
        f"_UPDATE_SPECS annotation doesn't mention _UpdateSpec: {ann}"
    )


def test_non_pk_allowlist_parametrize_uses_frozensets() -> None:
    """parametrize on test_non_pk_columns_match_allowlist passes frozensets directly."""
    from autopilot.db import CatalogDB

    fn = TestPipelineSchema.test_non_pk_columns_match_allowlist
    markers = [m for m in fn.pytestmark if m.name == "parametrize"]
    assert len(markers) == 1, f"Expected 1 parametrize marker, got {len(markers)}"
    marker = markers[0]

    # (a) argnames uses 'allowed', not 'allowed_attr'
    assert marker.args[0] == "table, allowed", (
        f"argnames: expected 'table, allowed', got {marker.args[0]!r}"
    )

    # (b) each argvalues tuple has shape (str, frozenset)
    argvalues = marker.args[1]
    for i, av in enumerate(argvalues):
        assert isinstance(av[1], frozenset), (
            f"argvalues[{i}][1] is not a frozenset: {type(av[1])}"
        )

    # (c) the three frozenset values match the CatalogDB class attributes
    assert argvalues[0][1] == CatalogDB._GATE_ALLOWED_COLUMNS
    assert argvalues[1][1] == CatalogDB._JOB_ALLOWED_COLUMNS
    assert argvalues[2][1] == CatalogDB._RUN_ALLOWED_COLUMNS


def test_update_column_validation_parametrizes_use_update_specs_keys() -> None:
    """Entity parametrize on each TestUpdateColumnValidation test derives from _UPDATE_SPECS.

    Checks two invariants:
    (a) The argvalues of the 'entity' parametrize marker equal list(_UPDATE_SPECS) by value.
    (b) The source text of each test method contains the literal 'list(_UPDATE_SPECS)' so
        that hardcoded lists cannot silently drift from _UPDATE_SPECS.
    """
    import inspect

    test_methods = [
        TestUpdateColumnValidation.test_rejects_single_disallowed_column,
        TestUpdateColumnValidation.test_rejects_mix_of_valid_and_invalid,
        TestUpdateColumnValidation.test_rejects_multiple_disallowed_columns,
    ]

    for fn in test_methods:
        name = fn.__name__
        markers = [m for m in fn.pytestmark if m.name == "parametrize"]
        entity_markers = [m for m in markers if m.args[0] == "entity"]
        assert len(entity_markers) == 1, (
            f"{name}: expected exactly one 'entity' parametrize marker, "
            f"got {len(entity_markers)}"
        )
        marker = entity_markers[0]

        # (a) argvalues match _UPDATE_SPECS keys by value
        assert list(marker.args[1]) == list(_UPDATE_SPECS), (
            f"{name}: entity parametrize argvalues {list(marker.args[1])!r} "
            f"!= list(_UPDATE_SPECS) {list(_UPDATE_SPECS)!r}"
        )

        # (b) source text must contain 'list(_UPDATE_SPECS)' to enforce the
        #     source-of-truth invariant — hardcoded literals are disallowed
        src = inspect.getsource(fn)
        assert "list(_UPDATE_SPECS)" in src, (
            f"{name}: parametrize decorator must use list(_UPDATE_SPECS), "
            f"not a hardcoded literal"
        )


def test_update_specs_have_distinct_invalid_cols() -> None:
    """Every _UPDATE_SPECS entry has a non-empty invalid_col and all values are distinct."""
    invalid_cols = [spec["invalid_col"] for spec in _UPDATE_SPECS.values()]
    for entity, col in zip(_UPDATE_SPECS, invalid_cols):
        assert col, f"_UPDATE_SPECS[{entity!r}]['invalid_col'] is empty"
    assert len(set(invalid_cols)) == len(_UPDATE_SPECS), (
        f"invalid_col values are not all distinct: {invalid_cols}"
    )


def test_rejects_single_disallowed_column_has_bad_col_kind_parametrize() -> None:
    """test_rejects_single_disallowed_column has a bad_col_kind parametrize with 'generic'/'specific'."""
    fn = TestUpdateColumnValidation.test_rejects_single_disallowed_column
    markers = [m for m in fn.pytestmark if m.name == "parametrize"]
    bad_col_kind_markers = [m for m in markers if m.args[0] == "bad_col_kind"]
    assert len(bad_col_kind_markers) == 1, (
        f"Expected exactly one 'bad_col_kind' parametrize marker, "
        f"got {len(bad_col_kind_markers)}"
    )
    argvalues = list(bad_col_kind_markers[0].args[1])
    assert "generic" in argvalues, f"'generic' missing from bad_col_kind argvalues: {argvalues}"
    assert "specific" in argvalues, f"'specific' missing from bad_col_kind argvalues: {argvalues}"


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
        cur = catalog_db.conn.execute("PRAGMA index_info(idx_pipeline_jobs_stage_status)")
        cols = [row[2] for row in cur.fetchall()]
        assert cols == ["stage", "status"]

    def test_pipeline_jobs_run_id_index(self, catalog_db):
        """Index exists on pipeline_jobs(run_id)."""
        cur = catalog_db.conn.execute("PRAGMA index_info(idx_pipeline_jobs_run_id)")
        cols = [row[2] for row in cur.fetchall()]
        assert cols == ["run_id"]

    def test_pipeline_events_event_type_index(self, catalog_db):
        """Index exists on pipeline_events(event_type)."""
        cur = catalog_db.conn.execute("PRAGMA index_info(idx_pipeline_events_event_type)")
        cols = [row[2] for row in cur.fetchall()]
        assert cols == ["event_type"]

    def test_pipeline_events_created_at_index(self, catalog_db):
        """Index exists on pipeline_events(created_at)."""
        cur = catalog_db.conn.execute("PRAGMA index_info(idx_pipeline_events_created_at)")
        cols = [row[2] for row in cur.fetchall()]
        assert cols == ["created_at"]

    @pytest.mark.parametrize(
        "table, allowed",
        [
            ("pipeline_gates", CatalogDB._GATE_ALLOWED_COLUMNS),
            ("pipeline_jobs", CatalogDB._JOB_ALLOWED_COLUMNS),
            ("pipeline_runs", CatalogDB._RUN_ALLOWED_COLUMNS),
        ],
        ids=["gates", "jobs", "runs"],
    )
    def test_non_pk_columns_match_allowlist(
        self, catalog_db: CatalogDB, table: str, allowed: frozenset[str]
    ) -> None:
        """Non-PK columns in the schema == the update allowlist frozenset.

        Catches drift in either direction: a column added to the CREATE TABLE
        but not the frozenset, or a stale frozenset entry for a removed column.
        """
        cur = catalog_db.conn.execute(f"PRAGMA table_info({table})")  # noqa: S608
        non_pk_cols = {row[1] for row in cur.fetchall() if row[5] == 0}
        assert non_pk_cols == allowed, (
            f"Schema drift detected for {table}:\n"
            f"  in schema but not allowlist: {non_pk_cols - allowed}\n"
            f"  in allowlist but not schema: {allowed - non_pk_cols}"
        )


# -- Gates CRUD tests -------------------------------------------------------


class TestGatesCRUD:
    """Tests for pipeline gates CRUD methods."""

    def test_init_default_gates_populates_stages(self, catalog_db):
        """init_default_gates() inserts rows for all known pipeline stages."""
        catalog_db.init_default_gates()
        cur = catalog_db.conn.execute("SELECT stage FROM pipeline_gates ORDER BY stage")
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
        catalog_db.update_gate("analyze", mode="manual", status="approved", notes="LGTM")
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



# -- Jobs CRUD tests --------------------------------------------------------


class TestJobsCRUD:
    """Tests for pipeline jobs CRUD methods."""

    def test_insert_job_all_fields(self, catalog_db):
        """insert_job() with all fields stores them correctly."""
        catalog_db.insert_job(
            "j1",
            "ingest",
            "media_import",
            target_id="m1",
            target_label="clip.mp4",
            status="running",
            started_at="2026-01-01T00:00:00",
            finished_at="2026-01-01T00:01:00",
            duration_seconds=60.0,
            progress_pct=100.0,
            error_message=None,
            worker="w1",
            run_id="r1",
        )
        job = catalog_db.get_job("j1")
        assert job is not None
        assert job["job_id"] == "j1"
        assert job["stage"] == "ingest"
        assert job["job_type"] == "media_import"
        assert job["target_id"] == "m1"
        assert job["target_label"] == "clip.mp4"
        assert job["status"] == "running"
        assert job["duration_seconds"] == 60.0
        assert job["run_id"] == "r1"

    def test_insert_job_minimal_fields(self, catalog_db):
        """insert_job() with only required fields uses defaults."""
        catalog_db.insert_job("j2", "analyze", "transcribe")
        job = catalog_db.get_job("j2")
        assert job is not None
        assert job["status"] == "pending"
        assert job["target_id"] is None
        assert job["run_id"] is None

    def test_get_job_returns_dict(self, catalog_db):
        """get_job() returns a plain dict."""
        catalog_db.insert_job("j3", "classify", "cluster")
        job = catalog_db.get_job("j3")
        assert isinstance(job, dict)

    def test_get_job_not_found(self, catalog_db):
        """get_job() returns None for a nonexistent job_id."""
        assert catalog_db.get_job("nonexistent") is None

    def test_update_job_modifies_fields(self, catalog_db):
        """update_job() changes specified fields."""
        catalog_db.insert_job("j4", "ingest", "media_import")
        catalog_db.update_job("j4", status="done", finished_at="2026-01-01T00:05:00")
        job = catalog_db.get_job("j4")
        assert job is not None
        assert job["status"] == "done"
        assert job["finished_at"] == "2026-01-01T00:05:00"

    def test_list_jobs_no_filter(self, catalog_db):
        """list_jobs() with no filters returns all jobs."""
        catalog_db.insert_job("j5", "ingest", "media_import")
        catalog_db.insert_job("j6", "analyze", "transcribe")
        jobs = catalog_db.list_jobs()
        assert len(jobs) == 2

    def test_list_jobs_filter_by_stage(self, catalog_db):
        """list_jobs(stage=...) returns only matching stage."""
        catalog_db.insert_job("j7", "ingest", "media_import")
        catalog_db.insert_job("j8", "analyze", "transcribe")
        jobs = catalog_db.list_jobs(stage="ingest")
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "j7"

    def test_list_jobs_filter_by_status(self, catalog_db):
        """list_jobs(status=...) returns only matching status."""
        catalog_db.insert_job("j9", "ingest", "media_import", status="done")
        catalog_db.insert_job("j10", "ingest", "media_import")
        jobs = catalog_db.list_jobs(status="pending")
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "j10"

    def test_list_jobs_filter_by_job_type(self, catalog_db):
        """list_jobs(job_type=...) returns only matching type."""
        catalog_db.insert_job("j11", "analyze", "transcribe")
        catalog_db.insert_job("j12", "analyze", "detect_objects")
        jobs = catalog_db.list_jobs(job_type="transcribe")
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "j11"

    def test_list_jobs_filter_by_run_id(self, catalog_db):
        """list_jobs(run_id=...) returns only matching run."""
        catalog_db.insert_job("j13", "ingest", "x", run_id="r1")
        catalog_db.insert_job("j14", "ingest", "x", run_id="r2")
        jobs = catalog_db.list_jobs(run_id="r1")
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "j13"

    def test_list_jobs_combined_filters(self, catalog_db):
        """list_jobs() with multiple filters uses AND logic."""
        catalog_db.insert_job("j15", "ingest", "x", status="done", run_id="r1")
        catalog_db.insert_job("j16", "ingest", "x", status="pending", run_id="r1")
        catalog_db.insert_job("j17", "analyze", "y", status="done", run_id="r1")
        jobs = catalog_db.list_jobs(stage="ingest", status="done", run_id="r1")
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "j15"

    def test_list_jobs_empty_result(self, catalog_db):
        """list_jobs() returns empty list when nothing matches."""
        jobs = catalog_db.list_jobs(stage="nonexistent")
        assert jobs == []

    def test_count_jobs_by_status_returns_dict(self, catalog_db):
        """count_jobs_by_status() returns {status: count} mapping."""
        catalog_db.insert_job("j18", "ingest", "x", status="pending")
        catalog_db.insert_job("j19", "ingest", "x", status="pending")
        catalog_db.insert_job("j20", "ingest", "x", status="done")
        counts = catalog_db.count_jobs_by_status("ingest")
        assert counts == {"pending": 2, "done": 1}

    def test_count_jobs_by_status_with_run_id_filter(self, catalog_db):
        """count_jobs_by_status() with run_id filters correctly."""
        catalog_db.insert_job("j21", "ingest", "x", status="done", run_id="r1")
        catalog_db.insert_job("j22", "ingest", "x", status="pending", run_id="r1")
        catalog_db.insert_job("j23", "ingest", "x", status="done", run_id="r2")
        counts = catalog_db.count_jobs_by_status("ingest", run_id="r1")
        assert counts == {"done": 1, "pending": 1}



# -- Events CRUD tests ------------------------------------------------------


class TestEventsCRUD:
    """Tests for pipeline events CRUD methods."""

    def test_insert_event_returns_event_id(self, catalog_db):
        """insert_event() returns the auto-generated event_id."""
        eid = catalog_db.insert_event("stage_started", stage="ingest")
        assert isinstance(eid, int)
        assert eid >= 1

    def test_insert_event_auto_creates_at(self, catalog_db):
        """insert_event() sets created_at via SQL DEFAULT."""
        eid = catalog_db.insert_event("stage_started")
        cur = catalog_db.conn.execute(
            "SELECT created_at FROM pipeline_events WHERE event_id = ?",
            (eid,),
        )
        row = cur.fetchone()
        assert row is not None
        assert row[0] is not None  # created_at was populated

    def test_insert_event_with_all_fields(self, catalog_db):
        """insert_event() stores all provided fields."""
        eid = catalog_db.insert_event(
            "job_completed",
            stage="analyze",
            job_id="j1",
            payload_json='{"result": "ok"}',
        )
        cur = catalog_db.conn.execute(
            "SELECT * FROM pipeline_events WHERE event_id = ?",
            (eid,),
        )
        row = dict(cur.fetchone())
        assert row["event_type"] == "job_completed"
        assert row["stage"] == "analyze"
        assert row["job_id"] == "j1"
        assert row["payload_json"] == '{"result": "ok"}'

    def test_insert_event_raises_on_none_lastrowid(self, catalog_db, monkeypatch):
        """insert_event() raises RuntimeError when lastrowid is None."""
        from unittest.mock import MagicMock

        mock_cursor = MagicMock()
        mock_cursor.lastrowid = None
        mock_conn = MagicMock()
        mock_conn.execute.return_value = mock_cursor

        monkeypatch.setattr(catalog_db, "conn", mock_conn)
        with pytest.raises(RuntimeError, match="lastrowid is None"):
            catalog_db.insert_event("stage_started", stage="ingest")

    def test_get_events_since_returns_newer_events(self, catalog_db):
        """get_events_since() returns only events after the given id."""
        e1 = catalog_db.insert_event("a")
        e2 = catalog_db.insert_event("b")
        e3 = catalog_db.insert_event("c")
        events = catalog_db.get_events_since(e1)
        ids = [e["event_id"] for e in events]
        assert ids == [e2, e3]

    def test_get_events_since_zero_returns_all(self, catalog_db):
        """get_events_since(0) returns all events."""
        catalog_db.insert_event("a")
        catalog_db.insert_event("b")
        events = catalog_db.get_events_since(0)
        assert len(events) == 2

    def test_get_events_since_empty(self, catalog_db):
        """get_events_since() returns empty list when no events match."""
        events = catalog_db.get_events_since(0)
        assert events == []

    def test_prune_events_removes_old(self, catalog_db):
        """prune_events() removes events older than threshold."""
        # Insert events with backdated created_at
        catalog_db.conn.execute(
            "INSERT INTO pipeline_events (event_type, created_at) "
            "VALUES (?, datetime('now', '-48 hours'))",
            ("old_event",),
        )
        catalog_db.insert_event("recent_event")
        catalog_db.prune_events(hours=24)
        events = catalog_db.get_events_since(0)
        assert len(events) == 1
        assert events[0]["event_type"] == "recent_event"

    def test_prune_events_keeps_recent(self, catalog_db):
        """prune_events() keeps events newer than threshold."""
        catalog_db.insert_event("recent1")
        catalog_db.insert_event("recent2")
        catalog_db.prune_events(hours=24)
        events = catalog_db.get_events_since(0)
        assert len(events) == 2

    @pytest.mark.parametrize(
        ("hours", "expected_param"),
        [(48, "-48 hours"), (1, "-1 hours")],
    )
    def test_prune_events_parameterizes_hours(
        self, catalog_db, hours, expected_param,
    ):
        """prune_events() uses parameterized SQL, not f-string interpolation."""
        # Seed a canary event backdated well beyond any threshold so prune
        # should always delete it — verifies the DELETE actually executes.
        catalog_db.conn.execute(
            "INSERT INTO pipeline_events (event_type, created_at) "
            "VALUES (?, datetime('now', '-72 hours'))",
            ("old_spy_canary",),
        )
        with patch.object(
            catalog_db, "conn", wraps=catalog_db.conn,
        ) as mock_conn:
            catalog_db.prune_events(hours=hours)

            # --- Data-level assertion: canary must be gone ---------------
            remaining = catalog_db.get_events_since(0)
            assert not any(
                e["event_type"] == "old_spy_canary" for e in remaining
            ), "old_spy_canary survived prune — DELETE did not execute"

            # --- Call-args assertions ------------------------------------
            # Extract positional args from each execute call; find the DELETE
            all_calls = [
                c.args for c in mock_conn.execute.call_args_list
            ]
            delete_calls = [c for c in all_calls if "DELETE" in c[0]]
            assert len(delete_calls) == 1
            sql_arg = delete_calls[0][0]
            # The literal hours value should NOT appear in the SQL string
            assert str(hours) not in sql_arg, (
                f"hours value interpolated into SQL: {sql_arg}"
            )
            # A parameters tuple/list must have been passed as second arg
            assert len(delete_calls[0]) >= 2, (
                "No parameters passed to execute — query is not parameterized"
            )
            # Verify the exact parameter value passed
            assert delete_calls[0][1] == (expected_param,), (
                f"Parameter value mismatch — expected ('{expected_param}',)"
            )

    def test_prune_events_rejects_zero_hours(self, catalog_db):
        """prune_events(hours=0) raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            catalog_db.prune_events(hours=0)

    def test_prune_events_rejects_negative_hours(self, catalog_db):
        """prune_events(hours=-5) raises ValueError mentioning the value."""
        with pytest.raises(ValueError, match="-5"):
            catalog_db.prune_events(hours=-5)

    @pytest.mark.parametrize(
        "bad_hours, expected_fragment",
        [
            (0.5, "0.5"),
            (1.5, "1.5"),
            (-0.5, "-0.5"),
            ("24", "'24'"),
            (None, "None"),
            (True, "True"),
            (False, "False"),
        ],
    )
    def test_prune_events_rejects_non_integer_hours(
        self, catalog_db, bad_hours, expected_fragment
    ):
        """prune_events() raises ValueError for non-integer hours values."""
        with pytest.raises(ValueError, match=expected_fragment):
            catalog_db.prune_events(hours=bad_hours)

    def test_prune_events_accepts_positive_hours(self, catalog_db):
        """prune_events(hours=1) does not raise for valid positive values."""
        catalog_db.insert_event("recent_event")
        catalog_db.prune_events(hours=1)  # should not raise
        events = catalog_db.get_events_since(0)
        assert len(events) == 1  # recent event is kept


# -- Runs CRUD tests --------------------------------------------------------


class TestRunsCRUD:
    """Tests for pipeline runs CRUD methods."""

    def test_insert_run_minimal(self, catalog_db):
        """insert_run() with only required fields uses defaults."""
        catalog_db.insert_run("r1", started_at="2026-01-01T00:00:00")
        run = catalog_db.get_run("r1")
        assert run is not None
        assert run["run_id"] == "r1"
        assert run["started_at"] == "2026-01-01T00:00:00"
        assert run["status"] == "running"
        assert run["config_snapshot"] is None

    def test_insert_run_all_fields(self, catalog_db):
        """insert_run() with all fields stores them correctly."""
        catalog_db.insert_run(
            "r2",
            started_at="2026-01-01T00:00:00",
            config_snapshot='{"key": "val"}',
            current_stage="ingest",
            status="running",
            budget_remaining_seconds=3600.0,
        )
        run = catalog_db.get_run("r2")
        assert run is not None
        assert run["config_snapshot"] == '{"key": "val"}'
        assert run["current_stage"] == "ingest"
        assert run["budget_remaining_seconds"] == 3600.0

    def test_get_run_returns_dict(self, catalog_db):
        """get_run() returns a plain dict."""
        catalog_db.insert_run("r3", started_at="2026-01-01T00:00:00")
        run = catalog_db.get_run("r3")
        assert isinstance(run, dict)

    def test_get_run_not_found(self, catalog_db):
        """get_run() returns None for a nonexistent run_id."""
        assert catalog_db.get_run("nonexistent") is None

    def test_update_run_modifies_fields(self, catalog_db):
        """update_run() changes specified fields."""
        catalog_db.insert_run("r4", started_at="2026-01-01T00:00:00")
        catalog_db.update_run(
            "r4",
            status="completed",
            finished_at="2026-01-01T01:00:00",
            wall_clock_seconds=3600.0,
        )
        run = catalog_db.get_run("r4")
        assert run is not None
        assert run["status"] == "completed"
        assert run["finished_at"] == "2026-01-01T01:00:00"
        assert run["wall_clock_seconds"] == 3600.0

    def test_get_current_run_returns_latest_running(self, catalog_db):
        """get_current_run() returns the most recently started running run."""
        catalog_db.insert_run("r5", started_at="2026-01-01T00:00:00")
        catalog_db.insert_run("r6", started_at="2026-01-02T00:00:00")
        current = catalog_db.get_current_run()
        assert current is not None
        assert current["run_id"] == "r6"

    def test_get_current_run_ignores_completed(self, catalog_db):
        """get_current_run() ignores runs with non-running status."""
        catalog_db.insert_run("r7", started_at="2026-01-01T00:00:00")
        catalog_db.update_run("r7", status="completed")
        catalog_db.insert_run("r8", started_at="2026-01-02T00:00:00")
        current = catalog_db.get_current_run()
        assert current is not None
        assert current["run_id"] == "r8"

    def test_get_current_run_returns_none_when_no_running(self, catalog_db):
        """get_current_run() returns None when all runs are completed."""
        catalog_db.insert_run("r9", started_at="2026-01-01T00:00:00", status="completed")
        assert catalog_db.get_current_run() is None

    def test_list_runs_ordered_by_started_at_desc(self, catalog_db):
        """list_runs() returns runs ordered by started_at descending."""
        catalog_db.insert_run("r10", started_at="2026-01-01T00:00:00")
        catalog_db.insert_run("r11", started_at="2026-01-03T00:00:00")
        catalog_db.insert_run("r12", started_at="2026-01-02T00:00:00")
        runs = catalog_db.list_runs()
        assert len(runs) == 3
        assert [r["run_id"] for r in runs] == ["r11", "r12", "r10"]



# -- Parametrized update-column validation tests ------------------------------


class TestUpdateColumnValidation:
    """Cross-entity tests for update method column rejection."""

    @pytest.mark.parametrize("bad_col_kind", ["generic", "specific"])
    @pytest.mark.parametrize("entity", list(_UPDATE_SPECS))
    def test_rejects_single_disallowed_column(
        self, catalog_db: CatalogDB, entity: str, bad_col_kind: str
    ) -> None:
        """update_*() raises ValueError naming the bad column."""
        spec = _UPDATE_SPECS[entity]
        bad_col = "evil_col" if bad_col_kind == "generic" else spec["invalid_col"]
        spec["setup"](catalog_db)
        with pytest.raises(ValueError, match=bad_col):
            spec["update"](catalog_db, **{bad_col: "bad"})
        # Row must be untouched (atomicity): disallowed-only call issues no SQL.
        row = spec["get"](catalog_db)
        assert row is not None
        assert row[spec["valid_col"]] == spec["default_val"]

    @pytest.mark.parametrize("entity", list(_UPDATE_SPECS))
    def test_rejects_mix_of_valid_and_invalid(
        self, catalog_db: CatalogDB, entity: str
    ) -> None:
        """update_*() raises ValueError and does not apply valid columns."""
        spec = _UPDATE_SPECS[entity]
        spec["setup"](catalog_db)
        with pytest.raises(ValueError, match="Disallowed column"):
            spec["update"](
                catalog_db,
                **{spec["valid_col"]: spec["valid_val"], "evil_col": "bad"},
            )
        # Valid column must NOT have been applied (atomicity).
        row = spec["get"](catalog_db)
        assert row is not None
        assert row[spec["valid_col"]] == spec["default_val"]

    @pytest.mark.parametrize("entity", list(_UPDATE_SPECS))
    def test_rejects_multiple_disallowed_columns(
        self, catalog_db: CatalogDB, entity: str
    ) -> None:
        """update_*() names all bad columns in sorted order when >1 are passed."""
        spec = _UPDATE_SPECS[entity]
        spec["setup"](catalog_db)
        with pytest.raises(ValueError, match="Disallowed column") as exc_info:
            spec["update"](catalog_db, zzz_col="y", aaa_col="x")
        msg = str(exc_info.value)
        # Both bad column names appear in the error message.
        assert "aaa_col" in msg
        assert "zzz_col" in msg
        # sorted() order: 'aaa_col' appears before 'zzz_col'.
        assert msg.index("aaa_col") < msg.index("zzz_col")
        # Row must still exist (atomicity): disallowed-only call issues no SQL.
        row = spec["get"](catalog_db)
        assert row is not None


# -- Cross-table integration tests ------------------------------------------


class TestPipelineCrossTable:
    """Integration tests spanning multiple pipeline tables."""

    def test_jobs_reference_run_id(self, catalog_db):
        """Jobs with run_id can be counted by status for that run."""
        catalog_db.insert_run("r1", started_at="2026-01-01T00:00:00")
        catalog_db.insert_job("j1", "ingest", "import", status="done", run_id="r1")
        catalog_db.insert_job("j2", "ingest", "import", status="pending", run_id="r1")
        catalog_db.insert_job("j3", "ingest", "import", status="done", run_id="r2")
        counts = catalog_db.count_jobs_by_status("ingest", run_id="r1")
        assert counts == {"done": 1, "pending": 1}

    def test_events_reference_job_id(self, catalog_db):
        """Events can store and retrieve a job_id reference."""
        catalog_db.insert_job("j10", "analyze", "transcribe")
        eid = catalog_db.insert_event("job_started", stage="analyze", job_id="j10")
        events = catalog_db.get_events_since(eid - 1)
        assert len(events) == 1
        assert events[0]["job_id"] == "j10"

    def test_update_nonexistent_gate_is_noop(self, catalog_db):
        """update_gate() on a missing stage doesn't error."""
        catalog_db.update_gate("nonexistent", mode="manual")
        # No error raised; nothing inserted
        assert catalog_db.get_gate("nonexistent") is None

    def test_update_nonexistent_job_is_noop(self, catalog_db):
        """update_job() on a missing job_id doesn't error."""
        catalog_db.update_job("nonexistent", status="done")
        # No error raised; nothing inserted
        assert catalog_db.get_job("nonexistent") is None
