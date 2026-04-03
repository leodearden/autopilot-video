"""Tests for shared test helpers in conftest.py.

Validates the consolidated helpers (parse_sse_body, seed_narrative,
seed_cluster, PIPELINE_STAGES) behave correctly, including the
multi-data-line SSE fix.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from autopilot.db import CatalogDB
from tests.conftest import (
    PIPELINE_STAGES,
    _parse_sse_body,
    _seed_cluster,
    _seed_narrative,
    parse_sse_body,
    seed_cluster,
    seed_narrative,
)

# ---------------------------------------------------------------------------
# PIPELINE_STAGES
# ---------------------------------------------------------------------------


class TestPipelineStages:
    """Tests for the PIPELINE_STAGES constant."""

    def test_has_nine_stages(self) -> None:
        """PIPELINE_STAGES contains exactly 9 stages."""
        assert len(PIPELINE_STAGES) == 9

    def test_is_tuple(self) -> None:
        """PIPELINE_STAGES is a tuple (immutable)."""
        assert isinstance(PIPELINE_STAGES, tuple)

    def test_expected_stages(self) -> None:
        """PIPELINE_STAGES contains all expected stage names."""
        expected = {
            "ingest", "analyze", "classify", "narrate", "script",
            "edl", "source", "render", "upload",
        }
        assert set(PIPELINE_STAGES) == expected


# ---------------------------------------------------------------------------
# parse_sse_body
# ---------------------------------------------------------------------------


class TestParseSSEBody:
    """Tests for parse_sse_body / _parse_sse_body."""

    def test_aliases_match(self) -> None:
        """_parse_sse_body is an alias for parse_sse_body."""
        assert _parse_sse_body is parse_sse_body

    def test_parses_single_event(self) -> None:
        """Parses a single SSE event with id, event, and data fields."""
        text = "id: 1\nevent: stage_started\ndata: {\"stage\":\"INGEST\"}\n\n"
        events = parse_sse_body(text)
        assert len(events) == 1
        assert events[0]["id"] == "1"
        assert events[0]["event"] == "stage_started"
        assert events[0]["data"] == '{"stage":"INGEST"}'

    def test_parses_multiple_events(self) -> None:
        """Parses multiple SSE events separated by blank lines."""
        text = (
            "id: 1\nevent: stage_started\ndata: first\n\n"
            "id: 2\nevent: stage_completed\ndata: second\n\n"
        )
        events = parse_sse_body(text)
        assert len(events) == 2
        assert events[0]["data"] == "first"
        assert events[1]["data"] == "second"

    def test_multi_data_lines_concatenated(self) -> None:
        """Multiple data: lines within one event are joined with newlines.

        This is the multi-data-line bug fix: previously each data: line
        overwrote the previous one.  Per the SSE spec, multiple data: lines
        should be concatenated with newlines.
        """
        text = (
            "id: 1\n"
            "event: message\n"
            "data: line one\n"
            "data: line two\n"
            "data: line three\n"
            "\n"
        )
        events = parse_sse_body(text)
        assert len(events) == 1
        assert events[0]["data"] == "line one\nline two\nline three"

    def test_empty_input(self) -> None:
        """Empty input returns empty list."""
        assert parse_sse_body("") == []

    def test_trailing_event_without_blank_line(self) -> None:
        """Event at end of stream without trailing blank line is still captured."""
        text = "id: 1\nevent: test\ndata: payload"
        events = parse_sse_body(text)
        assert len(events) == 1
        assert events[0]["data"] == "payload"


# ---------------------------------------------------------------------------
# seed_narrative
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> Iterator[CatalogDB]:
    """Create a CatalogDB with a temp file-backed database."""
    db_path = str(tmp_path / "catalog.db")
    _db = CatalogDB(db_path)
    yield _db
    _db.close()


class TestSeedNarrative:
    """Tests for seed_narrative / _seed_narrative."""

    def test_aliases_match(self) -> None:
        """_seed_narrative is an alias for seed_narrative."""
        assert _seed_narrative is seed_narrative

    def test_inserts_with_defaults(self, db: CatalogDB) -> None:
        """seed_narrative inserts a narrative with sensible defaults."""
        seed_narrative(db, "n-1")
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "Morning Walk"
        assert row["status"] == "proposed"

    def test_overrides_work(self, db: CatalogDB) -> None:
        """seed_narrative accepts keyword overrides."""
        seed_narrative(db, "n-1", title="Custom Title", status="approved")
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "Custom Title"
        assert row["status"] == "approved"

    def test_commit_true_by_default(self, db: CatalogDB) -> None:
        """With commit=True (default), data survives a rollback."""
        seed_narrative(db, "n-1")
        # Since commit was called, data is persisted even if we rollback
        db.conn.rollback()
        row = db.get_narrative("n-1")
        assert row is not None

    def test_commit_false_skips_commit(self, db: CatalogDB) -> None:
        """With commit=False, data is NOT auto-committed."""
        # Use a connection with non-autocommit mode
        db.conn.isolation_level = "DEFERRED"  # non-autocommit
        seed_narrative(db, "n-1", commit=False)
        db.conn.rollback()
        row = db.get_narrative("n-1")
        assert row is None


# ---------------------------------------------------------------------------
# seed_cluster
# ---------------------------------------------------------------------------


class TestSeedCluster:
    """Tests for seed_cluster / _seed_cluster."""

    def test_aliases_match(self) -> None:
        """_seed_cluster is an alias for seed_cluster."""
        assert _seed_cluster is seed_cluster

    def test_inserts_with_defaults(self, db: CatalogDB) -> None:
        """seed_cluster inserts a cluster with sensible defaults."""
        seed_cluster(db, "c-1")
        row = db.get_activity_cluster("c-1")
        assert row is not None
        assert row["label"] == "Morning Activity"
        assert row["location_label"] == "Park"

    def test_overrides_work(self, db: CatalogDB) -> None:
        """seed_cluster accepts keyword overrides."""
        seed_cluster(db, "c-1", label="Beach Walk", location_label="Beach")
        row = db.get_activity_cluster("c-1")
        assert row is not None
        assert row["label"] == "Beach Walk"
        assert row["location_label"] == "Beach"

    def test_commit_false_skips_commit(self, db: CatalogDB) -> None:
        """With commit=False, data is NOT auto-committed."""
        db.conn.isolation_level = "DEFERRED"  # non-autocommit
        seed_cluster(db, "c-1", commit=False)
        db.conn.rollback()
        row = db.get_activity_cluster("c-1")
        assert row is None


# ---------------------------------------------------------------------------
# seed_cluster_via_app
# ---------------------------------------------------------------------------


class TestSeedClusterViaApp:
    """Tests for seed_cluster_via_app / _seed_cluster_via_db alias."""

    def test_importable(self) -> None:
        """seed_cluster_via_app and _seed_cluster_via_db are importable from conftest."""
        from tests.conftest import _seed_cluster_via_db, seed_cluster_via_app
        assert callable(seed_cluster_via_app)
        assert _seed_cluster_via_db is seed_cluster_via_app

    def test_inserts_with_defaults(self, tmp_path: Path) -> None:
        """seed_cluster_via_app inserts a cluster into the app's DB with defaults."""
        from autopilot.web.app import create_app
        from tests.conftest import seed_cluster_via_app

        db_path = str(tmp_path / "catalog.db")
        # Initialize the DB so tables exist
        init_db = CatalogDB(db_path)
        init_db.close()

        app = create_app(db_path)
        seed_cluster_via_app(app, "c-1")

        # Read back from DB to verify
        verify_db = CatalogDB(db_path)
        row = verify_db.get_activity_cluster("c-1")
        verify_db.close()

        assert row is not None
        assert row["label"] == "Morning Activity"
        assert row["location_label"] == "Park"

    def test_overrides_work(self, tmp_path: Path) -> None:
        """seed_cluster_via_app accepts keyword overrides."""
        from autopilot.web.app import create_app
        from tests.conftest import seed_cluster_via_app

        db_path = str(tmp_path / "catalog.db")
        init_db = CatalogDB(db_path)
        init_db.close()

        app = create_app(db_path)
        seed_cluster_via_app(app, "c-1", label="Beach Walk", location_label="Beach")

        verify_db = CatalogDB(db_path)
        row = verify_db.get_activity_cluster("c-1")
        verify_db.close()

        assert row is not None
        assert row["label"] == "Beach Walk"
        assert row["location_label"] == "Beach"
