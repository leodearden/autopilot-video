"""Tests for rowcount return values from CatalogDB update methods.

Each update_* method wraps _execute_kwargs_update which returns an int
rowcount.  These tests verify the contract: 1 for an existing row,
0 for a nonexistent primary key, 0 for empty kwargs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator
from unittest.mock import MagicMock

import pytest

from autopilot.db import CatalogDB

if TYPE_CHECKING:
    pass

# -- fixtures ----------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> Iterator[CatalogDB]:
    """Create a CatalogDB with a temp file-backed database."""
    _db = CatalogDB(str(tmp_path / "catalog.db"))
    yield _db  # type: ignore[misc]
    _db.close()


# -- helpers -----------------------------------------------------------------

def _seed_narrative(
    db: CatalogDB,
    narrative_id: str = "n-1",
    **overrides: object,
) -> None:
    """Insert a narrative with sensible defaults."""
    defaults: dict[str, object] = {
        "title": "Morning Walk",
        "description": "A walk in the park",
        "proposed_duration_seconds": 120.0,
        "activity_cluster_ids_json": '["c-1","c-2"]',
        "arc_notes": "peaceful start",
        "emotional_journey": "calm \u2192 happy",
        "status": "proposed",
    }
    defaults.update(overrides)
    db.insert_narrative(narrative_id, **defaults)  # type: ignore[arg-type]
    db.conn.commit()


# ---------------------------------------------------------------------------
# TestUpdateNarrativeRowcount — task-278 step-1
# ---------------------------------------------------------------------------

class TestUpdateNarrativeRowcount:
    """Tests that update_narrative returns int rowcount."""

    def test_returns_1_for_existing_narrative(self, db: CatalogDB) -> None:
        """update_narrative returns 1 when updating an existing row."""
        _seed_narrative(db, "n-1", title="Old")
        result = db.update_narrative("n-1", title="New")
        assert result == 1

    def test_returns_0_for_nonexistent_narrative(self, db: CatalogDB) -> None:
        """update_narrative returns 0 for a non-existent narrative_id."""
        result = db.update_narrative("nonexistent", title="X")
        assert result == 0

    def test_returns_0_for_empty_kwargs(self, db: CatalogDB) -> None:
        """update_narrative returns 0 for empty kwargs (early return)."""
        _seed_narrative(db, "n-1")
        result = db.update_narrative("n-1")
        assert result == 0


# ---------------------------------------------------------------------------
# TestUpdateGateRowcount — task-278 step-3
# ---------------------------------------------------------------------------

class TestUpdateGateRowcount:
    """Tests that update_gate returns int rowcount."""

    def test_returns_1_for_existing_gate(self, db: CatalogDB) -> None:
        """update_gate returns 1 when updating an existing gate row."""
        db.init_default_gates()
        db.conn.commit()
        result = db.update_gate("classify", mode="auto")
        assert result == 1

    def test_returns_0_for_nonexistent_gate(self, db: CatalogDB) -> None:
        """update_gate returns 0 for a non-existent stage."""
        result = db.update_gate("nonexistent", mode="auto")
        assert result == 0

    def test_returns_0_for_empty_kwargs(self, db: CatalogDB) -> None:
        """update_gate returns 0 for empty kwargs (early return)."""
        db.init_default_gates()
        db.conn.commit()
        result = db.update_gate("classify")
        assert result == 0


# ---------------------------------------------------------------------------
# TestUpdateJobRowcount — task-278 step-5
# ---------------------------------------------------------------------------

class TestUpdateJobRowcount:
    """Tests that update_job returns int rowcount."""

    def test_returns_1_for_existing_job(self, db: CatalogDB) -> None:
        """update_job returns 1 when updating an existing row."""
        db.insert_job("j-1", "ingest", "media_import")
        db.conn.commit()
        result = db.update_job("j-1", status="running")
        assert result == 1

    def test_returns_0_for_nonexistent_job(self, db: CatalogDB) -> None:
        """update_job returns 0 for a non-existent job_id."""
        result = db.update_job("nonexistent", status="running")
        assert result == 0

    def test_returns_0_for_empty_kwargs(self, db: CatalogDB) -> None:
        """update_job returns 0 for empty kwargs (early return)."""
        db.insert_job("j-1", "ingest", "media_import")
        db.conn.commit()
        result = db.update_job("j-1")
        assert result == 0


# ---------------------------------------------------------------------------
# TestUpdateRunRowcount — task-278 step-7
# ---------------------------------------------------------------------------

class TestUpdateRunRowcount:
    """Tests that update_run returns int rowcount."""

    def test_returns_1_for_existing_run(self, db: CatalogDB) -> None:
        """update_run returns 1 when updating an existing row."""
        db.insert_run("r-1", started_at="2026-01-01T00:00:00")
        db.conn.commit()
        result = db.update_run("r-1", status="finished")
        assert result == 1

    def test_returns_0_for_nonexistent_run(self, db: CatalogDB) -> None:
        """update_run returns 0 for a non-existent run_id."""
        result = db.update_run("nonexistent", status="finished")
        assert result == 0

    def test_returns_0_for_empty_kwargs(self, db: CatalogDB) -> None:
        """update_run returns 0 for empty kwargs (early return)."""
        db.insert_run("r-1", started_at="2026-01-01T00:00:00")
        db.conn.commit()
        result = db.update_run("r-1")
        assert result == 0


# ---------------------------------------------------------------------------
# TestRowcountSentinelGuard — task-308 step-1
# ---------------------------------------------------------------------------

class TestRowcountSentinelGuard:
    """Guard: rowcount=-1 sentinel from sqlite3 should be clamped to 0."""

    def test_negative_rowcount_clamped_to_zero(self, db: CatalogDB) -> None:
        """When cursor.rowcount is -1, _execute_kwargs_update returns 0."""
        _seed_narrative(db, "n-1", title="Original")

        mock_cursor = MagicMock()
        mock_cursor.rowcount = -1

        from unittest.mock import patch

        with patch.object(db, "conn") as mock_conn:
            mock_conn.execute.return_value = mock_cursor
            result = db.update_narrative("n-1", title="Updated")

        assert result == 0, (
            "Expected 0 when cursor.rowcount is -1 (sentinel), "
            f"but got {result}"
        )
