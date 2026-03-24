"""Tests for review hub and narrative review routes + DB methods."""

from __future__ import annotations

from pathlib import Path

import pytest

from autopilot.db import CatalogDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path) -> CatalogDB:
    """Create a CatalogDB with a temp file-backed database."""
    db_path = str(tmp_path / "catalog.db")
    _db = CatalogDB(db_path)
    yield _db  # type: ignore[misc]
    _db.close()


def _seed_narrative(db: CatalogDB, narrative_id: str = "n-1", **overrides: object) -> None:
    """Insert a narrative with sensible defaults, overridable via kwargs."""
    defaults: dict[str, object] = {
        "title": "Morning Walk",
        "description": "A walk in the park",
        "proposed_duration_seconds": 120.0,
        "activity_cluster_ids_json": '["c-1","c-2"]',
        "arc_notes": "peaceful start",
        "emotional_journey": "calm → happy",
        "status": "proposed",
    }
    defaults.update(overrides)
    db.insert_narrative(narrative_id, **defaults)  # type: ignore[arg-type]
    db.conn.commit()


# ---------------------------------------------------------------------------
# TestUpdateNarrative — step-1
# ---------------------------------------------------------------------------

class TestUpdateNarrative:
    """Tests for CatalogDB.update_narrative(**kwargs)."""

    def test_update_title(self, db: CatalogDB) -> None:
        """update_narrative updates the title field."""
        _seed_narrative(db, "n-1", title="Old Title")
        db.update_narrative("n-1", title="New Title")
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "New Title"

    def test_update_description(self, db: CatalogDB) -> None:
        """update_narrative updates the description field."""
        _seed_narrative(db, "n-1", description="Old Desc")
        db.update_narrative("n-1", description="New Desc")
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["description"] == "New Desc"

    def test_update_proposed_duration_seconds(self, db: CatalogDB) -> None:
        """update_narrative updates proposed_duration_seconds."""
        _seed_narrative(db, "n-1", proposed_duration_seconds=60.0)
        db.update_narrative("n-1", proposed_duration_seconds=180.0)
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["proposed_duration_seconds"] == 180.0

    def test_leaves_unmentioned_fields_unchanged(self, db: CatalogDB) -> None:
        """update_narrative only updates explicitly provided fields."""
        _seed_narrative(db, "n-1", title="Keep Me", description="Change Me")
        db.update_narrative("n-1", description="Changed")
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "Keep Me"
        assert row["description"] == "Changed"

    def test_noop_on_empty_kwargs(self, db: CatalogDB) -> None:
        """update_narrative with no kwargs is a no-op (no error)."""
        _seed_narrative(db, "n-1", title="Same")
        db.update_narrative("n-1")
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "Same"

    def test_update_multiple_fields(self, db: CatalogDB) -> None:
        """update_narrative can update multiple fields at once."""
        _seed_narrative(db, "n-1", title="Old", description="Old", proposed_duration_seconds=60.0)
        db.update_narrative("n-1", title="New", description="New", proposed_duration_seconds=90.0)
        db.conn.commit()
        row = db.get_narrative("n-1")
        assert row is not None
        assert row["title"] == "New"
        assert row["description"] == "New"
        assert row["proposed_duration_seconds"] == 90.0
