"""Tests for the media list page and API endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest

from autopilot.db import CatalogDB


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog_db():
    """In-memory CatalogDB with autocommit for test convenience."""
    db = CatalogDB(":memory:")
    db.conn.isolation_level = None  # autocommit
    yield db
    db.close()


# ---------------------------------------------------------------------------
# has_captions tests
# ---------------------------------------------------------------------------


class TestHasCaptions:
    """Tests for CatalogDB.has_captions(media_id)."""

    def test_has_captions_returns_false_when_no_captions(self, catalog_db: CatalogDB) -> None:
        """has_captions returns False when no caption rows exist for the media_id."""
        catalog_db.insert_media("m1", "/video/beach.mp4")
        assert catalog_db.has_captions("m1") is False

    def test_has_captions_returns_true_after_inserting_caption(self, catalog_db: CatalogDB) -> None:
        """has_captions returns True after a caption is inserted for the media_id."""
        catalog_db.insert_media("m1", "/video/beach.mp4")
        catalog_db.upsert_caption("m1", 0.0, 5.0, "A beach scene", "blip2")
        assert catalog_db.has_captions("m1") is True

    def test_has_captions_does_not_match_other_media(self, catalog_db: CatalogDB) -> None:
        """has_captions returns False for a media_id that has no captions, even if another does."""
        catalog_db.insert_media("m1", "/video/beach.mp4")
        catalog_db.insert_media("m2", "/video/forest.mp4")
        catalog_db.upsert_caption("m1", 0.0, 5.0, "A beach scene", "blip2")
        assert catalog_db.has_captions("m2") is False
