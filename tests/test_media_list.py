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


# ---------------------------------------------------------------------------
# query_media tests
# ---------------------------------------------------------------------------


def _seed_media(db: CatalogDB, count: int = 3) -> list[str]:
    """Insert ``count`` media rows and return their ids."""
    ids = []
    for i in range(count):
        mid = f"m{i}"
        db.insert_media(
            mid,
            f"/video/clip_{i:03d}.mp4",
            duration_seconds=float(60 + i),
            resolution_w=1920,
            resolution_h=1080,
            created_at=f"2025-01-{(i + 1):02d}T00:00:00",
            status="ingested" if i % 2 == 0 else "analyzed",
        )
        ids.append(mid)
    return ids


class TestQueryMedia:
    """Tests for CatalogDB.query_media()."""

    def test_query_media_returns_all_when_no_filters(self, catalog_db: CatalogDB) -> None:
        """query_media with no filters returns all media and total count."""
        _seed_media(catalog_db, 3)
        result = catalog_db.query_media()
        assert len(result["items"]) == 3
        assert result["total"] == 3

    def test_query_media_pagination(self, catalog_db: CatalogDB) -> None:
        """query_media with page/per_page returns correct subset."""
        _seed_media(catalog_db, 5)
        result = catalog_db.query_media(page=2, per_page=2)
        assert len(result["items"]) == 2
        assert result["total"] == 5

    def test_query_media_text_filter(self, catalog_db: CatalogDB) -> None:
        """query_media with q= filters on file_path LIKE."""
        catalog_db.insert_media("beach1", "/video/beach_sunset.mp4")
        catalog_db.insert_media("forest1", "/video/forest_walk.mp4")
        catalog_db.insert_media("beach2", "/video/beach_waves.mp4")
        result = catalog_db.query_media(q="beach")
        assert result["total"] == 2
        paths = [item["file_path"] for item in result["items"]]
        assert all("beach" in p for p in paths)

    def test_query_media_status_filter(self, catalog_db: CatalogDB) -> None:
        """query_media with status= filters by status column."""
        _seed_media(catalog_db, 4)  # alternating ingested/analyzed
        result = catalog_db.query_media(status="analyzed")
        assert result["total"] == 2
        assert all(item["status"] == "analyzed" for item in result["items"])

    def test_query_media_sort_by_file_path_asc(self, catalog_db: CatalogDB) -> None:
        """query_media sorts by file_path ascending."""
        catalog_db.insert_media("a", "/video/alpha.mp4")
        catalog_db.insert_media("b", "/video/beta.mp4")
        catalog_db.insert_media("c", "/video/gamma.mp4")
        result = catalog_db.query_media(sort="file_path", order="asc")
        paths = [item["file_path"] for item in result["items"]]
        assert paths == sorted(paths)

    def test_query_media_sort_by_file_path_desc(self, catalog_db: CatalogDB) -> None:
        """query_media sorts by file_path descending."""
        catalog_db.insert_media("a", "/video/alpha.mp4")
        catalog_db.insert_media("b", "/video/beta.mp4")
        catalog_db.insert_media("c", "/video/gamma.mp4")
        result = catalog_db.query_media(sort="file_path", order="desc")
        paths = [item["file_path"] for item in result["items"]]
        assert paths == sorted(paths, reverse=True)

    def test_query_media_sort_by_created_at(self, catalog_db: CatalogDB) -> None:
        """query_media sorts by created_at."""
        catalog_db.insert_media("c", "/c.mp4", created_at="2025-03-01T00:00:00")
        catalog_db.insert_media("a", "/a.mp4", created_at="2025-01-01T00:00:00")
        catalog_db.insert_media("b", "/b.mp4", created_at="2025-02-01T00:00:00")
        result = catalog_db.query_media(sort="created_at", order="asc")
        dates = [item["created_at"] for item in result["items"]]
        assert dates == sorted(dates)

    def test_query_media_analysis_flags(self, catalog_db: CatalogDB) -> None:
        """query_media returns has_* flags for each analysis type."""
        catalog_db.insert_media("m1", "/video/full.mp4")
        catalog_db.insert_media("m2", "/video/empty.mp4")

        # Seed analysis data for m1
        catalog_db.upsert_transcript("m1", "[]", "en")
        catalog_db.batch_insert_detections([("m1", 0, "[]")])
        catalog_db.batch_insert_faces([("m1", 0, 0, "{}", None, None)])
        catalog_db.batch_insert_embeddings([("m1", 0, b"\x00")])
        catalog_db.batch_insert_audio_events([("m1", 0.0, "[]")])
        catalog_db.upsert_caption("m1", 0.0, 5.0, "test", "blip2")

        result = catalog_db.query_media(sort="file_path", order="desc")
        items = {item["id"]: item for item in result["items"]}

        # m1 should have all flags True
        m1 = items["m1"]
        assert m1["has_transcript"] is True
        assert m1["has_detections"] is True
        assert m1["has_faces"] is True
        assert m1["has_embeddings"] is True
        assert m1["has_audio_events"] is True
        assert m1["has_captions"] is True

        # m2 should have all flags False
        m2 = items["m2"]
        assert m2["has_transcript"] is False
        assert m2["has_detections"] is False
        assert m2["has_faces"] is False
        assert m2["has_embeddings"] is False
        assert m2["has_audio_events"] is False
        assert m2["has_captions"] is False
