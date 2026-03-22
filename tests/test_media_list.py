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


# ---------------------------------------------------------------------------
# API endpoint fixtures (real DB file so the app opens its own connection)
# ---------------------------------------------------------------------------


@pytest.fixture
def media_db(tmp_path: Path) -> CatalogDB:
    """Create a CatalogDB backed by a real file for web endpoint tests."""
    db_path = str(tmp_path / "catalog.db")
    db = CatalogDB(db_path)
    db.conn.isolation_level = None  # autocommit
    db._test_path = db_path  # stash for fixture chaining
    return db


@pytest.fixture
def seeded_db(media_db: CatalogDB) -> CatalogDB:
    """media_db pre-seeded with 5 media files + analysis data on m0."""
    for i in range(5):
        mid = f"m{i}"
        media_db.insert_media(
            mid,
            f"/video/clip_{i:03d}.mp4",
            duration_seconds=float(60 + i),
            resolution_w=1920,
            resolution_h=1080,
            created_at=f"2025-01-{(i + 1):02d}T00:00:00",
            status="ingested" if i % 2 == 0 else "analyzed",
        )
    # Analysis data on m0
    media_db.upsert_transcript("m0", "[]", "en")
    media_db.batch_insert_detections([("m0", 0, "[]")])
    return media_db


@pytest.fixture
def media_app(seeded_db: CatalogDB):
    """FastAPI app pointing at the seeded DB."""
    from autopilot.web.app import create_app

    return create_app(seeded_db._test_path)


@pytest.fixture
def media_client(media_app):
    """TestClient for media endpoint tests."""
    from starlette.testclient import TestClient

    return TestClient(media_app)


# ---------------------------------------------------------------------------
# GET /api/media JSON tests
# ---------------------------------------------------------------------------


class TestApiMediaJson:
    """Tests for GET /api/media JSON endpoint."""

    def test_api_media_returns_json(self, media_client) -> None:
        """GET /api/media returns 200 with JSON containing 'items' and 'total'."""
        resp = media_client.get("/api/media")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] == 5

    def test_api_media_pagination_params(self, media_client) -> None:
        """?page=2&per_page=2 returns correct subset."""
        resp = media_client.get("/api/media?page=2&per_page=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        assert data["total"] == 5

    def test_api_media_filter_by_status(self, media_client) -> None:
        """?status=analyzed filters correctly."""
        resp = media_client.get("/api/media?status=analyzed")
        assert resp.status_code == 200
        data = resp.json()
        assert all(item["status"] == "analyzed" for item in data["items"])

    def test_api_media_text_search(self, media_client, seeded_db: CatalogDB) -> None:
        """?q=clip_000 filters by filename substring."""
        resp = media_client.get("/api/media?q=clip_000")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert "clip_000" in data["items"][0]["file_path"]

    def test_api_media_sort(self, media_client) -> None:
        """?sort=file_path&order=asc returns sorted results."""
        resp = media_client.get("/api/media?sort=file_path&order=asc")
        assert resp.status_code == 200
        data = resp.json()
        paths = [item["file_path"] for item in data["items"]]
        assert paths == sorted(paths)

    def test_api_media_items_have_analysis_flags(self, media_client) -> None:
        """Each item has has_transcript, has_detections, etc. boolean fields."""
        resp = media_client.get("/api/media")
        data = resp.json()
        item = data["items"][0]
        for flag in (
            "has_transcript",
            "has_detections",
            "has_faces",
            "has_embeddings",
            "has_audio_events",
            "has_captions",
        ):
            assert flag in item, f"Missing flag: {flag}"
            assert isinstance(item[flag], bool), f"{flag} should be bool"


# ---------------------------------------------------------------------------
# GET /media HTML page tests
# ---------------------------------------------------------------------------


class TestMediaPage:
    """Tests for GET /media HTML page."""

    def test_media_page_returns_html(self, media_client) -> None:
        """GET /media returns 200 with text/html content-type."""
        resp = media_client.get("/media")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_media_page_extends_base(self, media_client) -> None:
        """Response contains nav elements from base.html."""
        resp = media_client.get("/media")
        assert "Autopilot Video" in resp.text

    def test_media_page_contains_table(self, media_client) -> None:
        """Response contains a table with expected column headers."""
        resp = media_client.get("/media")
        html = resp.text
        assert "<table" in html
        for header in ("Filename", "Duration", "Resolution", "Status"):
            assert header in html, f"Column header '{header}' not found"

    def test_media_page_contains_filter_controls(self, media_client) -> None:
        """Response contains filter inputs (text search, status dropdown)."""
        resp = media_client.get("/media")
        html = resp.text
        # Text search input
        assert 'name="q"' in html or 'id="search"' in html
        # Status dropdown
        assert "<select" in html

    def test_media_page_has_htmx_attributes(self, media_client) -> None:
        """Filter elements have hx-get and hx-target for partial swap."""
        resp = media_client.get("/media")
        html = resp.text
        assert "hx-get" in html
        assert "hx-target" in html


# ---------------------------------------------------------------------------
# HTMX partial response tests
# ---------------------------------------------------------------------------


class TestHtmxPartial:
    """Tests for HTMX partial response from GET /api/media."""

    def test_api_media_htmx_returns_partial(self, media_client) -> None:
        """GET /api/media with HX-Request:true returns HTML table rows, not JSON."""
        resp = media_client.get("/api/media", headers={"HX-Request": "true"})
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        # Should contain <tr> elements, not JSON
        assert "<tr" in resp.text

    def test_partial_media_row_has_analysis_badges(self, media_client) -> None:
        """Partial response has analysis badge indicators for media with analysis data."""
        # seeded_db has transcript + detections on m0
        resp = media_client.get("/api/media", headers={"HX-Request": "true"})
        html = resp.text
        assert "ASR" in html  # transcript badge
        assert "YOLO" in html  # detections badge

    def test_partial_media_row_duration_format(self, media_client) -> None:
        """Duration is formatted as M:SS or H:MM:SS, not raw seconds."""
        resp = media_client.get("/api/media", headers={"HX-Request": "true"})
        html = resp.text
        # seeded items have 60-64 seconds → "1:00", "1:01", etc.
        assert "1:00" in html

    def test_partial_media_row_resolution_display(self, media_client) -> None:
        """Resolution displayed as WxH."""
        resp = media_client.get("/api/media", headers={"HX-Request": "true"})
        assert "1920x1080" in resp.text
