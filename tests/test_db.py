"""Tests for autopilot.db CatalogDB class."""

from __future__ import annotations

import sqlite3
import threading

import pytest


# -- CatalogDB class basics --------------------------------------------------


class TestCatalogDBBasics:
    """Tests for CatalogDB connection management and lifecycle."""

    def test_catalogdb_creates_connection(self, catalog_db):
        """CatalogDB with ':memory:' creates a valid connection."""
        assert catalog_db.conn is not None
        assert isinstance(catalog_db.conn, sqlite3.Connection)

    def test_catalogdb_wal_mode(self, tmp_path):
        """CatalogDB on a file-based DB enables WAL journal mode."""
        from autopilot.db import CatalogDB

        db_path = str(tmp_path / "test.db")
        db = CatalogDB(db_path)
        try:
            cur = db.conn.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            assert mode == "wal", f"Expected WAL mode, got {mode}"
        finally:
            db.close()

    def test_catalogdb_thread_safety(self, catalog_db):
        """CatalogDB connection works from a different thread."""
        results: list[bool] = []

        def access_db():
            try:
                catalog_db.conn.execute("SELECT 1")
                results.append(True)
            except sqlite3.ProgrammingError:
                results.append(False)

        t = threading.Thread(target=access_db)
        t.start()
        t.join()
        assert results == [True], "Connection should be usable from another thread"

    def test_catalogdb_context_manager(self):
        """Context manager __enter__ returns self, __exit__ commits."""
        from autopilot.db import CatalogDB

        db = CatalogDB(":memory:")
        try:
            with db as ctx:
                assert ctx is db
                # Create a temp table and insert inside context manager
                ctx.conn.execute(
                    "CREATE TABLE IF NOT EXISTS _test (val TEXT)"
                )
                ctx.conn.execute("INSERT INTO _test VALUES ('hello')")
            # After __exit__, data should be committed
            cur = db.conn.execute("SELECT val FROM _test")
            assert cur.fetchone()[0] == "hello"
        finally:
            db.close()

    def test_catalogdb_close(self, catalog_db):
        """close() closes the underlying connection."""
        catalog_db.close()
        with pytest.raises(sqlite3.ProgrammingError):
            catalog_db.conn.execute("SELECT 1")


# -- Schema creation ----------------------------------------------------------

EXPECTED_TABLES = sorted([
    "media_files",
    "transcripts",
    "shot_boundaries",
    "detections",
    "face_clusters",
    "clip_embeddings",
    "audio_events",
    "activity_clusters",
    "narratives",
    "edit_plans",
    "crop_paths",
    "uploads",
])

EXPECTED_MEDIA_FILES_COLUMNS = [
    "id",
    "file_path",
    "sha256_prefix",
    "codec",
    "resolution_w",
    "resolution_h",
    "fps",
    "duration_seconds",
    "created_at",
    "gps_lat",
    "gps_lon",
    "audio_channels",
    "status",
    "metadata_json",
]


class TestSchema:
    """Tests for CatalogDB._create_schema()."""

    def test_schema_creates_all_12_tables(self, catalog_db):
        """_create_schema() creates all 12 expected tables."""
        cur = catalog_db.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        )
        tables = sorted(row[0] for row in cur.fetchall())
        assert tables == EXPECTED_TABLES

    def test_schema_media_files_columns(self, catalog_db):
        """media_files table has all expected columns."""
        cur = catalog_db.conn.execute("PRAGMA table_info(media_files)")
        columns = [row[1] for row in cur.fetchall()]
        assert columns == EXPECTED_MEDIA_FILES_COLUMNS

    def test_schema_is_idempotent(self, catalog_db):
        """Calling _create_schema() twice does not error."""
        catalog_db._create_schema()
        # Verify tables still exist
        cur = catalog_db.conn.execute(
            "SELECT count(*) FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        assert cur.fetchone()[0] == 12

    def test_schema_foreign_keys(self, catalog_db):
        """Foreign key constraints exist on child tables."""
        cur = catalog_db.conn.execute(
            "PRAGMA foreign_key_list(transcripts)"
        )
        fks = cur.fetchall()
        assert len(fks) >= 1
        # Check the FK references media_files.id
        fk = fks[0]
        assert fk[2] == "media_files"  # table
        assert fk[3] == "media_id"  # from
        assert fk[4] == "id"  # to


# -- media_files CRUD --------------------------------------------------------


class TestMediaFilesCRUD:
    """Tests for media_files table CRUD operations."""

    def test_insert_media(self, catalog_db):
        """Insert a media file row and retrieve it."""
        catalog_db.insert_media(
            id="m1",
            file_path="/videos/test.mp4",
            sha256_prefix="abc123",
            codec="h264",
            resolution_w=1920,
            resolution_h=1080,
            fps=30.0,
            duration_seconds=120.5,
        )
        cur = catalog_db.conn.execute(
            "SELECT * FROM media_files WHERE id = ?", ("m1",)
        )
        row = cur.fetchone()
        assert row is not None
        assert row["file_path"] == "/videos/test.mp4"
        assert row["codec"] == "h264"

    def test_get_media(self, catalog_db):
        """get_media returns correct row as dict."""
        catalog_db.insert_media(id="m1", file_path="/videos/test.mp4")
        result = catalog_db.get_media("m1")
        assert result is not None
        assert result["id"] == "m1"
        assert result["file_path"] == "/videos/test.mp4"
        assert result["status"] == "ingested"

    def test_get_media_not_found(self, catalog_db):
        """get_media returns None for nonexistent id."""
        assert catalog_db.get_media("nonexistent") is None

    def test_update_media_status(self, catalog_db):
        """update_media_status changes status field."""
        catalog_db.insert_media(id="m1", file_path="/videos/test.mp4")
        catalog_db.update_media_status("m1", "analyzing")
        result = catalog_db.get_media("m1")
        assert result is not None
        assert result["status"] == "analyzing"

    def test_list_by_status(self, catalog_db):
        """list_by_status returns only matching rows."""
        catalog_db.insert_media(
            id="m1", file_path="/a.mp4", status="ingested"
        )
        catalog_db.insert_media(
            id="m2", file_path="/b.mp4", status="analyzing"
        )
        catalog_db.insert_media(
            id="m3", file_path="/c.mp4", status="ingested"
        )
        results = catalog_db.list_by_status("ingested")
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"m1", "m3"}

    def test_find_by_hash(self, catalog_db):
        """find_by_hash returns the media with matching sha256_prefix."""
        catalog_db.insert_media(
            id="m1", file_path="/a.mp4", sha256_prefix="abc123"
        )
        result = catalog_db.find_by_hash("abc123")
        assert result is not None
        assert result["id"] == "m1"

    def test_find_by_hash_not_found(self, catalog_db):
        """find_by_hash returns None for nonexistent hash."""
        assert catalog_db.find_by_hash("nonexistent") is None

    def test_insert_media_duplicate_id(self, catalog_db):
        """Inserting duplicate primary key raises IntegrityError."""
        catalog_db.insert_media(id="m1", file_path="/a.mp4")
        with pytest.raises(sqlite3.IntegrityError):
            catalog_db.insert_media(id="m1", file_path="/b.mp4")
