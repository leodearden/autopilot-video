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


# -- transcripts & shot_boundaries CRUD --------------------------------------


def _insert_test_media(catalog_db, media_id: str = "m1") -> None:
    """Helper to insert a media file for FK-dependent tests."""
    catalog_db.insert_media(id=media_id, file_path=f"/{media_id}.mp4")


class TestTranscriptsCRUD:
    """Tests for transcripts table CRUD operations."""

    def test_upsert_transcript(self, catalog_db):
        """Insert then update a transcript, verify latest value."""
        _insert_test_media(catalog_db)
        catalog_db.upsert_transcript(
            "m1", segments_json='[{"start": 0}]', language="en"
        )
        result = catalog_db.get_transcript("m1")
        assert result is not None
        assert result["language"] == "en"

        # Update
        catalog_db.upsert_transcript(
            "m1", segments_json='[{"start": 0}, {"start": 5}]', language="fr"
        )
        result = catalog_db.get_transcript("m1")
        assert result is not None
        assert result["language"] == "fr"
        assert result["segments_json"] == '[{"start": 0}, {"start": 5}]'

    def test_get_transcript(self, catalog_db):
        """Retrieve transcript by media_id."""
        _insert_test_media(catalog_db)
        catalog_db.upsert_transcript("m1", segments_json="[]", language="en")
        result = catalog_db.get_transcript("m1")
        assert result is not None
        assert result["media_id"] == "m1"

    def test_get_transcript_not_found(self, catalog_db):
        """get_transcript returns None for nonexistent media_id."""
        assert catalog_db.get_transcript("nonexistent") is None

    def test_transcript_json_roundtrip(self, catalog_db):
        """Store JSON string, retrieve identical string."""
        _insert_test_media(catalog_db)
        json_str = '{"segments": [{"start": 0.0, "end": 1.5, "text": "hello"}]}'
        catalog_db.upsert_transcript("m1", segments_json=json_str, language="en")
        result = catalog_db.get_transcript("m1")
        assert result is not None
        assert result["segments_json"] == json_str


class TestShotBoundariesCRUD:
    """Tests for shot_boundaries table CRUD operations."""

    def test_upsert_boundaries(self, catalog_db):
        """Insert shot boundaries with method."""
        _insert_test_media(catalog_db)
        catalog_db.upsert_boundaries(
            "m1",
            boundaries_json="[[0, 100, 'cut']]",
            method="transnetv2",
        )
        result = catalog_db.get_boundaries("m1", method="transnetv2")
        assert result is not None
        assert result["method"] == "transnetv2"

    def test_get_boundaries(self, catalog_db):
        """Retrieve boundaries by media_id."""
        _insert_test_media(catalog_db)
        catalog_db.upsert_boundaries(
            "m1", boundaries_json="[]", method="transnetv2"
        )
        result = catalog_db.get_boundaries("m1", method="transnetv2")
        assert result is not None
        assert result["media_id"] == "m1"

    def test_boundaries_composite_key(self, catalog_db):
        """Same media_id with different methods stored separately."""
        _insert_test_media(catalog_db)
        catalog_db.upsert_boundaries(
            "m1", boundaries_json="[1]", method="transnetv2"
        )
        catalog_db.upsert_boundaries(
            "m1", boundaries_json="[2]", method="pyscenedetect"
        )
        results = catalog_db.get_boundaries("m1")
        assert len(results) == 2
        methods = {r["method"] for r in results}
        assert methods == {"transnetv2", "pyscenedetect"}


# -- detections CRUD ---------------------------------------------------------


class TestDetectionsCRUD:
    """Tests for detections table CRUD operations."""

    def test_batch_insert_detections(self, catalog_db):
        """Insert 100 detection rows in one call and verify count."""
        _insert_test_media(catalog_db)
        rows = [
            ("m1", i, f'{{"track_id": {i}}}')
            for i in range(100)
        ]
        catalog_db.batch_insert_detections(rows)
        cur = catalog_db.conn.execute(
            "SELECT count(*) FROM detections WHERE media_id = 'm1'"
        )
        assert cur.fetchone()[0] == 100

    def test_get_detections_for_frame(self, catalog_db):
        """Retrieve detections for a specific media_id + frame_number."""
        _insert_test_media(catalog_db)
        catalog_db.batch_insert_detections([
            ("m1", 10, '{"class": "person"}'),
            ("m1", 20, '{"class": "car"}'),
        ])
        result = catalog_db.get_detections_for_frame("m1", 10)
        assert result is not None
        assert result["frame_number"] == 10
        assert result["detections_json"] == '{"class": "person"}'

    def test_get_detections_for_range(self, catalog_db):
        """Retrieve detections for media_id between frame_start and frame_end."""
        _insert_test_media(catalog_db)
        catalog_db.batch_insert_detections([
            ("m1", 5, "[]"),
            ("m1", 10, "[]"),
            ("m1", 15, "[]"),
            ("m1", 20, "[]"),
        ])
        results = catalog_db.get_detections_for_range("m1", 8, 18)
        assert len(results) == 2
        frames = {r["frame_number"] for r in results}
        assert frames == {10, 15}

    def test_batch_insert_detections_empty(self, catalog_db):
        """Batch insert with empty list does not error."""
        catalog_db.batch_insert_detections([])

    def test_detections_json_roundtrip(self, catalog_db):
        """Verify JSON text stored and retrieved identically."""
        _insert_test_media(catalog_db)
        json_str = '[{"track_id": 1, "class": "person", "bbox": [10, 20, 100, 200]}]'
        catalog_db.batch_insert_detections([("m1", 0, json_str)])
        result = catalog_db.get_detections_for_frame("m1", 0)
        assert result is not None
        assert result["detections_json"] == json_str


# -- face_clusters & clip_embeddings CRUD ------------------------------------

import struct


def _make_embedding(dim: int = 512) -> bytes:
    """Create a dummy embedding blob of float32 values."""
    return struct.pack(f"{dim}f", *[float(i) for i in range(dim)])


class TestFaceClustersCRUD:
    """Tests for face_clusters table CRUD operations."""

    def test_insert_face_cluster(self, catalog_db):
        """Insert a face cluster with label and BLOB embedding."""
        emb = _make_embedding(512)
        catalog_db.insert_face_cluster(
            cluster_id=1,
            label="Alice",
            representative_embedding=emb,
            sample_image_paths='["/crops/face1.jpg"]',
        )
        result = catalog_db.get_face_cluster_by_id(1)
        assert result is not None
        assert result["label"] == "Alice"
        assert result["representative_embedding"] == emb

    def test_update_face_label(self, catalog_db):
        """Update label for existing cluster."""
        catalog_db.insert_face_cluster(cluster_id=1, label="Unknown")
        catalog_db.update_face_label(1, "Bob")
        result = catalog_db.get_face_cluster_by_id(1)
        assert result is not None
        assert result["label"] == "Bob"

    def test_get_face_clusters(self, catalog_db):
        """List all face clusters."""
        catalog_db.insert_face_cluster(cluster_id=1, label="Alice")
        catalog_db.insert_face_cluster(cluster_id=2, label="Bob")
        results = catalog_db.get_face_clusters()
        assert len(results) == 2

    def test_get_face_cluster_by_id(self, catalog_db):
        """Retrieve single cluster by id."""
        catalog_db.insert_face_cluster(cluster_id=1, label="Alice")
        result = catalog_db.get_face_cluster_by_id(1)
        assert result is not None
        assert result["cluster_id"] == 1
        # Nonexistent
        assert catalog_db.get_face_cluster_by_id(999) is None


class TestClipEmbeddingsCRUD:
    """Tests for clip_embeddings table CRUD operations."""

    def test_batch_insert_embeddings(self, catalog_db):
        """Insert multiple embeddings with BLOB data."""
        _insert_test_media(catalog_db)
        emb = _make_embedding(768)
        rows = [("m1", i, emb) for i in range(10)]
        catalog_db.batch_insert_embeddings(rows)
        cur = catalog_db.conn.execute(
            "SELECT count(*) FROM clip_embeddings WHERE media_id = 'm1'"
        )
        assert cur.fetchone()[0] == 10

    def test_get_embeddings_for_media(self, catalog_db):
        """Retrieve all embeddings for a media_id."""
        _insert_test_media(catalog_db)
        emb = _make_embedding(768)
        catalog_db.batch_insert_embeddings([
            ("m1", 0, emb),
            ("m1", 10, emb),
        ])
        results = catalog_db.get_embeddings_for_media("m1")
        assert len(results) == 2
        frames = {r["frame_number"] for r in results}
        assert frames == {0, 10}

    def test_embedding_blob_roundtrip(self, catalog_db):
        """Verify binary data survives round-trip."""
        _insert_test_media(catalog_db)
        emb = _make_embedding(768)
        catalog_db.batch_insert_embeddings([("m1", 0, emb)])
        results = catalog_db.get_embeddings_for_media("m1")
        assert len(results) == 1
        assert results[0]["embedding"] == emb
        # Verify we can unpack it back
        values = struct.unpack(f"{768}f", results[0]["embedding"])
        assert values[0] == 0.0
        assert values[767] == 767.0
