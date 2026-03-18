"""Tests for autopilot.db CatalogDB class."""

from __future__ import annotations

import sqlite3
import struct
import threading
import time

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
                ctx.conn.execute("CREATE TABLE IF NOT EXISTS _test (val TEXT)")
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

EXPECTED_TABLES = sorted(
    [
        "media_files",
        "transcripts",
        "shot_boundaries",
        "detections",
        "faces",
        "face_clusters",
        "clip_embeddings",
        "audio_events",
        "activity_clusters",
        "narratives",
        "edit_plans",
        "crop_paths",
        "uploads",
    ]
)

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

    def test_schema_creates_all_13_tables(self, catalog_db):
        """_create_schema() creates all 13 expected tables."""
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
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        assert cur.fetchone()[0] == 13

    def test_schema_foreign_keys(self, catalog_db):
        """Foreign key constraints exist on child tables."""
        cur = catalog_db.conn.execute("PRAGMA foreign_key_list(transcripts)")
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
        cur = catalog_db.conn.execute("SELECT * FROM media_files WHERE id = ?", ("m1",))
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
        catalog_db.insert_media(id="m1", file_path="/a.mp4", status="ingested")
        catalog_db.insert_media(id="m2", file_path="/b.mp4", status="analyzing")
        catalog_db.insert_media(id="m3", file_path="/c.mp4", status="ingested")
        results = catalog_db.list_by_status("ingested")
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"m1", "m3"}

    def test_find_by_hash(self, catalog_db):
        """find_by_hash returns the media with matching sha256_prefix."""
        catalog_db.insert_media(id="m1", file_path="/a.mp4", sha256_prefix="abc123")
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
        catalog_db.upsert_transcript("m1", segments_json='[{"start": 0}]', language="en")
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
        catalog_db.upsert_boundaries("m1", boundaries_json="[]", method="transnetv2")
        result = catalog_db.get_boundaries("m1", method="transnetv2")
        assert result is not None
        assert result["media_id"] == "m1"

    def test_boundaries_composite_key(self, catalog_db):
        """Same media_id with different methods stored separately."""
        _insert_test_media(catalog_db)
        catalog_db.upsert_boundaries("m1", boundaries_json="[1]", method="transnetv2")
        catalog_db.upsert_boundaries("m1", boundaries_json="[2]", method="pyscenedetect")
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
        rows = [("m1", i, f'{{"track_id": {i}}}') for i in range(100)]
        catalog_db.batch_insert_detections(rows)
        cur = catalog_db.conn.execute("SELECT count(*) FROM detections WHERE media_id = 'm1'")
        assert cur.fetchone()[0] == 100

    def test_get_detections_for_frame(self, catalog_db):
        """Retrieve detections for a specific media_id + frame_number."""
        _insert_test_media(catalog_db)
        catalog_db.batch_insert_detections(
            [
                ("m1", 10, '{"class": "person"}'),
                ("m1", 20, '{"class": "car"}'),
            ]
        )
        result = catalog_db.get_detections_for_frame("m1", 10)
        assert result is not None
        assert result["frame_number"] == 10
        assert result["detections_json"] == '{"class": "person"}'

    def test_get_detections_for_range(self, catalog_db):
        """Retrieve detections for media_id between frame_start and frame_end."""
        _insert_test_media(catalog_db)
        catalog_db.batch_insert_detections(
            [
                ("m1", 5, "[]"),
                ("m1", 10, "[]"),
                ("m1", 15, "[]"),
                ("m1", 20, "[]"),
            ]
        )
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
        cur = catalog_db.conn.execute("SELECT count(*) FROM clip_embeddings WHERE media_id = 'm1'")
        assert cur.fetchone()[0] == 10

    def test_get_embeddings_for_media(self, catalog_db):
        """Retrieve all embeddings for a media_id."""
        _insert_test_media(catalog_db)
        emb = _make_embedding(768)
        catalog_db.batch_insert_embeddings(
            [
                ("m1", 0, emb),
                ("m1", 10, emb),
            ]
        )
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


# -- audio_events & activity_clusters CRUD ------------------------------------


class TestAudioEventsCRUD:
    """Tests for audio_events table CRUD operations."""

    def test_batch_insert_audio_events(self, catalog_db):
        """Insert multiple audio events."""
        _insert_test_media(catalog_db)
        rows = [
            ("m1", 0.5, '{"class": "speech"}'),
            ("m1", 1.5, '{"class": "music"}'),
            ("m1", 3.0, '{"class": "silence"}'),
        ]
        catalog_db.batch_insert_audio_events(rows)
        cur = catalog_db.conn.execute("SELECT count(*) FROM audio_events WHERE media_id = 'm1'")
        assert cur.fetchone()[0] == 3

    def test_get_events_for_range(self, catalog_db):
        """Retrieve events for media_id between start and end timestamps."""
        _insert_test_media(catalog_db)
        catalog_db.batch_insert_audio_events(
            [
                ("m1", 0.5, '{"class": "speech"}'),
                ("m1", 1.5, '{"class": "music"}'),
                ("m1", 3.0, '{"class": "silence"}'),
                ("m1", 5.0, '{"class": "speech"}'),
            ]
        )
        results = catalog_db.get_audio_events_for_range("m1", 1.0, 4.0)
        assert len(results) == 2
        timestamps = {r["timestamp_seconds"] for r in results}
        assert timestamps == {1.5, 3.0}


class TestActivityClustersCRUD:
    """Tests for activity_clusters table CRUD operations."""

    def test_insert_activity_cluster(self, catalog_db):
        """Insert activity cluster with all fields."""
        catalog_db.insert_activity_cluster(
            cluster_id="ac1",
            label="Beach Day",
            description="Fun at the beach",
            time_start="2024-01-01T09:00:00",
            time_end="2024-01-01T17:00:00",
            location_label="Santa Monica",
            gps_center_lat=34.0195,
            gps_center_lon=-118.4912,
            clip_ids_json='["m1", "m2"]',
        )
        results = catalog_db.get_activity_clusters()
        assert len(results) == 1
        assert results[0]["label"] == "Beach Day"

    def test_get_activity_clusters(self, catalog_db):
        """List all activity clusters."""
        catalog_db.insert_activity_cluster(cluster_id="ac1", label="A")
        catalog_db.insert_activity_cluster(cluster_id="ac2", label="B")
        results = catalog_db.get_activity_clusters()
        assert len(results) == 2

    def test_update_activity_cluster(self, catalog_db):
        """Update label and description of an activity cluster."""
        catalog_db.insert_activity_cluster(cluster_id="ac1", label="Old", description="Old desc")
        catalog_db.update_activity_cluster("ac1", label="New", description="New desc")
        results = catalog_db.get_activity_clusters()
        assert results[0]["label"] == "New"
        assert results[0]["description"] == "New desc"

    def test_activity_cluster_json_roundtrip(self, catalog_db):
        """Verify clip_ids_json round-trips."""
        json_str = '["m1", "m2", "m3"]'
        catalog_db.insert_activity_cluster(cluster_id="ac1", clip_ids_json=json_str)
        results = catalog_db.get_activity_clusters()
        assert results[0]["clip_ids_json"] == json_str


# -- narratives, edit_plans, crop_paths, uploads CRUD -------------------------


class TestNarrativesCRUD:
    """Tests for narratives table CRUD operations."""

    def test_insert_narrative(self, catalog_db):
        """Insert a narrative with all fields."""
        catalog_db.insert_narrative(
            narrative_id="n1",
            title="Beach Day Video",
            description="A fun day at the beach",
            proposed_duration_seconds=180.0,
            activity_cluster_ids_json='["ac1", "ac2"]',
            arc_notes="Start with arrival, end with sunset",
        )
        result = catalog_db.get_narrative("n1")
        assert result is not None
        assert result["title"] == "Beach Day Video"
        assert result["status"] == "proposed"

    def test_get_narrative(self, catalog_db):
        """Retrieve narrative by id."""
        catalog_db.insert_narrative(narrative_id="n1", title="Test")
        result = catalog_db.get_narrative("n1")
        assert result is not None
        assert result["narrative_id"] == "n1"
        # Nonexistent
        assert catalog_db.get_narrative("nonexistent") is None

    def test_update_narrative_status(self, catalog_db):
        """Change narrative status."""
        catalog_db.insert_narrative(narrative_id="n1", title="Test")
        catalog_db.update_narrative_status("n1", "approved")
        result = catalog_db.get_narrative("n1")
        assert result is not None
        assert result["status"] == "approved"

    def test_list_narratives(self, catalog_db):
        """List all narratives, optionally filter by status."""
        catalog_db.insert_narrative(narrative_id="n1", title="A", status="proposed")
        catalog_db.insert_narrative(narrative_id="n2", title="B", status="approved")
        catalog_db.insert_narrative(narrative_id="n3", title="C", status="proposed")
        # All
        assert len(catalog_db.list_narratives()) == 3
        # Filtered
        proposed = catalog_db.list_narratives(status="proposed")
        assert len(proposed) == 2
        ids = {r["narrative_id"] for r in proposed}
        assert ids == {"n1", "n3"}


class TestEditPlansCRUD:
    """Tests for edit_plans table CRUD operations."""

    def test_upsert_edit_plan(self, catalog_db):
        """Insert and update an edit plan."""
        catalog_db.insert_narrative(narrative_id="n1", title="Test")
        catalog_db.upsert_edit_plan(
            narrative_id="n1",
            edl_json='{"timeline": []}',
            otio_path="/exports/n1.otio",
        )
        result = catalog_db.get_edit_plan("n1")
        assert result is not None
        assert result["edl_json"] == '{"timeline": []}'

        # Update
        catalog_db.upsert_edit_plan(
            narrative_id="n1",
            edl_json='{"timeline": [{"clip": "m1"}]}',
            validation_json='{"ok": true}',
        )
        result = catalog_db.get_edit_plan("n1")
        assert result is not None
        assert '{"clip": "m1"}' in result["edl_json"]

    def test_get_edit_plan(self, catalog_db):
        """Retrieve edit plan by narrative_id."""
        catalog_db.insert_narrative(narrative_id="n1", title="Test")
        catalog_db.upsert_edit_plan(narrative_id="n1", edl_json="{}")
        result = catalog_db.get_edit_plan("n1")
        assert result is not None
        # Nonexistent
        assert catalog_db.get_edit_plan("nonexistent") is None


class TestCropPathsCRUD:
    """Tests for crop_paths table CRUD operations."""

    def test_upsert_crop_path(self, catalog_db):
        """Insert crop path with BLOB data."""
        _insert_test_media(catalog_db)
        path_data = b"\x00\x01\x02\x03" * 100
        catalog_db.upsert_crop_path(
            media_id="m1",
            target_aspect="16:9",
            subject_track_id=1,
            smoothing_tau=0.5,
            path_data=path_data,
        )
        result = catalog_db.get_crop_path("m1", "16:9", 1)
        assert result is not None
        assert result["path_data"] == path_data

    def test_get_crop_path(self, catalog_db):
        """Retrieve crop path by composite key."""
        _insert_test_media(catalog_db)
        catalog_db.upsert_crop_path(
            media_id="m1",
            target_aspect="9:16",
            subject_track_id=0,
        )
        result = catalog_db.get_crop_path("m1", "9:16", 0)
        assert result is not None
        assert result["target_aspect"] == "9:16"
        # Nonexistent combo
        assert catalog_db.get_crop_path("m1", "16:9", 0) is None


class TestUploadsCRUD:
    """Tests for uploads table CRUD operations."""

    def test_insert_upload(self, catalog_db):
        """Insert an upload record."""
        catalog_db.insert_narrative(narrative_id="n1", title="Test")
        catalog_db.insert_upload(
            narrative_id="n1",
            youtube_video_id="abc123",
            youtube_url="https://youtube.com/watch?v=abc123",
            uploaded_at="2024-01-15T12:00:00",
            privacy_status="public",
        )
        result = catalog_db.get_upload("n1")
        assert result is not None
        assert result["youtube_video_id"] == "abc123"
        assert result["privacy_status"] == "public"

    def test_get_upload(self, catalog_db):
        """Retrieve upload by narrative_id."""
        catalog_db.insert_narrative(narrative_id="n1", title="Test")
        catalog_db.insert_upload(narrative_id="n1")
        result = catalog_db.get_upload("n1")
        assert result is not None
        assert result["privacy_status"] == "unlisted"
        # Nonexistent
        assert catalog_db.get_upload("nonexistent") is None


# -- Integration / edge-case tests -------------------------------------------


class TestIntegration:
    """Integration and edge-case tests for CatalogDB."""

    def test_transaction_commit(self):
        """__exit__ commits data written via raw SQL inside a 'with' block.

        Uses raw conn.execute (not CRUD methods) so the ONLY commit path
        is __exit__'s conn.commit().  This exercises the context manager's
        commit logic directly.
        """
        from autopilot.db import CatalogDB

        db = CatalogDB(":memory:")
        try:
            with db:
                db.conn.execute(
                    "INSERT INTO media_files (id, file_path) VALUES ('m1', '/a.mp4')"
                )
            # After __exit__ commits, data should be visible
            cur = db.conn.execute("SELECT * FROM media_files WHERE id = 'm1'")
            row = cur.fetchone()
            assert row is not None
            assert row["id"] == "m1"
        finally:
            db.close()

    def test_transaction_rollback(self):
        """__exit__ rolls back public CRUD writes on exception.

        Uses db.insert_media() (a public CRUD method) inside a 'with' block
        followed by a deliberate exception.  Since CRUD methods no longer
        auto-commit, __exit__'s rollback should discard the insert.
        """
        from autopilot.db import CatalogDB

        db = CatalogDB(":memory:")
        try:
            try:
                with db:
                    db.insert_media(id="m1", file_path="/a.mp4")
                    raise ValueError("intentional error")
            except ValueError:
                pass
            # Data should NOT be persisted due to rollback
            result = db.get_media("m1")
            assert result is None, "rollback should have discarded insert_media"
        finally:
            db.close()

    def test_batch_insert_performance(self, catalog_db):
        """Insert 1000 detection rows in under 1 second."""
        catalog_db.insert_media(id="m1", file_path="/a.mp4")
        rows = [("m1", i, f'{{"frame": {i}}}') for i in range(1000)]
        start = time.monotonic()
        catalog_db.batch_insert_detections(rows)
        elapsed = time.monotonic() - start
        assert elapsed < 1.0, f"Batch insert took {elapsed:.2f}s, expected <1s"
        cur = catalog_db.conn.execute("SELECT count(*) FROM detections WHERE media_id = 'm1'")
        assert cur.fetchone()[0] == 1000

    def test_foreign_key_enforcement(self, catalog_db):
        """Insert transcript referencing nonexistent media_id raises IntegrityError."""
        with pytest.raises(sqlite3.IntegrityError):
            catalog_db.upsert_transcript("nonexistent_media", segments_json="[]", language="en")

    def test_json1_extension_available(self, catalog_db):
        """Verify json_extract works via a SQL query on a JSON column."""
        catalog_db.insert_media(
            id="m1",
            file_path="/a.mp4",
            metadata_json='{"camera": "GoPro", "model": "Hero12"}',
        )
        cur = catalog_db.conn.execute(
            "SELECT json_extract(metadata_json, '$.camera') AS camera "
            "FROM media_files WHERE id = 'm1'"
        )
        row = cur.fetchone()
        assert row is not None
        assert row["camera"] == "GoPro"


# -- Transaction semantics tests (review fix) --------------------------------


class TestTransactionSemantics:
    """Tests for correct transaction semantics using public CRUD API methods.

    These tests verify that CRUD methods do NOT auto-commit, so that the
    context manager's commit/rollback is the sole commit point.
    """

    def test_crud_does_not_autocommit(self):
        """CRUD methods should not commit internally; rollback undoes them."""
        from autopilot.db import CatalogDB

        db = CatalogDB(":memory:")
        try:
            db.insert_media(id="m1", file_path="/a.mp4")
            db.conn.rollback()
            # After rollback, the insert should be undone
            result = db.get_media("m1")
            assert result is None, (
                "insert_media should not auto-commit; rollback should undo the insert"
            )
        finally:
            db.close()

    def test_context_manager_commits_crud(self):
        """Context manager commits CRUD operations on successful exit."""
        from autopilot.db import CatalogDB

        db = CatalogDB(":memory:")
        try:
            with db:
                db.insert_media(id="m1", file_path="/a.mp4")
            # After __exit__ commits, data should persist
            result = db.get_media("m1")
            assert result is not None
            assert result["id"] == "m1"
        finally:
            db.close()

    def test_context_manager_rollback_crud(self):
        """Context manager rolls back CRUD operations on exception."""
        from autopilot.db import CatalogDB

        db = CatalogDB(":memory:")
        try:
            try:
                with db:
                    db.insert_media(id="m1", file_path="/a.mp4")
                    raise ValueError("intentional error")
            except ValueError:
                pass
            # insert_media's data should be rolled back
            result = db.get_media("m1")
            assert result is None, (
                "insert_media inside 'with' block should be rolled back on exception"
            )
        finally:
            db.close()

    def test_multi_op_atomicity(self):
        """Multiple CRUD ops in a 'with' block are atomic — all or nothing."""
        from autopilot.db import CatalogDB

        db = CatalogDB(":memory:")
        try:
            try:
                with db:
                    db.insert_media(id="m1", file_path="/a.mp4")
                    db.upsert_transcript("m1", segments_json="[]", language="en")
                    raise ValueError("intentional error after both inserts")
            except ValueError:
                pass
            # NEITHER media nor transcript should persist
            assert db.get_media("m1") is None, "media row should be rolled back"
            assert db.get_transcript("m1") is None, "transcript row should be rolled back"
        finally:
            db.close()

    def test_explicit_commit_without_context_manager(self):
        """Calling db.conn.commit() explicitly persists CRUD data."""
        from autopilot.db import CatalogDB

        db = CatalogDB(":memory:")
        try:
            db.insert_media(id="m1", file_path="/a.mp4")
            db.conn.commit()
            # After explicit commit, data should persist even after rollback
            db.conn.rollback()
            result = db.get_media("m1")
            assert result is not None
            assert result["id"] == "m1"
        finally:
            db.close()


# -- Init safety tests (review fix) ------------------------------------------


class TestInitSafety:
    """Tests for CatalogDB.__init__ resource safety."""

    def test_init_failure_closes_connection(self, monkeypatch):
        """If _create_schema() fails, the connection should be closed."""
        from unittest.mock import MagicMock

        from autopilot.db import CatalogDB

        # Capture the connection object so we can check it's closed afterward
        captured: dict[str, sqlite3.Connection] = {}
        original_connect = sqlite3.connect

        def spy_connect(*args, **kwargs):
            conn = original_connect(*args, **kwargs)
            captured["conn"] = conn
            return conn

        monkeypatch.setattr(sqlite3, "connect", spy_connect)
        monkeypatch.setattr(
            CatalogDB, "_create_schema", MagicMock(side_effect=RuntimeError("schema fail"))
        )

        with pytest.raises(RuntimeError, match="schema fail"):
            CatalogDB(":memory:")

        # Verify the connection was closed — executing on a closed connection
        # raises ProgrammingError
        conn = captured["conn"]
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")
