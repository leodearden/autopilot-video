"""SQLite catalog database interface for autopilot-video."""

from __future__ import annotations

import sqlite3
from typing import Self


class CatalogDB:
    """Wraps a SQLite database with WAL mode, thread safety, and schema management.

    Usage::

        db = CatalogDB("catalog.db")
        with db:
            db.insert_media(...)
        db.close()
    """

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Enable WAL mode for concurrent read access
        self.conn.execute("PRAGMA journal_mode=WAL")
        # Enable foreign key enforcement
        self.conn.execute("PRAGMA foreign_keys=ON")

        self._create_schema()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()

    def _create_schema(self) -> None:
        """Create all 12 catalog tables if they don't already exist."""
        self.conn.executescript(
            """\
            CREATE TABLE IF NOT EXISTS media_files (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                sha256_prefix TEXT,
                codec TEXT,
                resolution_w INTEGER,
                resolution_h INTEGER,
                fps REAL,
                duration_seconds REAL,
                created_at TEXT,
                gps_lat REAL,
                gps_lon REAL,
                audio_channels INTEGER,
                status TEXT DEFAULT 'ingested',
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS transcripts (
                media_id TEXT REFERENCES media_files(id),
                segments_json TEXT,
                language TEXT,
                PRIMARY KEY (media_id)
            );

            CREATE TABLE IF NOT EXISTS shot_boundaries (
                media_id TEXT REFERENCES media_files(id),
                boundaries_json TEXT,
                method TEXT,
                PRIMARY KEY (media_id, method)
            );

            CREATE TABLE IF NOT EXISTS detections (
                media_id TEXT REFERENCES media_files(id),
                frame_number INTEGER,
                detections_json TEXT,
                PRIMARY KEY (media_id, frame_number)
            );

            CREATE TABLE IF NOT EXISTS face_clusters (
                cluster_id INTEGER PRIMARY KEY,
                label TEXT,
                representative_embedding BLOB,
                sample_image_paths TEXT
            );

            CREATE TABLE IF NOT EXISTS clip_embeddings (
                media_id TEXT REFERENCES media_files(id),
                frame_number INTEGER,
                embedding BLOB,
                PRIMARY KEY (media_id, frame_number)
            );

            CREATE TABLE IF NOT EXISTS audio_events (
                media_id TEXT REFERENCES media_files(id),
                timestamp_seconds REAL,
                events_json TEXT,
                PRIMARY KEY (media_id, timestamp_seconds)
            );

            CREATE TABLE IF NOT EXISTS activity_clusters (
                cluster_id TEXT PRIMARY KEY,
                label TEXT,
                description TEXT,
                time_start TEXT,
                time_end TEXT,
                location_label TEXT,
                gps_center_lat REAL,
                gps_center_lon REAL,
                clip_ids_json TEXT
            );

            CREATE TABLE IF NOT EXISTS narratives (
                narrative_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                proposed_duration_seconds REAL,
                activity_cluster_ids_json TEXT,
                arc_notes TEXT,
                status TEXT DEFAULT 'proposed'
            );

            CREATE TABLE IF NOT EXISTS edit_plans (
                narrative_id TEXT REFERENCES narratives(narrative_id),
                edl_json TEXT,
                otio_path TEXT,
                validation_json TEXT,
                PRIMARY KEY (narrative_id)
            );

            CREATE TABLE IF NOT EXISTS crop_paths (
                media_id TEXT REFERENCES media_files(id),
                target_aspect TEXT,
                subject_track_id INTEGER,
                smoothing_tau REAL,
                path_data BLOB,
                PRIMARY KEY (media_id, target_aspect, subject_track_id)
            );

            CREATE TABLE IF NOT EXISTS uploads (
                narrative_id TEXT REFERENCES narratives(narrative_id),
                youtube_video_id TEXT,
                youtube_url TEXT,
                uploaded_at TEXT,
                privacy_status TEXT DEFAULT 'unlisted',
                PRIMARY KEY (narrative_id)
            );
            """
        )

    # -- media_files CRUD -------------------------------------------------------

    def insert_media(
        self,
        id: str,
        file_path: str,
        *,
        sha256_prefix: str | None = None,
        codec: str | None = None,
        resolution_w: int | None = None,
        resolution_h: int | None = None,
        fps: float | None = None,
        duration_seconds: float | None = None,
        created_at: str | None = None,
        gps_lat: float | None = None,
        gps_lon: float | None = None,
        audio_channels: int | None = None,
        status: str = "ingested",
        metadata_json: str | None = None,
    ) -> None:
        """Insert a new media file row."""
        self.conn.execute(
            "INSERT INTO media_files "
            "(id, file_path, sha256_prefix, codec, resolution_w, "
            "resolution_h, fps, duration_seconds, created_at, "
            "gps_lat, gps_lon, audio_channels, status, metadata_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                id,
                file_path,
                sha256_prefix,
                codec,
                resolution_w,
                resolution_h,
                fps,
                duration_seconds,
                created_at,
                gps_lat,
                gps_lon,
                audio_channels,
                status,
                metadata_json,
            ),
        )
        self.conn.commit()

    def get_media(self, media_id: str) -> dict[str, object] | None:
        """Get a media file by id, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM media_files WHERE id = ?", (media_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def update_media_status(self, media_id: str, status: str) -> None:
        """Update the status of a media file."""
        self.conn.execute(
            "UPDATE media_files SET status = ? WHERE id = ?",
            (status, media_id),
        )
        self.conn.commit()

    def list_by_status(self, status: str) -> list[dict[str, object]]:
        """List all media files with a given status."""
        cur = self.conn.execute(
            "SELECT * FROM media_files WHERE status = ?", (status,)
        )
        return [dict(row) for row in cur.fetchall()]

    def find_by_hash(self, sha256_prefix: str) -> dict[str, object] | None:
        """Find a media file by its SHA-256 prefix, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM media_files WHERE sha256_prefix = ?",
            (sha256_prefix,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # -- transcripts CRUD -------------------------------------------------------

    def upsert_transcript(
        self, media_id: str, segments_json: str, language: str
    ) -> None:
        """Insert or replace a transcript for a media file."""
        self.conn.execute(
            "INSERT OR REPLACE INTO transcripts "
            "(media_id, segments_json, language) VALUES (?, ?, ?)",
            (media_id, segments_json, language),
        )
        self.conn.commit()

    def get_transcript(self, media_id: str) -> dict[str, object] | None:
        """Get a transcript by media_id, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM transcripts WHERE media_id = ?", (media_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # -- shot_boundaries CRUD ---------------------------------------------------

    def upsert_boundaries(
        self, media_id: str, boundaries_json: str, method: str
    ) -> None:
        """Insert or replace shot boundaries for a media file and method."""
        self.conn.execute(
            "INSERT OR REPLACE INTO shot_boundaries "
            "(media_id, boundaries_json, method) VALUES (?, ?, ?)",
            (media_id, boundaries_json, method),
        )
        self.conn.commit()

    def get_boundaries(
        self, media_id: str, method: str | None = None
    ) -> dict[str, object] | list[dict[str, object]]:
        """Get shot boundaries for a media file.

        If method is specified, returns a single dict or None.
        If method is None, returns a list of all boundaries for that media.
        """
        if method is not None:
            cur = self.conn.execute(
                "SELECT * FROM shot_boundaries "
                "WHERE media_id = ? AND method = ?",
                (media_id, method),
            )
            row = cur.fetchone()
            return dict(row) if row else None  # type: ignore[return-value]
        cur = self.conn.execute(
            "SELECT * FROM shot_boundaries WHERE media_id = ?",
            (media_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    # -- detections CRUD --------------------------------------------------------

    def batch_insert_detections(
        self, rows: list[tuple[str, int, str]]
    ) -> None:
        """Batch insert detection rows: (media_id, frame_number, detections_json)."""
        if not rows:
            return
        self.conn.executemany(
            "INSERT INTO detections "
            "(media_id, frame_number, detections_json) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def get_detections_for_frame(
        self, media_id: str, frame_number: int
    ) -> dict[str, object] | None:
        """Get detections for a specific frame, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM detections "
            "WHERE media_id = ? AND frame_number = ?",
            (media_id, frame_number),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_detections_for_range(
        self, media_id: str, frame_start: int, frame_end: int
    ) -> list[dict[str, object]]:
        """Get all detections for a media file within a frame range."""
        cur = self.conn.execute(
            "SELECT * FROM detections "
            "WHERE media_id = ? AND frame_number >= ? AND frame_number <= ?",
            (media_id, frame_start, frame_end),
        )
        return [dict(row) for row in cur.fetchall()]

    # -- face_clusters CRUD -----------------------------------------------------

    def insert_face_cluster(
        self,
        cluster_id: int,
        label: str | None = None,
        representative_embedding: bytes | None = None,
        sample_image_paths: str | None = None,
    ) -> None:
        """Insert a new face cluster."""
        self.conn.execute(
            "INSERT INTO face_clusters "
            "(cluster_id, label, representative_embedding, sample_image_paths) "
            "VALUES (?, ?, ?, ?)",
            (cluster_id, label, representative_embedding, sample_image_paths),
        )
        self.conn.commit()

    def update_face_label(self, cluster_id: int, label: str) -> None:
        """Update the label of a face cluster."""
        self.conn.execute(
            "UPDATE face_clusters SET label = ? WHERE cluster_id = ?",
            (label, cluster_id),
        )
        self.conn.commit()

    def get_face_clusters(self) -> list[dict[str, object]]:
        """List all face clusters."""
        cur = self.conn.execute("SELECT * FROM face_clusters")
        return [dict(row) for row in cur.fetchall()]

    def get_face_cluster_by_id(
        self, cluster_id: int
    ) -> dict[str, object] | None:
        """Get a face cluster by id, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM face_clusters WHERE cluster_id = ?",
            (cluster_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # -- clip_embeddings CRUD ---------------------------------------------------

    def batch_insert_embeddings(
        self, rows: list[tuple[str, int, bytes]]
    ) -> None:
        """Batch insert clip embeddings: (media_id, frame_number, embedding)."""
        if not rows:
            return
        self.conn.executemany(
            "INSERT INTO clip_embeddings "
            "(media_id, frame_number, embedding) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def get_embeddings_for_media(
        self, media_id: str
    ) -> list[dict[str, object]]:
        """Get all clip embeddings for a media file."""
        cur = self.conn.execute(
            "SELECT * FROM clip_embeddings WHERE media_id = ?",
            (media_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    # -- audio_events CRUD ------------------------------------------------------

    def batch_insert_audio_events(
        self, rows: list[tuple[str, float, str]]
    ) -> None:
        """Batch insert audio events: (media_id, timestamp_seconds, events_json)."""
        if not rows:
            return
        self.conn.executemany(
            "INSERT INTO audio_events "
            "(media_id, timestamp_seconds, events_json) VALUES (?, ?, ?)",
            rows,
        )
        self.conn.commit()

    def get_audio_events_for_range(
        self, media_id: str, start_seconds: float, end_seconds: float
    ) -> list[dict[str, object]]:
        """Get audio events for a media file within a time range."""
        cur = self.conn.execute(
            "SELECT * FROM audio_events "
            "WHERE media_id = ? "
            "AND timestamp_seconds >= ? AND timestamp_seconds <= ?",
            (media_id, start_seconds, end_seconds),
        )
        return [dict(row) for row in cur.fetchall()]

    # -- activity_clusters CRUD -------------------------------------------------

    def insert_activity_cluster(
        self,
        cluster_id: str,
        *,
        label: str | None = None,
        description: str | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        location_label: str | None = None,
        gps_center_lat: float | None = None,
        gps_center_lon: float | None = None,
        clip_ids_json: str | None = None,
    ) -> None:
        """Insert a new activity cluster."""
        self.conn.execute(
            "INSERT INTO activity_clusters "
            "(cluster_id, label, description, time_start, time_end, "
            "location_label, gps_center_lat, gps_center_lon, clip_ids_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cluster_id,
                label,
                description,
                time_start,
                time_end,
                location_label,
                gps_center_lat,
                gps_center_lon,
                clip_ids_json,
            ),
        )
        self.conn.commit()

    def get_activity_clusters(self) -> list[dict[str, object]]:
        """List all activity clusters."""
        cur = self.conn.execute("SELECT * FROM activity_clusters")
        return [dict(row) for row in cur.fetchall()]

    def update_activity_cluster(
        self, cluster_id: str, **kwargs: object
    ) -> None:
        """Update fields of an activity cluster by keyword arguments."""
        if not kwargs:
            return
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(cluster_id)
        self.conn.execute(
            f"UPDATE activity_clusters SET {set_clause} "  # noqa: S608
            "WHERE cluster_id = ?",
            values,
        )
        self.conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()
