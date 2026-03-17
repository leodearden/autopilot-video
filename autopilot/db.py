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

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()
