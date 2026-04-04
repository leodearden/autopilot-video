"""SQLite catalog database interface for autopilot-video."""

from __future__ import annotations

import sqlite3
from typing import Any, Self, cast


class CatalogDB:
    """Wraps a SQLite database with WAL mode, thread safety, and schema management.

    CRUD write methods do NOT auto-commit.  Callers must either use the context
    manager (``with db:``) for automatic commit/rollback, or call
    ``db.conn.commit()`` explicitly after writes.

    Usage::

        db = CatalogDB("catalog.db")
        with db:
            db.insert_media(...)
            db.upsert_transcript(...)
        # both writes committed atomically on __exit__
        db.close()
    """

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        try:
            # Enable WAL mode for concurrent read access
            self.conn.execute("PRAGMA journal_mode=WAL")
            # Enable foreign key enforcement
            self.conn.execute("PRAGMA foreign_keys=ON")

            self._create_schema()
        except Exception:
            self.conn.close()
            raise

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
        """Create all 19 catalog tables if they don't already exist."""
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

            CREATE TABLE IF NOT EXISTS faces (
                media_id TEXT REFERENCES media_files(id),
                frame_number INTEGER,
                face_index INTEGER,
                bbox_json TEXT,
                embedding BLOB,
                cluster_id INTEGER,
                PRIMARY KEY (media_id, frame_number, face_index)
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
                clip_ids_json TEXT,
                excluded INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS narratives (
                narrative_id TEXT PRIMARY KEY,
                title TEXT,
                description TEXT,
                proposed_duration_seconds REAL,
                activity_cluster_ids_json TEXT,
                arc_notes TEXT,
                emotional_journey TEXT,
                status TEXT DEFAULT 'proposed'
            );

            CREATE TABLE IF NOT EXISTS edit_plans (
                narrative_id TEXT REFERENCES narratives(narrative_id),
                edl_json TEXT,
                otio_path TEXT,
                validation_json TEXT,
                render_path TEXT,
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

            CREATE TABLE IF NOT EXISTS captions (
                media_id TEXT REFERENCES media_files(id),
                start_time REAL,
                end_time REAL,
                caption TEXT,
                model_name TEXT,
                PRIMARY KEY (media_id, start_time, end_time)
            );

            CREATE TABLE IF NOT EXISTS narrative_scripts (
                narrative_id TEXT REFERENCES narratives(narrative_id),
                script_json TEXT,
                created_at TEXT,
                PRIMARY KEY (narrative_id)
            );

            CREATE TABLE IF NOT EXISTS uploads (
                narrative_id TEXT REFERENCES narratives(narrative_id),
                youtube_video_id TEXT,
                youtube_url TEXT,
                uploaded_at TEXT,
                privacy_status TEXT DEFAULT 'unlisted',
                PRIMARY KEY (narrative_id)
            );

            -- Pipeline control tables
            CREATE TABLE IF NOT EXISTS pipeline_gates (
                stage TEXT PRIMARY KEY,
                mode TEXT DEFAULT 'auto',
                status TEXT DEFAULT 'idle',
                decided_at TEXT,
                decided_by TEXT DEFAULT 'system',
                notes TEXT,
                timeout_hours REAL
            );

            CREATE TABLE IF NOT EXISTS pipeline_jobs (
                job_id TEXT PRIMARY KEY,
                stage TEXT NOT NULL,
                job_type TEXT NOT NULL,
                target_id TEXT,
                target_label TEXT,
                status TEXT DEFAULT 'pending',
                started_at TEXT,
                finished_at TEXT,
                duration_seconds REAL,
                progress_pct REAL,
                error_message TEXT,
                worker TEXT,
                run_id TEXT
            );

            CREATE TABLE IF NOT EXISTS pipeline_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                stage TEXT,
                job_id TEXT,
                payload_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                config_snapshot TEXT,
                current_stage TEXT,
                status TEXT DEFAULT 'running',
                wall_clock_seconds REAL,
                budget_remaining_seconds REAL
            );

            -- Performance indexes
            CREATE INDEX IF NOT EXISTS idx_media_files_sha256
                ON media_files(sha256_prefix);
            CREATE INDEX IF NOT EXISTS idx_media_files_status
                ON media_files(status);
            CREATE INDEX IF NOT EXISTS idx_detections_media_frame
                ON detections(media_id, frame_number);
            CREATE INDEX IF NOT EXISTS idx_faces_media_frame
                ON faces(media_id, frame_number);
            CREATE INDEX IF NOT EXISTS idx_audio_events_media_time
                ON audio_events(media_id, timestamp_seconds);
            CREATE INDEX IF NOT EXISTS idx_narratives_status
                ON narratives(status);
            CREATE INDEX IF NOT EXISTS idx_captions_media
                ON captions(media_id);
            CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_stage_status
                ON pipeline_jobs(stage, status);
            CREATE INDEX IF NOT EXISTS idx_pipeline_jobs_run_id
                ON pipeline_jobs(run_id);
            CREATE INDEX IF NOT EXISTS idx_pipeline_events_event_type
                ON pipeline_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_pipeline_events_created_at
                ON pipeline_events(created_at);
            """
        )
        # -- Schema migrations (idempotent) ------------------------------------
        # Databases created before the 'excluded' column was added need it now.
        try:
            self.conn.execute(
                "ALTER TABLE activity_clusters "
                "ADD COLUMN excluded INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError as e:
            if "duplicate column name" not in str(e).lower():
                raise  # only suppress "column already exists"


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

    def get_media(self, media_id: str) -> dict[str, object] | None:
        """Get a media file by id, or None if not found."""
        cur = self.conn.execute("SELECT * FROM media_files WHERE id = ?", (media_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def count_embeddings_for_media(self, media_id: str) -> int:
        """Count clip embeddings for a media file without loading BLOBs."""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM clip_embeddings WHERE media_id = ?",
            (media_id,),
        )
        return cur.fetchone()[0]

    def get_media_detail(self, media_id: str) -> dict[str, Any] | None:
        """Get aggregated detail for a media file including all analysis data.

        Returns None if media_id not found. Otherwise returns a dict with keys:
        media, transcript, detections, faces, audio_events,
        embedding_count, face_clusters.

        Binary BLOB fields (embedding, representative_embedding) are stripped
        from faces and face_clusters to keep the result JSON-serializable.
        """
        media = self.get_media(media_id)
        if media is None:
            return None

        transcript = self.get_transcript(media_id)
        detections = self.get_detections_for_media(media_id)
        faces = self.get_faces_for_media(media_id)
        audio_events = self.get_audio_events_for_media(media_id)
        embedding_count = self.count_embeddings_for_media(media_id)

        # Strip binary embedding BLOBs from face rows
        faces = [{k: v for k, v in f.items() if k != "embedding"} for f in faces]

        # Build face_clusters lookup for faces that have cluster assignments.
        # Coerce keys to str and strip representative_embedding BLOBs here
        # (presentation concern) so get_face_clusters_by_ids can return raw rows.
        cluster_ids: set[int] = {
            cast(int, f["cluster_id"]) for f in faces if f.get("cluster_id") is not None
        }
        raw_clusters = self.get_face_clusters_by_ids(list(cluster_ids))
        face_clusters: dict[str, dict[str, object]] = {
            str(cid): {k: v for k, v in cluster.items() if k != "representative_embedding"}
            for cid, cluster in raw_clusters.items()
        }

        return {
            "media": media,
            "transcript": transcript,
            "detections": detections,
            "faces": faces,
            "audio_events": audio_events,
            "embedding_count": embedding_count,
            "face_clusters": face_clusters,
        }

    def update_media_status(self, media_id: str, status: str) -> None:
        """Update the status of a media file."""
        self.conn.execute(
            "UPDATE media_files SET status = ? WHERE id = ?",
            (status, media_id),
        )

    def list_all_media(self) -> list[dict[str, object]]:
        """List all media files."""
        cur = self.conn.execute("SELECT * FROM media_files")
        return [dict(row) for row in cur.fetchall()]

    def list_by_status(self, status: str) -> list[dict[str, object]]:
        """List all media files with a given status."""
        cur = self.conn.execute("SELECT * FROM media_files WHERE status = ?", (status,))
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

    def upsert_transcript(self, media_id: str, segments_json: str, language: str) -> None:
        """Insert or replace a transcript for a media file."""
        self.conn.execute(
            "INSERT OR REPLACE INTO transcripts "
            "(media_id, segments_json, language) VALUES (?, ?, ?)",
            (media_id, segments_json, language),
        )

    def get_transcript(self, media_id: str) -> dict[str, object] | None:
        """Get a transcript by media_id, or None if not found."""
        cur = self.conn.execute("SELECT * FROM transcripts WHERE media_id = ?", (media_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    # -- shot_boundaries CRUD ---------------------------------------------------

    def upsert_boundaries(self, media_id: str, boundaries_json: str, method: str) -> None:
        """Insert or replace shot boundaries for a media file and method."""
        self.conn.execute(
            "INSERT OR REPLACE INTO shot_boundaries "
            "(media_id, boundaries_json, method) VALUES (?, ?, ?)",
            (media_id, boundaries_json, method),
        )

    def get_boundaries(
        self, media_id: str, method: str | None = None
    ) -> dict[str, object] | list[dict[str, object]] | None:
        """Get shot boundaries for a media file.

        If method is specified, returns a single dict or None.
        If method is None, returns a list of all boundaries for that media.
        """
        if method is not None:
            cur = self.conn.execute(
                "SELECT * FROM shot_boundaries WHERE media_id = ? AND method = ?",
                (media_id, method),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        cur = self.conn.execute(
            "SELECT * FROM shot_boundaries WHERE media_id = ?",
            (media_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    # -- detections CRUD --------------------------------------------------------

    def batch_insert_detections(self, rows: list[tuple[str, int, str]]) -> None:
        """Batch insert detection rows: (media_id, frame_number, detections_json)."""
        if not rows:
            return
        self.conn.executemany(
            "INSERT INTO detections (media_id, frame_number, detections_json) VALUES (?, ?, ?)",
            rows,
        )

    def get_detections_for_frame(
        self, media_id: str, frame_number: int
    ) -> dict[str, object] | None:
        """Get detections for a specific frame, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM detections WHERE media_id = ? AND frame_number = ?",
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

    # -- faces CRUD -------------------------------------------------------------

    def batch_insert_faces(
        self, rows: list[tuple[str, int, int, str, bytes | None, int | None]]
    ) -> None:
        """Batch insert face rows.

        Each row: (media_id, frame_number, face_index, bbox_json, embedding, cluster_id).
        """
        if not rows:
            return
        self.conn.executemany(
            "INSERT INTO faces "
            "(media_id, frame_number, face_index, bbox_json, embedding, cluster_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )

    def get_faces_for_frame(self, media_id: str, frame_number: int) -> list[dict[str, object]]:
        """Get all faces for a specific media_id and frame_number."""
        cur = self.conn.execute(
            "SELECT * FROM faces WHERE media_id = ? AND frame_number = ?",
            (media_id, frame_number),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_faces_for_media(self, media_id: str) -> list[dict[str, object]]:
        """Get all faces for a media_id."""
        cur = self.conn.execute(
            "SELECT * FROM faces WHERE media_id = ?",
            (media_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_all_face_embeddings(self) -> list[dict[str, object]]:
        """Get all face rows with non-null embedding."""
        cur = self.conn.execute(
            "SELECT media_id, frame_number, face_index, embedding "
            "FROM faces WHERE embedding IS NOT NULL"
        )
        return [dict(row) for row in cur.fetchall()]

    def clear_face_clusters(self) -> None:
        """Delete all rows from face_clusters table."""
        self.conn.execute("DELETE FROM face_clusters")

    def reset_face_cluster_ids(self) -> None:
        """Set cluster_id to NULL for all faces."""
        self.conn.execute("UPDATE faces SET cluster_id = NULL")

    def batch_update_face_cluster_ids(self, updates: list[tuple[int, str, int, int]]) -> None:
        """Batch update cluster_id for faces: (cluster_id, media_id, frame_number, face_index)."""
        if not updates:
            return
        self.conn.executemany(
            "UPDATE faces SET cluster_id = ? "
            "WHERE media_id = ? AND frame_number = ? AND face_index = ?",
            updates,
        )

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

    def update_face_label(self, cluster_id: int, label: str) -> None:
        """Update the label of a face cluster."""
        self.conn.execute(
            "UPDATE face_clusters SET label = ? WHERE cluster_id = ?",
            (label, cluster_id),
        )

    def get_face_clusters(self) -> list[dict[str, object]]:
        """List all face clusters."""
        cur = self.conn.execute("SELECT * FROM face_clusters")
        return [dict(row) for row in cur.fetchall()]

    def get_face_clusters_by_ids(
        self, cluster_ids: list[int]
    ) -> dict[int, dict[str, object]]:
        """Batch-fetch face clusters by IDs using a single query.

        Returns a dict mapping cluster_id (int) to raw cluster row.
        Non-existent IDs are silently skipped; empty input returns an
        empty dict.
        """
        if not cluster_ids:
            return {}
        placeholders = ",".join("?" for _ in cluster_ids)
        cur = self.conn.execute(
            f"SELECT * FROM face_clusters WHERE cluster_id IN ({placeholders})",
            cluster_ids,
        )
        return {row["cluster_id"]: dict(row) for row in cur.fetchall()}

    def get_face_cluster_by_id(self, cluster_id: int) -> dict[str, object] | None:
        """Get a face cluster by id, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM face_clusters WHERE cluster_id = ?",
            (cluster_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # -- clip_embeddings CRUD ---------------------------------------------------

    def batch_insert_embeddings(self, rows: list[tuple[str, int, bytes]]) -> None:
        """Batch insert clip embeddings: (media_id, frame_number, embedding)."""
        if not rows:
            return
        self.conn.executemany(
            "INSERT INTO clip_embeddings (media_id, frame_number, embedding) VALUES (?, ?, ?)",
            rows,
        )

    def get_embeddings_for_media(self, media_id: str) -> list[dict[str, object]]:
        """Get all clip embeddings for a media file."""
        cur = self.conn.execute(
            "SELECT * FROM clip_embeddings WHERE media_id = ?",
            (media_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_all_clip_embeddings(self) -> list[dict[str, object]]:
        """Get all clip embeddings across all media files."""
        cur = self.conn.execute("SELECT * FROM clip_embeddings")
        return [dict(row) for row in cur.fetchall()]

    # -- audio_events CRUD ------------------------------------------------------

    def batch_insert_audio_events(self, rows: list[tuple[str, float, str]]) -> None:
        """Batch insert audio events: (media_id, timestamp_seconds, events_json)."""
        if not rows:
            return
        self.conn.executemany(
            "INSERT INTO audio_events (media_id, timestamp_seconds, events_json) VALUES (?, ?, ?)",
            rows,
        )

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

    def get_audio_events_for_media(self, media_id: str) -> list[dict[str, object]]:
        """Get all audio events for a media file."""
        cur = self.conn.execute(
            "SELECT * FROM audio_events WHERE media_id = ?",
            (media_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def get_detections_for_media(self, media_id: str) -> list[dict[str, object]]:
        """Get all detections for a media file."""
        cur = self.conn.execute(
            "SELECT * FROM detections WHERE media_id = ?",
            (media_id,),
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
        excluded: int = 0,
    ) -> None:
        """Insert a new activity cluster."""
        self.conn.execute(
            "INSERT INTO activity_clusters "
            "(cluster_id, label, description, time_start, time_end, "
            "location_label, gps_center_lat, gps_center_lon, clip_ids_json, excluded) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                excluded,
            ),
        )

    def get_activity_cluster(self, cluster_id: str) -> dict[str, object] | None:
        """Return a single activity cluster by ID, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM activity_clusters WHERE cluster_id = ?",
            (cluster_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_activity_clusters(self) -> list[dict[str, object]]:
        """List all activity clusters."""
        cur = self.conn.execute("SELECT * FROM activity_clusters")
        return [dict(row) for row in cur.fetchall()]

    def get_activity_clusters_by_ids(
        self, cluster_ids: list[str],
    ) -> dict[str, dict[str, object]]:
        """Return activity clusters for the given IDs as a dict keyed by cluster_id.

        Non-existent IDs are silently omitted. Empty input returns {}.
        """
        if not cluster_ids:
            return {}
        placeholders = ",".join("?" for _ in cluster_ids)
        cur = self.conn.execute(
            f"SELECT * FROM activity_clusters WHERE cluster_id IN ({placeholders})",
            cluster_ids,
        )
        return {row["cluster_id"]: dict(row) for row in cur.fetchall()}

    def batch_delete_activity_clusters(self, cluster_ids: list[str]) -> int:
        """Delete multiple activity clusters in one query.

        Returns number of rows actually deleted. Empty input returns 0.
        Non-existent IDs are silently skipped.
        """
        if not cluster_ids:
            return 0
        placeholders = ",".join("?" for _ in cluster_ids)
        cur = self.conn.execute(
            f"DELETE FROM activity_clusters WHERE cluster_id IN ({placeholders})",
            cluster_ids,
        )
        return cur.rowcount

    def count_non_excluded_clusters(self) -> int:
        """Return the number of non-excluded activity clusters."""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM activity_clusters WHERE excluded = 0",
        )
        return cur.fetchone()[0]  # type: ignore[index]

    def clear_activity_clusters(self) -> None:
        """Delete all rows from activity_clusters table."""
        self.conn.execute("DELETE FROM activity_clusters")

    _CLUSTER_ALLOWED_COLUMNS: frozenset[str] = frozenset({
        "label",
        "description",
        "time_start",
        "time_end",
        "location_label",
        "gps_center_lat",
        "gps_center_lon",
        "clip_ids_json",
        "excluded",
    })

    def update_activity_cluster(self, cluster_id: str, **kwargs: object) -> int:
        """Update fields of an activity cluster by keyword arguments.

        Returns number of rows affected (0 if not found or no kwargs).
        """
        if not kwargs:
            return 0
        bad_keys = set(kwargs) - self._CLUSTER_ALLOWED_COLUMNS
        if bad_keys:
            msg = f"Disallowed column(s) for cluster update: {sorted(bad_keys)}"
            raise ValueError(msg)
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(cluster_id)
        cur = self.conn.execute(
            f"UPDATE activity_clusters SET {set_clause} "  # noqa: S608
            "WHERE cluster_id = ?",
            values,
        )
        return cur.rowcount

    def delete_activity_cluster(self, cluster_id: str) -> None:
        """Delete a single activity cluster by id."""
        self.conn.execute(
            "DELETE FROM activity_clusters WHERE cluster_id = ?",
            (cluster_id,),
        )

    # -- narratives CRUD --------------------------------------------------------

    def insert_narrative(
        self,
        narrative_id: str,
        *,
        title: str | None = None,
        description: str | None = None,
        proposed_duration_seconds: float | None = None,
        activity_cluster_ids_json: str | None = None,
        arc_notes: str | None = None,
        emotional_journey: str | None = None,
        status: str = "proposed",
    ) -> None:
        """Insert a new narrative."""
        self.conn.execute(
            "INSERT INTO narratives "
            "(narrative_id, title, description, proposed_duration_seconds, "
            "activity_cluster_ids_json, arc_notes, emotional_journey, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                narrative_id,
                title,
                description,
                proposed_duration_seconds,
                activity_cluster_ids_json,
                arc_notes,
                emotional_journey,
                status,
            ),
        )

    def get_narrative(self, narrative_id: str) -> dict[str, object] | None:
        """Get a narrative by id, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM narratives WHERE narrative_id = ?",
            (narrative_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def update_narrative_status(self, narrative_id: str, status: str) -> int:
        """Update the status of a narrative. Returns number of rows affected."""
        cur = self.conn.execute(
            "UPDATE narratives SET status = ? WHERE narrative_id = ?",
            (status, narrative_id),
        )
        return cur.rowcount

    _NARRATIVE_ALLOWED_COLUMNS: frozenset[str] = frozenset({
        "title",
        "description",
        "proposed_duration_seconds",
        "activity_cluster_ids_json",
        "arc_notes",
        "emotional_journey",
        "status",
    })

    def update_narrative(self, narrative_id: str, **kwargs: object) -> None:
        """Update fields of a narrative by keyword arguments."""
        if not kwargs:
            return
        bad_keys = set(kwargs) - self._NARRATIVE_ALLOWED_COLUMNS
        if bad_keys:
            msg = f"Disallowed column(s) for narrative update: {sorted(bad_keys)}"
            raise ValueError(msg)
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(narrative_id)
        self.conn.execute(
            f"UPDATE narratives SET {set_clause} WHERE narrative_id = ?",  # noqa: S608 — column names validated above
            values,
        )

    def list_narratives(self, status: str | None = None) -> list[dict[str, object]]:
        """List all narratives, optionally filtered by status."""
        if status is not None:
            cur = self.conn.execute("SELECT * FROM narratives WHERE status = ?", (status,))
        else:
            cur = self.conn.execute("SELECT * FROM narratives")
        return [dict(row) for row in cur.fetchall()]

    # -- edit_plans CRUD --------------------------------------------------------

    def upsert_edit_plan(
        self,
        narrative_id: str,
        edl_json: str | None = None,
        *,
        otio_path: str | None = None,
        validation_json: str | None = None,
        render_path: str | None = None,
    ) -> None:
        """Insert or replace an edit plan for a narrative.

        When updating an existing plan (e.g. adding render_path after render),
        only the provided non-None fields are updated. If no plan exists yet,
        a new row is inserted.
        """
        existing = self.get_edit_plan(narrative_id)
        if existing is not None:
            # Merge: keep existing values for fields not explicitly provided
            edl_json = (
                edl_json if edl_json is not None else cast("str | None", existing.get("edl_json"))
            )
            otio_path = (
                otio_path
                if otio_path is not None
                else cast("str | None", existing.get("otio_path"))
            )
            validation_json = (
                validation_json
                if validation_json is not None
                else cast("str | None", existing.get("validation_json"))
            )
            render_path = (
                render_path
                if render_path is not None
                else cast("str | None", existing.get("render_path"))
            )

        self.conn.execute(
            "INSERT OR REPLACE INTO edit_plans "
            "(narrative_id, edl_json, otio_path, validation_json, render_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (narrative_id, edl_json, otio_path, validation_json, render_path),
        )

    def get_edit_plan(self, narrative_id: str) -> dict[str, object] | None:
        """Get an edit plan by narrative_id, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM edit_plans WHERE narrative_id = ?",
            (narrative_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_edit_plans(self) -> list[dict[str, object]]:
        """List all edit plans with narrative title via LEFT JOIN."""
        cur = self.conn.execute(
            "SELECT e.*, n.title AS narrative_title "
            "FROM edit_plans e "
            "LEFT JOIN narratives n ON e.narrative_id = n.narrative_id",
        )
        return [dict(row) for row in cur.fetchall()]

    # -- narrative_scripts CRUD -------------------------------------------------

    def upsert_narrative_script(
        self,
        narrative_id: str,
        script_json: str,
        *,
        created_at: str | None = None,
    ) -> None:
        """Insert or replace a narrative script."""
        if created_at is None:
            from datetime import datetime, timezone

            created_at = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            "INSERT OR REPLACE INTO narrative_scripts "
            "(narrative_id, script_json, created_at) VALUES (?, ?, ?)",
            (narrative_id, script_json, created_at),
        )

    def get_narrative_script(self, narrative_id: str) -> dict[str, object] | None:
        """Get a narrative script by narrative_id, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM narrative_scripts WHERE narrative_id = ?",
            (narrative_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # -- crop_paths CRUD --------------------------------------------------------

    def upsert_crop_path(
        self,
        media_id: str,
        target_aspect: str,
        subject_track_id: int,
        *,
        smoothing_tau: float | None = None,
        path_data: bytes | None = None,
    ) -> None:
        """Insert or replace a crop path."""
        self.conn.execute(
            "INSERT OR REPLACE INTO crop_paths "
            "(media_id, target_aspect, subject_track_id, "
            "smoothing_tau, path_data) VALUES (?, ?, ?, ?, ?)",
            (media_id, target_aspect, subject_track_id, smoothing_tau, path_data),
        )

    def get_crop_path(
        self,
        media_id: str,
        target_aspect: str,
        subject_track_id: int,
    ) -> dict[str, object] | None:
        """Get a crop path by composite key, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM crop_paths "
            "WHERE media_id = ? AND target_aspect = ? "
            "AND subject_track_id = ?",
            (media_id, target_aspect, subject_track_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # -- uploads CRUD -----------------------------------------------------------

    def insert_upload(
        self,
        narrative_id: str,
        *,
        youtube_video_id: str | None = None,
        youtube_url: str | None = None,
        uploaded_at: str | None = None,
        privacy_status: str = "unlisted",
    ) -> None:
        """Insert an upload record."""
        self.conn.execute(
            "INSERT INTO uploads "
            "(narrative_id, youtube_video_id, youtube_url, "
            "uploaded_at, privacy_status) VALUES (?, ?, ?, ?, ?)",
            (
                narrative_id,
                youtube_video_id,
                youtube_url,
                uploaded_at,
                privacy_status,
            ),
        )

    def list_uploads(self) -> list[dict[str, object]]:
        """List all uploads with narrative title via LEFT JOIN, newest first."""
        cur = self.conn.execute(
            "SELECT u.*, n.title AS narrative_title "
            "FROM uploads u "
            "LEFT JOIN narratives n ON u.narrative_id = n.narrative_id "
            "ORDER BY u.uploaded_at DESC",
        )
        return [dict(row) for row in cur.fetchall()]

    def get_upload(self, narrative_id: str) -> dict[str, object] | None:
        """Get an upload by narrative_id, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM uploads WHERE narrative_id = ?",
            (narrative_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    # -- captions CRUD ----------------------------------------------------------

    def upsert_caption(
        self,
        media_id: str,
        start_time: float,
        end_time: float,
        caption: str,
        model_name: str,
    ) -> None:
        """Insert or replace a caption for a media clip segment."""
        self.conn.execute(
            "INSERT OR REPLACE INTO captions "
            "(media_id, start_time, end_time, caption, model_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (media_id, start_time, end_time, caption, model_name),
        )

    def get_caption(
        self, media_id: str, start_time: float, end_time: float
    ) -> dict[str, object] | None:
        """Get a caption by composite key, or None if not found."""
        cur = self.conn.execute(
            "SELECT * FROM captions WHERE media_id = ? AND start_time = ? AND end_time = ?",
            (media_id, start_time, end_time),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_captions_for_media(self, media_id: str) -> list[dict[str, object]]:
        """Get all captions for a media file, ordered by start_time."""
        cur = self.conn.execute(
            "SELECT * FROM captions WHERE media_id = ? ORDER BY start_time",
            (media_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    # -- Checkpoint convenience methods -----------------------------------------

    def has_transcript(self, media_id: str) -> bool:
        """Return True if a transcript exists for the given media_id."""
        cur = self.conn.execute(
            "SELECT 1 FROM transcripts WHERE media_id = ? LIMIT 1",
            (media_id,),
        )
        return cur.fetchone() is not None

    def has_detections(self, media_id: str) -> bool:
        """Return True if any detections exist for the given media_id."""
        cur = self.conn.execute(
            "SELECT 1 FROM detections WHERE media_id = ? LIMIT 1",
            (media_id,),
        )
        return cur.fetchone() is not None

    def has_boundaries(self, media_id: str) -> bool:
        """Return True if any shot boundaries exist for the given media_id."""
        cur = self.conn.execute(
            "SELECT 1 FROM shot_boundaries WHERE media_id = ? LIMIT 1",
            (media_id,),
        )
        return cur.fetchone() is not None

    def has_faces(self, media_id: str) -> bool:
        """Return True if any faces exist for the given media_id."""
        cur = self.conn.execute(
            "SELECT 1 FROM faces WHERE media_id = ? LIMIT 1",
            (media_id,),
        )
        return cur.fetchone() is not None

    def has_embeddings(self, media_id: str) -> bool:
        """Return True if any clip embeddings exist for the given media_id."""
        cur = self.conn.execute(
            "SELECT 1 FROM clip_embeddings WHERE media_id = ? LIMIT 1",
            (media_id,),
        )
        return cur.fetchone() is not None

    def has_audio_events(self, media_id: str) -> bool:
        """Return True if any audio events exist for the given media_id."""
        cur = self.conn.execute(
            "SELECT 1 FROM audio_events WHERE media_id = ? LIMIT 1",
            (media_id,),
        )
        return cur.fetchone() is not None

    def has_captions(self, media_id: str) -> bool:
        """Return True if any captions exist for the given media_id."""
        cur = self.conn.execute(
            "SELECT 1 FROM captions WHERE media_id = ? LIMIT 1",
            (media_id,),
        )
        return cur.fetchone() is not None

    # -- Media query (paginated, filtered, with analysis flags) ----------------

    _QUERY_MEDIA_SORT_WHITELIST = frozenset(
        {
            "file_path",
            "duration_seconds",
            "created_at",
            "resolution_w",
            "status",
        }
    )

    def query_media(
        self,
        *,
        q: str | None = None,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        sort: str = "created_at",
        order: str = "desc",
        page: int = 1,
        per_page: int = 50,
    ) -> dict:
        """Query media files with filtering, sorting, pagination, and analysis flags.

        Returns ``{"items": [...], "total": int}`` where each item is a dict
        with all media_files columns plus boolean ``has_transcript``,
        ``has_detections``, ``has_faces``, ``has_embeddings``,
        ``has_audio_events``, and ``has_captions`` flags.
        """
        # Validate sort column against whitelist to prevent SQL injection
        if sort not in self._QUERY_MEDIA_SORT_WHITELIST:
            sort = "created_at"
        order_kw = "ASC" if order.lower() == "asc" else "DESC"

        # Build WHERE clauses
        where_parts: list[str] = []
        params: list[object] = []
        if q:
            where_parts.append("m.file_path LIKE ?")
            params.append(f"%{q}%")
        if status:
            where_parts.append("m.status = ?")
            params.append(status)
        if date_from:
            where_parts.append("m.created_at >= ?")
            params.append(date_from)
        if date_to:
            where_parts.append("m.created_at <= ?")
            params.append(date_to)

        where_clause = " AND ".join(where_parts) if where_parts else "1"

        # Count total matching rows
        count_sql = f"SELECT COUNT(*) FROM media_files m WHERE {where_clause}"
        total = self.conn.execute(count_sql, params).fetchone()[0]

        # Main query with analysis flags via EXISTS subqueries
        offset = (page - 1) * per_page
        sql = f"""
            SELECT m.*,
                EXISTS(SELECT 1 FROM transcripts  WHERE media_id = m.id)
                    AS has_transcript,
                EXISTS(SELECT 1 FROM detections    WHERE media_id = m.id)
                    AS has_detections,
                EXISTS(SELECT 1 FROM faces         WHERE media_id = m.id)
                    AS has_faces,
                EXISTS(SELECT 1 FROM clip_embeddings WHERE media_id = m.id)
                    AS has_embeddings,
                EXISTS(SELECT 1 FROM audio_events  WHERE media_id = m.id)
                    AS has_audio_events,
                EXISTS(SELECT 1 FROM captions      WHERE media_id = m.id)
                    AS has_captions
            FROM media_files m
            WHERE {where_clause}
            ORDER BY m.{sort} {order_kw}
            LIMIT ? OFFSET ?
        """
        rows = self.conn.execute(sql, [*params, per_page, offset]).fetchall()
        items = []
        for row in rows:
            d = dict(row)
            # SQLite returns 0/1 for EXISTS — normalize to bool
            for flag in (
                "has_transcript",
                "has_detections",
                "has_faces",
                "has_embeddings",
                "has_audio_events",
                "has_captions",
            ):
                d[flag] = bool(d[flag])
            items.append(d)

        return {"items": items, "total": total}

    # -- pipeline_gates CRUD ---------------------------------------------------

    _PIPELINE_STAGES = (
        "ingest",
        "analyze",
        "classify",
        "narrate",
        "script",
        "edl",
        "source",
        "render",
        "upload",
    )

    def init_default_gates(self) -> None:
        """Insert default gate rows for all known pipeline stages.

        Uses INSERT OR IGNORE so calling multiple times is safe and
        won't overwrite existing gate settings.
        """
        for stage in self._PIPELINE_STAGES:
            self.conn.execute(
                "INSERT OR IGNORE INTO pipeline_gates (stage) VALUES (?)",
                (stage,),
            )

    def get_gate(self, stage: str) -> dict[str, object] | None:
        """Return the gate row for *stage*, or ``None`` if not found."""
        cur = self.conn.execute(
            "SELECT * FROM pipeline_gates WHERE stage = ?",
            (stage,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_gates(self) -> list[dict[str, object]]:
        """Return all gate rows."""
        cur = self.conn.execute("SELECT * FROM pipeline_gates ORDER BY stage")
        return [dict(row) for row in cur.fetchall()]

    _GATE_ALLOWED_COLUMNS: frozenset[str] = frozenset({
        "mode",
        "status",
        "decided_at",
        "decided_by",
        "notes",
        "timeout_hours",
    })

    def update_gate(self, stage: str, **kwargs: object) -> None:
        """Update fields of a gate by keyword arguments."""
        if not kwargs:
            return
        bad_keys = set(kwargs) - self._GATE_ALLOWED_COLUMNS
        if bad_keys:
            msg = f"Disallowed column(s) for gate update: {sorted(bad_keys)}"
            raise ValueError(msg)
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(stage)
        self.conn.execute(
            f"UPDATE pipeline_gates SET {set_clause} "  # noqa: S608 — column names validated above
            "WHERE stage = ?",
            values,
        )

    # -- pipeline_jobs CRUD ----------------------------------------------------

    def insert_job(
        self,
        job_id: str,
        stage: str,
        job_type: str,
        *,
        target_id: str | None = None,
        target_label: str | None = None,
        status: str = "pending",
        started_at: str | None = None,
        finished_at: str | None = None,
        duration_seconds: float | None = None,
        progress_pct: float | None = None,
        error_message: str | None = None,
        worker: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Insert a new pipeline job row."""
        self.conn.execute(
            "INSERT INTO pipeline_jobs "
            "(job_id, stage, job_type, target_id, target_label, status, "
            "started_at, finished_at, duration_seconds, progress_pct, "
            "error_message, worker, run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                stage,
                job_type,
                target_id,
                target_label,
                status,
                started_at,
                finished_at,
                duration_seconds,
                progress_pct,
                error_message,
                worker,
                run_id,
            ),
        )

    def get_job(self, job_id: str) -> dict[str, object] | None:
        """Return the job row for *job_id*, or ``None`` if not found."""
        cur = self.conn.execute(
            "SELECT * FROM pipeline_jobs WHERE job_id = ?",
            (job_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def update_job(self, job_id: str, **kwargs: object) -> None:
        """Update fields of a job by keyword arguments."""
        if not kwargs:
            return
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(job_id)
        self.conn.execute(
            f"UPDATE pipeline_jobs SET {set_clause} "  # noqa: S608
            "WHERE job_id = ?",
            values,
        )

    def list_jobs(
        self,
        *,
        stage: str | None = None,
        status: str | None = None,
        job_type: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Return jobs matching optional filters (AND logic)."""
        clauses: list[str] = []
        params: list[object] = []
        if stage is not None:
            clauses.append("stage = ?")
            params.append(stage)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if job_type is not None:
            clauses.append("job_type = ?")
            params.append(job_type)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)

        sql = "SELECT * FROM pipeline_jobs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def count_jobs_by_status(self, stage: str, *, run_id: str | None = None) -> dict[str, int]:
        """Return ``{status: count}`` for jobs in *stage*."""
        params: list[object] = [stage]
        sql = "SELECT status, count(*) FROM pipeline_jobs WHERE stage = ?"
        if run_id is not None:
            sql += " AND run_id = ?"
            params.append(run_id)
        sql += " GROUP BY status"
        cur = self.conn.execute(sql, params)
        return {row[0]: row[1] for row in cur.fetchall()}

    # -- pipeline_events CRUD --------------------------------------------------

    def insert_event(
        self,
        event_type: str,
        *,
        stage: str | None = None,
        job_id: str | None = None,
        payload_json: str | None = None,
    ) -> int:
        """Insert a pipeline event and return the generated event_id."""
        cur = self.conn.execute(
            "INSERT INTO pipeline_events (event_type, stage, job_id, payload_json) "
            "VALUES (?, ?, ?, ?)",
            (event_type, stage, job_id, payload_json),
        )
        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("insert_event: lastrowid is None after INSERT")
        return rowid

    def get_events_since(self, event_id: int) -> list[dict[str, object]]:
        """Return events with event_id > *event_id*, ordered ascending."""
        cur = self.conn.execute(
            "SELECT * FROM pipeline_events WHERE event_id > ? ORDER BY event_id",
            (event_id,),
        )
        return [dict(row) for row in cur.fetchall()]

    def prune_events(self, *, hours: int = 24) -> None:
        """Delete events older than *hours* hours."""
        self.conn.execute(
            f"DELETE FROM pipeline_events WHERE created_at < datetime('now', '-{hours} hours')"  # noqa: S608
        )

    # -- pipeline_runs CRUD ----------------------------------------------------

    def insert_run(
        self,
        run_id: str,
        *,
        started_at: str,
        config_snapshot: str | None = None,
        current_stage: str | None = None,
        status: str = "running",
        budget_remaining_seconds: float | None = None,
    ) -> None:
        """Insert a new pipeline run row."""
        self.conn.execute(
            "INSERT INTO pipeline_runs "
            "(run_id, started_at, config_snapshot, current_stage, "
            "status, budget_remaining_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                started_at,
                config_snapshot,
                current_stage,
                status,
                budget_remaining_seconds,
            ),
        )

    def get_run(self, run_id: str) -> dict[str, object] | None:
        """Return the run row for *run_id*, or ``None`` if not found."""
        cur = self.conn.execute(
            "SELECT * FROM pipeline_runs WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def update_run(self, run_id: str, **kwargs: object) -> None:
        """Update fields of a run by keyword arguments."""
        if not kwargs:
            return
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values())
        values.append(run_id)
        self.conn.execute(
            f"UPDATE pipeline_runs SET {set_clause} "  # noqa: S608
            "WHERE run_id = ?",
            values,
        )

    def get_current_run(self) -> dict[str, object] | None:
        """Return the most recently started running run, or ``None``."""
        cur = self.conn.execute(
            "SELECT * FROM pipeline_runs WHERE status = 'running' ORDER BY started_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_runs(self) -> list[dict[str, object]]:
        """Return all runs ordered by started_at descending."""
        cur = self.conn.execute("SELECT * FROM pipeline_runs ORDER BY started_at DESC")
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()
