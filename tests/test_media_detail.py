"""Tests for the media detail page, tab endpoints, and detail API."""

from __future__ import annotations

import json
from collections.abc import Iterator
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


@pytest.fixture
def detail_db_path(tmp_path: Path) -> str:
    """Return the path for a test catalog DB file."""
    return str(tmp_path / "catalog.db")


@pytest.fixture
def detail_db(detail_db_path: str) -> Iterator[CatalogDB]:
    """Create a CatalogDB backed by a real file for web endpoint tests."""
    db = CatalogDB(detail_db_path)
    db.conn.isolation_level = None  # autocommit
    yield db
    db.close()


@pytest.fixture
def detail_seeded_db(detail_db: CatalogDB) -> CatalogDB:
    """DB seeded with test1 (full analysis data) and test2 (no analysis)."""
    db = detail_db

    # test1: media with full metadata
    db.insert_media(
        "test1",
        "/video/beach_sunset.mp4",
        codec="h264",
        fps=30.0,
        resolution_w=1920,
        resolution_h=1080,
        gps_lat=34.0522,
        gps_lon=-118.2437,
        sha256_prefix="abc123def456",
        audio_channels=2,
        duration_seconds=120.5,
        created_at="2025-06-15T10:30:00",
        status="analyzed",
        metadata_json=json.dumps({"camera": "GoPro Hero 12", "scene": "outdoor"}),
    )

    # Transcript with 3 segments
    segments = [
        {"start": 0.0, "end": 10.0, "text": "Hello everyone", "speaker": "Alice"},
        {"start": 10.5, "end": 20.0, "text": "Welcome to the beach", "speaker": "Alice"},
        {"start": 25.0, "end": 35.0, "text": "What a beautiful day", "speaker": "Bob"},
    ]
    db.upsert_transcript("test1", json.dumps(segments), "en")

    # Detections across 3 frames
    det_frame0 = [
        {"class": "person", "confidence": 0.95, "bbox": [10, 20, 100, 200]},
        {"class": "car", "confidence": 0.87, "bbox": [200, 50, 350, 150]},
    ]
    det_frame1 = [
        {"class": "person", "confidence": 0.92, "bbox": [15, 25, 105, 205]},
        {"class": "dog", "confidence": 0.80, "bbox": [300, 100, 400, 200]},
    ]
    det_frame2 = [
        {"class": "person", "confidence": 0.90, "bbox": [20, 30, 110, 210]},
    ]
    db.batch_insert_detections(
        [
            ("test1", 0, json.dumps(det_frame0)),
            ("test1", 30, json.dumps(det_frame1)),
            ("test1", 60, json.dumps(det_frame2)),
        ]
    )

    # 2 faces on separate frames, assigned to cluster_id=1 (with real BLOB embeddings)
    db.batch_insert_faces(
        [
            ("test1", 0, 0, json.dumps({"x": 10, "y": 20, "w": 50, "h": 50}), b"\x00" * 8, 1),
            ("test1", 30, 0, json.dumps({"x": 15, "y": 25, "w": 55, "h": 55}), b"\x00" * 8, 1),
        ]
    )

    # Face cluster with label 'Alice' (with real BLOB representative_embedding)
    db.insert_face_cluster(1, label="Alice", representative_embedding=b"\x00" * 8)

    # 2 audio events with classified events_json
    audio_events_0 = [
        {"class": "Speech", "confidence": 0.95},
        {"class": "Music", "confidence": 0.3},
    ]
    audio_events_1 = [
        {"class": "Ocean waves", "confidence": 0.88},
    ]
    db.batch_insert_audio_events(
        [
            ("test1", 5.0, json.dumps(audio_events_0)),
            ("test1", 60.0, json.dumps(audio_events_1)),
        ]
    )

    # 3 clip embeddings
    db.batch_insert_embeddings(
        [
            ("test1", 0, b"\x00" * 16),
            ("test1", 30, b"\x01" * 16),
            ("test1", 60, b"\x02" * 16),
        ]
    )

    # test2: media with no analysis data
    db.insert_media(
        "test2",
        "/video/forest_walk.mp4",
        duration_seconds=45.0,
        created_at="2025-06-16T08:00:00",
        status="ingested",
    )

    return db


@pytest.fixture
def detail_app(detail_seeded_db: CatalogDB, detail_db_path: str):
    """FastAPI app pointing at the seeded detail DB."""
    from autopilot.web.app import create_app

    return create_app(detail_db_path)


@pytest.fixture
def detail_client(detail_app):
    """TestClient for media detail endpoint tests."""
    from starlette.testclient import TestClient

    return TestClient(detail_app)


# ---------------------------------------------------------------------------
# Redundant 'detail' template context test (S11)
# ---------------------------------------------------------------------------


class TestRedundantDetailContext:
    """Test that the 'detail' dict should not be passed to template context."""

    def test_detail_page_context_does_not_include_detail_key(self) -> None:
        """media_detail_page template context should not include 'detail' key.

        The detail.html and tab_metadata.html templates only use 'media',
        'extra_metadata', and 'format_duration'. Verify by inspecting the
        source of media_detail_page for 'detail' in the context dict.
        """
        import inspect

        from autopilot.web.routes.media import media_detail_page

        source = inspect.getsource(media_detail_page)
        # After removal, the context dict should not contain '"detail"' key
        assert '"detail": detail' not in source, (
            "media_detail_page still passes 'detail' to template context"
        )


# ---------------------------------------------------------------------------
# CatalogDB.get_face_clusters_by_ids() tests
# ---------------------------------------------------------------------------


class TestGetFaceClustersByIds:
    """Tests for CatalogDB.get_face_clusters_by_ids(cluster_ids).

    These tests verify the raw-row contract: int keys and
    representative_embedding included (no presentation transforms).
    """

    def test_returns_dict_with_int_keys(self, detail_seeded_db: CatalogDB) -> None:
        """Returns dict mapping int cluster_id to cluster info."""
        result = detail_seeded_db.get_face_clusters_by_ids([1])
        assert 1 in result
        assert result[1]["label"] == "Alice"

    def test_includes_representative_embedding(self, detail_seeded_db: CatalogDB) -> None:
        """Returned cluster info includes representative_embedding (raw row)."""
        result = detail_seeded_db.get_face_clusters_by_ids([1])
        assert "representative_embedding" in result[1]

    def test_returns_empty_dict_for_empty_input(self, detail_seeded_db: CatalogDB) -> None:
        """Returns empty dict when given empty list."""
        result = detail_seeded_db.get_face_clusters_by_ids([])
        assert result == {}

    def test_skips_nonexistent_ids(self, detail_seeded_db: CatalogDB) -> None:
        """Silently skips IDs that don't exist in face_clusters table."""
        result = detail_seeded_db.get_face_clusters_by_ids([1, 999])
        assert 1 in result
        assert 999 not in result


# ---------------------------------------------------------------------------
# CatalogDB.get_media_detail() tests
# ---------------------------------------------------------------------------


class TestGetMediaDetail:
    """Tests for CatalogDB.get_media_detail(media_id)."""

    def test_returns_none_for_nonexistent_media(self, catalog_db: CatalogDB) -> None:
        """get_media_detail returns None when media_id does not exist."""
        result = catalog_db.get_media_detail("nonexistent")
        assert result is None

    def test_returns_dict_with_media_key(self, detail_seeded_db: CatalogDB) -> None:
        """get_media_detail returns a dict with 'media' key containing media row data."""
        result = detail_seeded_db.get_media_detail("test1")
        assert result is not None
        assert "media" in result
        assert result["media"]["id"] == "test1"
        assert result["media"]["codec"] == "h264"
        assert result["media"]["fps"] == 30.0
        assert result["media"]["resolution_w"] == 1920
        assert result["media"]["resolution_h"] == 1080

    def test_includes_transcript(self, detail_seeded_db: CatalogDB) -> None:
        """get_media_detail includes 'transcript' with parsed segments and language."""
        result = detail_seeded_db.get_media_detail("test1")
        assert result is not None
        assert "transcript" in result
        transcript = result["transcript"]
        assert transcript is not None
        assert transcript["language"] == "en"
        segments = json.loads(transcript["segments_json"])
        assert len(segments) == 3

    def test_includes_detections(self, detail_seeded_db: CatalogDB) -> None:
        """get_media_detail includes 'detections' with list of detection rows."""
        result = detail_seeded_db.get_media_detail("test1")
        assert result is not None
        assert "detections" in result
        assert len(result["detections"]) == 3  # 3 frames

    def test_includes_faces(self, detail_seeded_db: CatalogDB) -> None:
        """get_media_detail includes 'faces' with list of face rows."""
        result = detail_seeded_db.get_media_detail("test1")
        assert result is not None
        assert "faces" in result
        assert len(result["faces"]) == 2

    def test_includes_audio_events(self, detail_seeded_db: CatalogDB) -> None:
        """get_media_detail includes 'audio_events' with list of audio event rows."""
        result = detail_seeded_db.get_media_detail("test1")
        assert result is not None
        assert "audio_events" in result
        assert len(result["audio_events"]) == 2

    def test_includes_embedding_count(self, detail_seeded_db: CatalogDB) -> None:
        """get_media_detail includes 'embedding_count' (no full embeddings list)."""
        result = detail_seeded_db.get_media_detail("test1")
        assert result is not None
        assert "embedding_count" in result
        assert result["embedding_count"] == 3
        assert "embeddings" not in result

    def test_face_clusters_keys_are_strings(self, detail_seeded_db: CatalogDB) -> None:
        """face_clusters dict keys are str (not int) for JSON consistency."""
        result = detail_seeded_db.get_media_detail("test1")
        assert result is not None
        fc = result["face_clusters"]
        assert len(fc) > 0, "Expected at least one face cluster"
        for key in fc:
            assert isinstance(key, str), (
                f"face_clusters key {key!r} should be str,"
                f" got {type(key).__name__}"
            )

    def test_media_detail_strips_cluster_embedding_independently(
        self, detail_seeded_db: CatalogDB
    ) -> None:
        """get_media_detail strips representative_embedding from face_clusters.

        This ensures the presentation layer (get_media_detail) handles embedding
        stripping independently of get_face_clusters_by_ids, so the DB method
        can return raw rows without breaking JSON-serializability.
        """
        result = detail_seeded_db.get_media_detail("test1")
        assert result is not None
        fc = result["face_clusters"]
        assert len(fc) > 0, "Expected at least one face cluster"
        for cid, cluster in fc.items():
            assert "representative_embedding" not in cluster, (
                f"face_clusters[{cid!r}] should not contain"
                " representative_embedding"
            )

    def test_empty_analysis_for_media_without_data(self, detail_seeded_db: CatalogDB) -> None:
        """get_media_detail returns empty/None for media with no analysis data."""
        result = detail_seeded_db.get_media_detail("test2")
        assert result is not None
        assert result["media"]["id"] == "test2"
        assert result["transcript"] is None
        assert result["detections"] == []
        assert result["faces"] == []
        assert result["audio_events"] == []
        assert result["embedding_count"] == 0
        assert result["face_clusters"] == {}


# ---------------------------------------------------------------------------
# GET /api/media/{media_id} JSON endpoint tests
# ---------------------------------------------------------------------------


class TestApiMediaDetail:
    """Tests for GET /api/media/{media_id} JSON endpoint."""

    def test_returns_404_for_nonexistent_media(self, detail_client) -> None:
        """GET /api/media/nonexistent returns 404."""
        resp = detail_client.get("/api/media/nonexistent")
        assert resp.status_code == 404

    def test_returns_200_with_detail_keys(self, detail_client) -> None:
        """GET /api/media/test1 returns 200 with all expected keys."""
        resp = detail_client.get("/api/media/test1")
        assert resp.status_code == 200
        data = resp.json()
        for key in (
            "media",
            "transcript",
            "detections",
            "faces",
            "audio_events",
            "embedding_count",
        ):
            assert key in data, f"Missing key: {key}"

    def test_no_blob_fields_in_json_response(self, detail_client) -> None:
        """JSON API must not contain binary BLOB fields (embedding, representative_embedding)."""
        resp = detail_client.get("/api/media/test1")
        assert resp.status_code == 200
        data = resp.json()
        for face in data["faces"]:
            assert "embedding" not in face
        for cluster in data.get("face_clusters", {}).values():
            assert "representative_embedding" not in cluster

    def test_media_section_has_expected_fields(self, detail_client) -> None:
        """Media section contains codec, fps, resolution, etc."""
        resp = detail_client.get("/api/media/test1")
        data = resp.json()
        media = data["media"]
        assert media["codec"] == "h264"
        assert media["fps"] == 30.0
        assert media["resolution_w"] == 1920
        assert media["resolution_h"] == 1080
        assert media["audio_channels"] == 2
        assert media["sha256_prefix"] == "abc123def456"


# ---------------------------------------------------------------------------
# GET /api/media/{media_id}/transcript and /detections tests
# ---------------------------------------------------------------------------


class TestApiMediaTranscript:
    """Tests for GET /api/media/{media_id}/transcript."""

    def test_returns_404_for_missing_media(self, detail_client) -> None:
        """GET /api/media/nonexistent/transcript returns 404."""
        resp = detail_client.get("/api/media/nonexistent/transcript")
        assert resp.status_code == 404

    def test_returns_200_with_empty_transcript_when_media_has_no_transcript(
        self, detail_client,
    ) -> None:
        """GET /api/media/test2/transcript returns 200 with empty payload when transcript absent."""
        resp = detail_client.get("/api/media/test2/transcript")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] is None
        assert data["segments"] == []

    def test_returns_200_with_segments(self, detail_client) -> None:
        """GET /api/media/test1/transcript returns parsed segments and language."""
        resp = detail_client.get("/api/media/test1/transcript")
        assert resp.status_code == 200
        data = resp.json()
        assert data["language"] == "en"
        assert "segments" in data
        assert len(data["segments"]) == 3

    def test_segment_fields(self, detail_client) -> None:
        """Each segment has start, end, text, speaker fields with correct types."""
        resp = detail_client.get("/api/media/test1/transcript")
        data = resp.json()
        assert len(data["segments"]) > 1, "Need multiple segments for thorough validation"
        for i, seg in enumerate(data["segments"]):
            assert "start" in seg, f"segments[{i}] missing 'start'"
            assert "end" in seg, f"segments[{i}] missing 'end'"
            assert "text" in seg, f"segments[{i}] missing 'text'"
            assert "speaker" in seg, f"segments[{i}] missing 'speaker'"
            assert isinstance(seg["start"], float), f"segments[{i}] start should be float"
            assert isinstance(seg["end"], float), f"segments[{i}] end should be float"
            assert isinstance(seg["text"], str), f"segments[{i}] text should be str"


class TestApiMediaDetections:
    """Tests for GET /api/media/{media_id}/detections."""

    def test_returns_404_for_missing_media(self, detail_client) -> None:
        """GET /api/media/nonexistent/detections returns 404."""
        resp = detail_client.get("/api/media/nonexistent/detections")
        assert resp.status_code == 404

    def test_returns_200_with_summary(self, detail_client) -> None:
        """GET /api/media/test1/detections returns detection summary."""
        resp = detail_client.get("/api/media/test1/detections")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_detections" in data
        assert data["total_detections"] == 5  # 2+2+1
        assert "classes" in data
        assert "frame_count" in data
        assert data["frame_count"] == 3

    def test_per_class_counts(self, detail_client) -> None:
        """Detection summary has per-class counts."""
        resp = detail_client.get("/api/media/test1/detections")
        data = resp.json()
        classes = data["classes"]
        assert classes["person"] == 3
        assert classes["car"] == 1
        assert classes["dog"] == 1


# ---------------------------------------------------------------------------
# GET /media/{media_id} HTML detail page tests
# ---------------------------------------------------------------------------


class TestMediaDetailPage:
    """Tests for GET /media/{media_id} HTML detail page."""

    def test_returns_404_for_nonexistent_media(self, detail_client) -> None:
        """GET /media/nonexistent returns 404."""
        resp = detail_client.get("/media/nonexistent")
        assert resp.status_code == 404

    def test_returns_200_html(self, detail_client) -> None:
        """GET /media/test1 returns 200 with text/html."""
        resp = detail_client.get("/media/test1")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")

    def test_contains_tab_navigation(self, detail_client) -> None:
        """Detail page has tab navigation labels."""
        resp = detail_client.get("/media/test1")
        html = resp.text
        for tab in ("Metadata", "Transcript", "Detections", "Faces", "Audio Events", "Embeddings"):
            assert tab in html, f"Tab '{tab}' not found in page"

    def test_extends_base_template(self, detail_client) -> None:
        """Detail page extends base.html (contains nav)."""
        resp = detail_client.get("/media/test1")
        assert "Autopilot Video" in resp.text

    def test_metadata_tab_rendered_by_default(self, detail_client) -> None:
        """Metadata tab content is rendered inline by default (codec, resolution, duration)."""
        resp = detail_client.get("/media/test1")
        html = resp.text
        assert "h264" in html
        assert "1920" in html
        assert "1080" in html

    def test_tabs_have_htmx_attributes(self, detail_client) -> None:
        """Tabs have hx-get attributes pointing to /media/{id}/tab/{name}."""
        resp = detail_client.get("/media/test1")
        html = resp.text
        assert "hx-get" in html
        assert "/media/test1/tab/" in html


# ---------------------------------------------------------------------------
# HTMX tab endpoint tests (transcript & detections)
# ---------------------------------------------------------------------------


class TestTabTranscript:
    """Tests for GET /media/{media_id}/tab/transcript."""

    def test_transcript_tab_has_timestamps(self, detail_client) -> None:
        """Transcript tab returns HTML with timestamped lines."""
        resp = detail_client.get("/media/test1/tab/transcript")
        assert resp.status_code == 200
        html = resp.text
        assert "[00:00:00]" in html or "[0:00]" in html  # first segment starts at 0.0

    def test_transcript_tab_has_speaker_labels(self, detail_client) -> None:
        """Transcript tab shows speaker labels."""
        resp = detail_client.get("/media/test1/tab/transcript")
        html = resp.text
        assert "Alice" in html
        assert "Bob" in html

    def test_transcript_tab_shows_language(self, detail_client) -> None:
        """Transcript tab shows language indicator."""
        resp = detail_client.get("/media/test1/tab/transcript")
        assert "en" in resp.text


class TestTabDetections:
    """Tests for GET /media/{media_id}/tab/detections."""

    def test_detections_tab_has_class_names(self, detail_client) -> None:
        """Detections tab returns HTML with object class names."""
        resp = detail_client.get("/media/test1/tab/detections")
        assert resp.status_code == 200
        html = resp.text
        assert "person" in html
        assert "car" in html
        assert "dog" in html

    def test_detections_tab_has_total_count(self, detail_client) -> None:
        """Detections tab shows total detection count."""
        resp = detail_client.get("/media/test1/tab/detections")
        assert "5 detections" in resp.text  # template: '5 detections across 3 frames'


# ---------------------------------------------------------------------------
# HTMX tab tests: faces, audio events, embeddings
# ---------------------------------------------------------------------------


class TestTabFaces:
    """Tests for GET /media/{media_id}/tab/faces."""

    def test_faces_tab_shows_cluster_label(self, detail_client) -> None:
        """Faces tab shows cluster label 'Alice'."""
        resp = detail_client.get("/media/test1/tab/faces")
        assert resp.status_code == 200
        assert "Alice" in resp.text

    def test_faces_tab_shows_face_count(self, detail_client) -> None:
        """Faces tab shows count of faces per cluster."""
        resp = detail_client.get("/media/test1/tab/faces")
        assert "2 appearances" in resp.text  # template: '2 appearances'


class TestTabAudioEvents:
    """Tests for GET /media/{media_id}/tab/audio_events."""

    def test_audio_events_tab_shows_event_classes(self, detail_client) -> None:
        """Audio events tab shows event class names."""
        resp = detail_client.get("/media/test1/tab/audio_events")
        assert resp.status_code == 200
        assert "Speech" in resp.text
        assert "Ocean waves" in resp.text


class TestTabEmbeddings:
    """Tests for GET /media/{media_id}/tab/embeddings."""

    def test_embeddings_tab_shows_coverage(self, detail_client) -> None:
        """Embeddings tab shows 'X of Y frames sampled' format."""
        resp = detail_client.get("/media/test1/tab/embeddings")
        assert resp.status_code == 200
        html = resp.text
        # 3 embeddings out of ~3615 frames (30fps * 120.5s)
        assert "3 of" in html  # template: '3 of 3615 frames sampled'
        assert "frames sampled" in html


# ---------------------------------------------------------------------------
# media_row.html link test
# ---------------------------------------------------------------------------


class TestMediaRowLink:
    """Test that media_row.html links filename to detail page."""

    def test_filename_links_to_detail_page(self, detail_client) -> None:
        """media_row.html partial wraps filename in link to /media/{id}."""
        resp = detail_client.get("/api/media", headers={"HX-Request": "true"})
        assert resp.status_code == 200
        html = resp.text
        # test1 should have a link to /media/test1
        assert 'href="/media/test1"' in html or "href='/media/test1'" in html


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestMediaDetailIntegration:
    """End-to-end integration tests across all detail endpoints."""

    def test_detail_page_loads_200(self, detail_client) -> None:
        """GET /media/test1 returns 200 for fully analysed media."""
        resp = detail_client.get("/media/test1")
        assert resp.status_code == 200

    def test_all_tab_endpoints_return_200(self, detail_client) -> None:
        """All tab endpoints return 200 for media with full analysis."""
        for tab in ("metadata", "transcript", "detections", "faces", "audio_events", "embeddings"):
            resp = detail_client.get(f"/media/test1/tab/{tab}")
            assert resp.status_code == 200, f"Tab {tab} returned {resp.status_code}"
            assert "text/html" in resp.headers["content-type"]

    def test_all_json_api_endpoints_return_correct_data(self, detail_client) -> None:
        """All three JSON API endpoints return correct structured data."""
        # /api/media/test1
        resp = detail_client.get("/api/media/test1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["media"]["id"] == "test1"
        assert data["media"]["codec"] == "h264"
        assert data["transcript"] is not None
        assert len(data["detections"]) == 3
        assert len(data["faces"]) == 2
        assert len(data["audio_events"]) == 2
        assert data["embedding_count"] == 3

        # /api/media/test1/transcript
        resp = detail_client.get("/api/media/test1/transcript")
        assert resp.status_code == 200
        transcript = resp.json()
        assert transcript["language"] == "en"
        assert len(transcript["segments"]) == 3
        seg = transcript["segments"][0]
        assert "start" in seg and "end" in seg and "text" in seg

        # /api/media/test1/detections
        resp = detail_client.get("/api/media/test1/detections")
        assert resp.status_code == 200
        det = resp.json()
        assert det["total_detections"] == 5  # 2 + 2 + 1
        assert det["frame_count"] == 3
        assert det["classes"]["person"] == 3

    def test_no_analysis_media_returns_empty_sections(self, detail_client) -> None:
        """Detail endpoint for test2 (no analysis) returns gracefully empty data."""
        resp = detail_client.get("/api/media/test2")
        assert resp.status_code == 200
        data = resp.json()
        assert data["media"]["id"] == "test2"
        assert data["transcript"] is None
        assert data["detections"] == []
        assert data["faces"] == []
        assert data["audio_events"] == []
        assert data["embedding_count"] == 0
        assert data["face_clusters"] == {}

    def test_no_analysis_media_detail_page_loads(self, detail_client) -> None:
        """GET /media/test2 returns 200 even with no analysis data."""
        resp = detail_client.get("/media/test2")
        assert resp.status_code == 200

    def test_no_analysis_tabs_render_gracefully(self, detail_client) -> None:
        """Tab endpoints for media with no data still return 200."""
        for tab in ("metadata", "transcript", "detections", "faces", "audio_events", "embeddings"):
            resp = detail_client.get(f"/media/test2/tab/{tab}")
            assert resp.status_code == 200, f"Tab {tab} for test2 returned {resp.status_code}"

    def test_nonexistent_media_returns_404_everywhere(self, detail_client) -> None:
        """All endpoints return 404 for non-existent media."""
        assert detail_client.get("/media/nope").status_code == 404
        assert detail_client.get("/api/media/nope").status_code == 404
        assert detail_client.get("/api/media/nope/transcript").status_code == 404
        assert detail_client.get("/api/media/nope/detections").status_code == 404
        assert detail_client.get("/media/nope/tab/metadata").status_code == 404


# ---------------------------------------------------------------------------
# Robustness tests (malformed data / None guards)
# ---------------------------------------------------------------------------


@pytest.fixture
def robustness_db_path(tmp_path: Path) -> str:
    """Return the path for the robustness test catalog DB file."""
    return str(tmp_path / "robustness.db")


@pytest.fixture
def robustness_db(robustness_db_path: str) -> CatalogDB:
    """Create a CatalogDB seeded with malformed JSON data."""
    db = CatalogDB(robustness_db_path)
    db.conn.isolation_level = None

    # Media with malformed transcript segments_json
    db.insert_media(
        "mal_transcript",
        "/video/mal_transcript.mp4",
        duration_seconds=60.0,
        fps=30.0,
    )
    db.upsert_transcript("mal_transcript", "{{corrupt", "en")

    # Media with malformed detections_json (plus one valid frame)
    db.insert_media(
        "mal_detect",
        "/video/mal_detect.mp4",
        duration_seconds=60.0,
        fps=30.0,
    )
    valid_dets = [{"class": "person", "confidence": 0.9}]
    db.batch_insert_detections(
        [
            ("mal_detect", 0, "not-json"),
            ("mal_detect", 30, json.dumps(valid_dets)),
        ]
    )

    return db


@pytest.fixture
def robustness_app(robustness_db: CatalogDB, robustness_db_path: str):
    """FastAPI app pointing at the robustness DB."""
    from autopilot.web.app import create_app

    return create_app(robustness_db_path)


@pytest.fixture
def robustness_client(robustness_app):
    """TestClient for robustness tests."""
    from starlette.testclient import TestClient

    return TestClient(robustness_app)


class TestRobustness:
    """Tests for graceful handling of None and malformed JSON."""

    def test_format_timestamp_with_none(self) -> None:
        """_format_timestamp(None) returns '--:--:--' instead of TypeError."""
        from autopilot.web.routes.media import _format_timestamp

        result = _format_timestamp(None)
        assert result == "--:--:--"

    def test_api_transcript_malformed_segments(self, robustness_client) -> None:
        """GET /api/media/{id}/transcript with corrupt segments_json returns 200."""
        resp = robustness_client.get("/api/media/mal_transcript/transcript")
        assert resp.status_code == 200
        data = resp.json()
        assert data["segments"] == []
        assert data["language"] == "en"

    def test_api_detections_malformed_json(self, robustness_client) -> None:
        """GET /api/media/{id}/detections with corrupt detections_json returns 200."""
        resp = robustness_client.get("/api/media/mal_detect/detections")
        assert resp.status_code == 200
        data = resp.json()
        # Valid frame still counted
        assert data["total_detections"] >= 1
        assert data["frame_count"] == 2
        assert data["classes"].get("person", 0) == 1

    def test_detections_tab_malformed_json(self, robustness_client) -> None:
        """GET /media/{id}/tab/detections with corrupt detections_json returns 200."""
        resp = robustness_client.get("/media/mal_detect/tab/detections")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        # Valid frame's detections should still be aggregated
        assert "person" in resp.text


# ---------------------------------------------------------------------------
# Unit tests for _parse_metadata_json helper (S5)
# ---------------------------------------------------------------------------


class TestParseMetadataJson:
    """Tests for _parse_metadata_json(media) helper."""

    def test_returns_parsed_dict_for_valid_json(self) -> None:
        """Returns parsed dict when metadata_json is valid JSON."""
        from autopilot.web.routes.media import _parse_metadata_json

        media = {"metadata_json": '{"camera": "GoPro", "scene": "outdoor"}'}
        result = _parse_metadata_json(media)
        assert result == {"camera": "GoPro", "scene": "outdoor"}

    def test_returns_empty_dict_for_malformed_json(self) -> None:
        """Returns {} when metadata_json is malformed."""
        from autopilot.web.routes.media import _parse_metadata_json

        media = {"metadata_json": "{{corrupt"}
        result = _parse_metadata_json(media)
        assert result == {}

    def test_returns_empty_dict_for_none(self) -> None:
        """Returns {} when metadata_json is None."""
        from autopilot.web.routes.media import _parse_metadata_json

        media = {"metadata_json": None}
        result = _parse_metadata_json(media)
        assert result == {}

    def test_returns_empty_dict_for_missing_key(self) -> None:
        """Returns {} when metadata_json key is absent."""
        from autopilot.web.routes.media import _parse_metadata_json

        media = {}
        result = _parse_metadata_json(media)
        assert result == {}


# ---------------------------------------------------------------------------
# Unit tests for _aggregate_detections helper (S4)
# ---------------------------------------------------------------------------


class TestAggregateDetections:
    """Tests for _aggregate_detections(det_rows) helper."""

    def test_returns_named_tuple_with_correct_fields(self) -> None:
        """Returns a DetectionSummary with total, classes, frame_details."""
        from autopilot.web.routes.media import _aggregate_detections

        rows = [
            {"frame_number": 0, "detections_json": json.dumps([{"class": "person"}])},
        ]
        result = _aggregate_detections(rows)
        assert hasattr(result, "total")
        assert hasattr(result, "classes")
        assert hasattr(result, "frame_details")

    def test_counts_detections_across_frames(self) -> None:
        """Total counts all detections across all frames."""
        from autopilot.web.routes.media import _aggregate_detections

        rows = [
            {"frame_number": 0, "detections_json": json.dumps(
                [{"class": "person"}, {"class": "car"}],
            )},
            {"frame_number": 30, "detections_json": json.dumps(
                [{"class": "person"}],
            )},
        ]
        result = _aggregate_detections(rows)
        assert result.total == 3

    def test_aggregates_class_counts(self) -> None:
        """Classes dict has per-class counts."""
        from autopilot.web.routes.media import _aggregate_detections

        rows = [
            {"frame_number": 0, "detections_json": json.dumps(
                [{"class": "person"}, {"class": "car"}],
            )},
            {"frame_number": 30, "detections_json": json.dumps(
                [{"class": "person"}, {"class": "dog"}],
            )},
        ]
        result = _aggregate_detections(rows)
        assert result.classes == {"person": 2, "car": 1, "dog": 1}

    def test_frame_details_structure(self) -> None:
        """frame_details has correct structure with number and count keys."""
        from autopilot.web.routes.media import _aggregate_detections

        rows = [
            {"frame_number": 0, "detections_json": json.dumps(
                [{"class": "person"}, {"class": "car"}],
            )},
            {"frame_number": 30, "detections_json": json.dumps(
                [{"class": "dog"}],
            )},
        ]
        result = _aggregate_detections(rows)
        assert result.frame_details == [
            {"number": 0, "count": 2},
            {"number": 30, "count": 1},
        ]

    def test_handles_empty_list(self) -> None:
        """Returns zeros and empty collections for empty input."""
        from autopilot.web.routes.media import _aggregate_detections

        result = _aggregate_detections([])
        assert result.total == 0
        assert result.classes == {}
        assert result.frame_details == []

    def test_handles_malformed_detections_json(self) -> None:
        """Malformed detections_json treated as zero detections for that frame."""
        from autopilot.web.routes.media import _aggregate_detections

        rows = [
            {"frame_number": 0, "detections_json": "{{corrupt"},
            {"frame_number": 30, "detections_json": json.dumps([{"class": "person"}])},
        ]
        result = _aggregate_detections(rows)
        assert result.total == 1
        assert result.classes == {"person": 1}
        assert result.frame_details == [
            {"number": 0, "count": 0},
            {"number": 30, "count": 1},
        ]

    def test_none_detections_json(self) -> None:
        """None detections_json is treated as zero detections (exercises `if raw:` guard)."""
        from autopilot.web.routes.media import _aggregate_detections

        rows = [{"frame_number": 0, "detections_json": None}]
        result = _aggregate_detections(rows)
        assert result.total == 0
        assert result.classes == {}
        assert result.frame_details == [{"number": 0, "count": 0}]

    def test_detection_without_class_key(self) -> None:
        """Detection dict missing 'class' key falls back to 'unknown'."""
        from autopilot.web.routes.media import _aggregate_detections

        rows = [
            {"frame_number": 0, "detections_json": json.dumps([{"confidence": 0.9}])},
        ]
        result = _aggregate_detections(rows)
        assert result.total == 1
        assert result.classes == {"unknown": 1}
        assert result.frame_details == [{"number": 0, "count": 1}]


# ---------------------------------------------------------------------------
# Unit tests for _format_seconds helper (S6+S12)
# ---------------------------------------------------------------------------


class TestFormatSeconds:
    """Tests for _format_seconds(seconds, pad_hours=False) unified formatter."""

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0:00"),
            (59, "0:59"),
            (60, "1:00"),
            (119, "1:59"),
            (3600, "1:00:00"),
            (3661, "1:01:01"),
            (None, "--:--"),
        ],
    )
    def test_pad_hours_false(self, seconds, expected) -> None:
        """pad_hours=False: M:SS for <1hr, H:MM:SS for >=1hr, None->'--:--'."""
        from autopilot.web.routes.media import _format_seconds

        assert _format_seconds(seconds, pad_hours=False) == expected

    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "00:00:00"),
            (59, "00:00:59"),
            (60, "00:01:00"),
            (119, "00:01:59"),
            (3600, "01:00:00"),
            (3661, "01:01:01"),
            (None, "--:--:--"),
        ],
    )
    def test_pad_hours_true(self, seconds, expected) -> None:
        """pad_hours=True: always 00:MM:SS, None->'--:--:--'."""
        from autopilot.web.routes.media import _format_seconds

        assert _format_seconds(seconds, pad_hours=True) == expected


# ---------------------------------------------------------------------------
# Per-tab data fetching tests (S1 over-fetching fix)
# ---------------------------------------------------------------------------


class TestTabPerTabFetching:
    """Verify that each tab only calls the DB methods it needs.

    Uses mock patching to confirm that media_tab() dispatches per-tab
    rather than calling get_media_detail() for every tab.
    """

    _DB_METHODS = (
        "get_media_detail",
        "get_transcript",
        "get_detections_for_media",
        "get_faces_for_media",
        "get_audio_events_for_media",
        "count_embeddings_for_media",
    )

    _MEDIA_ROW = {
        "id": "test1",
        "filepath": "/video/test.mp4",
        "fps": 30.0,
        "duration_seconds": 120.5,
        "metadata_json": '{"camera": "GoPro"}',
    }

    def _get_client(self):
        """Import and create a test client with mocked DB."""
        from unittest.mock import MagicMock, patch

        from starlette.testclient import TestClient

        from autopilot.web.app import create_app

        app = create_app(":memory:")
        client = TestClient(app)
        mock_db = MagicMock()
        mock_db.get_media.return_value = self._MEDIA_ROW
        return client, mock_db, patch

    def _assert_only_called(self, mock_db, *allowed: str) -> None:
        """Assert that no DB methods outside *allowed* were called on mock_db."""
        allowed_set = set(allowed)
        for name in self._DB_METHODS:
            if name not in allowed_set:
                getattr(mock_db, name).assert_not_called()

    def test_metadata_tab_does_not_call_get_transcript(self, tmp_path) -> None:
        """Metadata tab should not call get_transcript."""
        client, mock_db, patch = self._get_client()

        with patch("autopilot.web.routes.media.get_db", return_value=mock_db):
            resp = client.get("/media/test1/tab/metadata")

        assert resp.status_code == 200
        mock_db.get_transcript.assert_not_called()
        mock_db.get_detections_for_media.assert_not_called()
        mock_db.get_faces_for_media.assert_not_called()
        mock_db.get_audio_events_for_media.assert_not_called()
        mock_db.count_embeddings_for_media.assert_not_called()
        mock_db.get_media_detail.assert_not_called()

    def test_transcript_tab_calls_get_transcript_only(self, tmp_path) -> None:
        """Transcript tab should call get_transcript but not detection/face methods."""
        client, mock_db, patch = self._get_client()
        mock_db.get_transcript.return_value = {"segments_json": "[]", "language": "en"}

        with patch("autopilot.web.routes.media.get_db", return_value=mock_db):
            resp = client.get("/media/test1/tab/transcript")

        assert resp.status_code == 200
        mock_db.get_transcript.assert_called_once()
        mock_db.get_detections_for_media.assert_not_called()
        mock_db.get_faces_for_media.assert_not_called()
        mock_db.get_media_detail.assert_not_called()

    def test_detections_tab_calls_get_detections_only(self, tmp_path) -> None:
        """Detections tab should call get_detections_for_media only."""
        client, mock_db, patch = self._get_client()
        mock_db.get_detections_for_media.return_value = []

        with patch("autopilot.web.routes.media.get_db", return_value=mock_db):
            resp = client.get("/media/test1/tab/detections")

        assert resp.status_code == 200
        mock_db.get_detections_for_media.assert_called_once()
        mock_db.get_transcript.assert_not_called()
        mock_db.get_faces_for_media.assert_not_called()
        mock_db.get_media_detail.assert_not_called()

    def test_faces_tab_calls_get_faces_only(self, tmp_path) -> None:
        """Faces tab should call get_faces_for_media (and face_clusters), not others."""
        client, mock_db, patch = self._get_client()
        mock_db.get_faces_for_media.return_value = []
        mock_db.get_face_clusters_by_ids.return_value = {}

        with patch("autopilot.web.routes.media.get_db", return_value=mock_db):
            resp = client.get("/media/test1/tab/faces")

        assert resp.status_code == 200
        mock_db.get_faces_for_media.assert_called_once()
        mock_db.get_transcript.assert_not_called()
        mock_db.get_detections_for_media.assert_not_called()
        mock_db.get_media_detail.assert_not_called()

    def test_audio_events_tab_calls_get_audio_events_only(self, tmp_path) -> None:
        """Audio events tab should call get_audio_events_for_media only."""
        client, mock_db, patch = self._get_client()
        mock_db.get_audio_events_for_media.return_value = []

        with patch("autopilot.web.routes.media.get_db", return_value=mock_db):
            resp = client.get("/media/test1/tab/audio_events")

        assert resp.status_code == 200
        mock_db.get_audio_events_for_media.assert_called_once()
        mock_db.get_transcript.assert_not_called()
        mock_db.get_detections_for_media.assert_not_called()
        mock_db.get_media_detail.assert_not_called()

    def test_assert_only_called_rejects_unexpected(self) -> None:
        """Meta-test: _assert_only_called raises when an un-allowed method was called."""
        from unittest.mock import MagicMock

        import pytest

        mock_db = MagicMock()
        mock_db.get_transcript()  # simulate an unexpected call
        with pytest.raises(AssertionError):
            self._assert_only_called(mock_db)  # no methods allowed → should raise

    def test_assert_only_called_permits_allowed(self) -> None:
        """Meta-test: _assert_only_called does not raise when the called method is allowed."""
        from unittest.mock import MagicMock

        mock_db = MagicMock()
        mock_db.get_transcript()  # simulate an expected call
        self._assert_only_called(mock_db, "get_transcript")  # should NOT raise

    def test_embeddings_tab_calls_count_embeddings_only(self, tmp_path) -> None:
        """Embeddings tab should call count_embeddings_for_media only."""
        client, mock_db, patch = self._get_client()
        mock_db.count_embeddings_for_media.return_value = 0

        with patch("autopilot.web.routes.media.get_db", return_value=mock_db):
            resp = client.get("/media/test1/tab/embeddings")

        assert resp.status_code == 200
        mock_db.count_embeddings_for_media.assert_called_once()
        mock_db.get_transcript.assert_not_called()
        mock_db.get_detections_for_media.assert_not_called()
        mock_db.get_media_detail.assert_not_called()
