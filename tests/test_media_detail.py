"""Tests for the media detail page, tab endpoints, and detail API."""

from __future__ import annotations

import json
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
def detail_db(detail_db_path: str) -> CatalogDB:
    """Create a CatalogDB backed by a real file for web endpoint tests."""
    db = CatalogDB(detail_db_path)
    db.conn.isolation_level = None  # autocommit
    return db


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
    db.batch_insert_detections([
        ("test1", 0, json.dumps(det_frame0)),
        ("test1", 30, json.dumps(det_frame1)),
        ("test1", 60, json.dumps(det_frame2)),
    ])

    # 2 faces on separate frames, assigned to cluster_id=1
    db.batch_insert_faces([
        ("test1", 0, 0, json.dumps({"x": 10, "y": 20, "w": 50, "h": 50}), None, 1),
        ("test1", 30, 0, json.dumps({"x": 15, "y": 25, "w": 55, "h": 55}), None, 1),
    ])

    # Face cluster with label 'Alice'
    db.insert_face_cluster(1, label="Alice")

    # 2 audio events with classified events_json
    audio_events_0 = [
        {"class": "Speech", "confidence": 0.95},
        {"class": "Music", "confidence": 0.3},
    ]
    audio_events_1 = [
        {"class": "Ocean waves", "confidence": 0.88},
    ]
    db.batch_insert_audio_events([
        ("test1", 5.0, json.dumps(audio_events_0)),
        ("test1", 60.0, json.dumps(audio_events_1)),
    ])

    # 3 clip embeddings
    db.batch_insert_embeddings([
        ("test1", 0, b"\x00" * 16),
        ("test1", 30, b"\x01" * 16),
        ("test1", 60, b"\x02" * 16),
    ])

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
