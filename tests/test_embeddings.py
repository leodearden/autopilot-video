"""Tests for frame embeddings and search index (autopilot.analyze.embeddings)."""

from __future__ import annotations

import struct
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDBHelpers:
    """Tests for CatalogDB clip embedding helper methods."""

    def test_get_all_clip_embeddings_empty(self, catalog_db) -> None:
        """get_all_clip_embeddings returns [] when clip_embeddings table is empty."""
        result = catalog_db.get_all_clip_embeddings()
        assert result == []

    def test_get_all_clip_embeddings_returns_all(self, catalog_db) -> None:
        """get_all_clip_embeddings returns all rows with non-null embeddings."""
        # Insert a media file first (FK constraint)
        catalog_db.insert_media("vid1", "/v1.mp4")
        blob1 = struct.pack("f" * 768, *([0.1] * 768))
        blob2 = struct.pack("f" * 768, *([0.2] * 768))
        catalog_db.batch_insert_embeddings([
            ("vid1", 0, blob1),
            ("vid1", 60, blob2),
        ])
        result = catalog_db.get_all_clip_embeddings()
        assert len(result) == 2
        assert result[0]["media_id"] == "vid1"
        assert result[0]["frame_number"] == 0
        assert result[0]["embedding"] == blob1
        assert result[1]["frame_number"] == 60

    def test_get_all_clip_embeddings_multiple_media(self, catalog_db) -> None:
        """get_all_clip_embeddings returns rows across multiple media files."""
        for mid in ("vid1", "vid2", "vid3"):
            catalog_db.insert_media(mid, f"/{mid}.mp4")
        blob = struct.pack("f" * 768, *([0.5] * 768))
        catalog_db.batch_insert_embeddings([
            ("vid1", 0, blob),
            ("vid2", 0, blob),
            ("vid2", 60, blob),
            ("vid3", 0, blob),
        ])
        result = catalog_db.get_all_clip_embeddings()
        assert len(result) == 4
        media_ids = {r["media_id"] for r in result}
        assert media_ids == {"vid1", "vid2", "vid3"}
