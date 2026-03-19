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


class TestPublicAPI:
    """Tests for module-level public API imports."""

    def test_embedding_error_importable(self) -> None:
        """EmbeddingError is importable and is an Exception subclass with message."""
        from autopilot.analyze.embeddings import EmbeddingError

        assert issubclass(EmbeddingError, Exception)
        err = EmbeddingError("test message")
        assert str(err) == "test message"

    def test_compute_embeddings_importable(self) -> None:
        """compute_embeddings is importable from autopilot.analyze.embeddings."""
        from autopilot.analyze.embeddings import compute_embeddings

        assert callable(compute_embeddings)

    def test_build_search_index_importable(self) -> None:
        """build_search_index is importable from autopilot.analyze.embeddings."""
        from autopilot.analyze.embeddings import build_search_index

        assert callable(build_search_index)

    def test_search_by_text_importable(self) -> None:
        """search_by_text is importable from autopilot.analyze.embeddings."""
        from autopilot.analyze.embeddings import search_by_text

        assert callable(search_by_text)

    def test_search_by_image_importable(self) -> None:
        """search_by_image is importable from autopilot.analyze.embeddings."""
        from autopilot.analyze.embeddings import search_by_image

        assert callable(search_by_image)

    def test_all_exports(self) -> None:
        """__all__ contains exactly the expected public API."""
        from autopilot.analyze import embeddings

        expected = {
            "EmbeddingError",
            "compute_embeddings",
            "build_search_index",
            "search_by_text",
            "search_by_image",
        }
        assert set(embeddings.__all__) == expected
