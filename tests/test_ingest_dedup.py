"""Tests for autopilot.ingest.dedup — SHA-256 hashing and duplicate detection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autopilot.ingest.dedup import compute_hash, find_duplicates, mark_duplicates


class TestComputeHash:
    """Tests for compute_hash SHA-256 first-64MB hashing."""

    def test_hash_small_file(self, tmp_path: Path) -> None:
        """compute_hash of a file < 64MB should hash the full content."""
        content = b"hello world" * 1000
        f = tmp_path / "small.mp4"
        f.write_bytes(content)

        expected = hashlib.sha256(content).hexdigest()
        assert compute_hash(f) == expected

    def test_hash_large_file(self, tmp_path: Path) -> None:
        """compute_hash of a file > 64MB should hash only the first 64MB."""
        chunk_64mb = b"\xAB" * (64 * 1024 * 1024)  # exactly 64MB
        extra = b"\xCD" * (1024 * 1024)  # 1MB extra
        f = tmp_path / "large.mp4"
        f.write_bytes(chunk_64mb + extra)

        expected = hashlib.sha256(chunk_64mb).hexdigest()
        assert compute_hash(f) == expected

    def test_hash_empty_file(self, tmp_path: Path) -> None:
        """compute_hash of an empty file should return SHA-256 of empty bytes."""
        f = tmp_path / "empty.mp4"
        f.write_bytes(b"")

        expected = hashlib.sha256(b"").hexdigest()
        assert compute_hash(f) == expected

    def test_hash_deterministic(self, tmp_path: Path) -> None:
        """compute_hash called twice on the same file should return the same result."""
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"deterministic content")

        h1 = compute_hash(f)
        h2 = compute_hash(f)
        assert h1 == h2

    def test_hash_different_files(self, tmp_path: Path) -> None:
        """compute_hash should return different hashes for different file contents."""
        f1 = tmp_path / "clip1.mp4"
        f1.write_bytes(b"content A")
        f2 = tmp_path / "clip2.mp4"
        f2.write_bytes(b"content B")

        assert compute_hash(f1) != compute_hash(f2)


class TestFindDuplicates:
    """Tests for find_duplicates using the catalog_db fixture."""

    def test_find_duplicates_with_matches(self, catalog_db) -> None:  # type: ignore[no-untyped-def]
        """find_duplicates should return pairs for files with matching sha256_prefix."""
        catalog_db.insert_media("m1", "/a.mp4", sha256_prefix="abc123")
        catalog_db.insert_media("m2", "/b.mp4", sha256_prefix="abc123")
        catalog_db.insert_media("m3", "/c.mp4", sha256_prefix="def456")

        pairs = find_duplicates(catalog_db)
        # m1 kept, m2 is duplicate
        assert len(pairs) == 1
        kept, dup = pairs[0]
        assert kept == "m1"
        assert dup == "m2"

    def test_find_duplicates_no_matches(self, catalog_db) -> None:  # type: ignore[no-untyped-def]
        """find_duplicates should return empty list when all hashes are unique."""
        catalog_db.insert_media("m1", "/a.mp4", sha256_prefix="aaa")
        catalog_db.insert_media("m2", "/b.mp4", sha256_prefix="bbb")
        catalog_db.insert_media("m3", "/c.mp4", sha256_prefix="ccc")

        pairs = find_duplicates(catalog_db)
        assert pairs == []

    def test_find_duplicates_null_hash(self, catalog_db) -> None:  # type: ignore[no-untyped-def]
        """find_duplicates should exclude files with sha256_prefix=None."""
        catalog_db.insert_media("m1", "/a.mp4", sha256_prefix=None)
        catalog_db.insert_media("m2", "/b.mp4", sha256_prefix=None)
        catalog_db.insert_media("m3", "/c.mp4", sha256_prefix="abc123")

        pairs = find_duplicates(catalog_db)
        assert pairs == []

    def test_find_duplicates_triple(self, catalog_db) -> None:  # type: ignore[no-untyped-def]
        """find_duplicates with 3 files sharing a hash should return 2 pairs."""
        catalog_db.insert_media("m1", "/a.mp4", sha256_prefix="abc123")
        catalog_db.insert_media("m2", "/b.mp4", sha256_prefix="abc123")
        catalog_db.insert_media("m3", "/c.mp4", sha256_prefix="abc123")

        pairs = find_duplicates(catalog_db)
        # m1 kept, m2 and m3 are duplicates
        assert len(pairs) == 2
        assert ("m1", "m2") in pairs
        assert ("m1", "m3") in pairs


class TestMarkDuplicates:
    """Tests for mark_duplicates status updates."""

    def test_mark_duplicates_updates_status(self, catalog_db) -> None:  # type: ignore[no-untyped-def]
        """mark_duplicates should set status='duplicate' on duplicate files."""
        catalog_db.insert_media("m1", "/a.mp4", sha256_prefix="abc123")
        catalog_db.insert_media("m2", "/b.mp4", sha256_prefix="abc123")
        catalog_db.insert_media("m3", "/c.mp4", sha256_prefix="def456")

        count = mark_duplicates(catalog_db)
        assert count == 1

        m1 = catalog_db.get_media("m1")
        m2 = catalog_db.get_media("m2")
        m3 = catalog_db.get_media("m3")
        assert m1 is not None and m1["status"] == "ingested"
        assert m2 is not None and m2["status"] == "duplicate"
        assert m3 is not None and m3["status"] == "ingested"

    def test_mark_duplicates_no_duplicates(self, catalog_db) -> None:  # type: ignore[no-untyped-def]
        """mark_duplicates should return 0 when no duplicates exist."""
        catalog_db.insert_media("m1", "/a.mp4", sha256_prefix="aaa")
        catalog_db.insert_media("m2", "/b.mp4", sha256_prefix="bbb")

        count = mark_duplicates(catalog_db)
        assert count == 0

        m1 = catalog_db.get_media("m1")
        m2 = catalog_db.get_media("m2")
        assert m1 is not None and m1["status"] == "ingested"
        assert m2 is not None and m2["status"] == "ingested"

    def test_mark_duplicates_preserves_non_duplicate_status(self, catalog_db) -> None:  # type: ignore[no-untyped-def]
        """mark_duplicates should not change status of non-duplicate files."""
        catalog_db.insert_media("m1", "/a.mp4", sha256_prefix="unique1", status="analyzing")
        catalog_db.insert_media("m2", "/b.mp4", sha256_prefix="unique2")

        mark_duplicates(catalog_db)

        m1 = catalog_db.get_media("m1")
        assert m1 is not None and m1["status"] == "analyzing"
