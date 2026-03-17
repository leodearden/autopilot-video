"""Tests for autopilot.ingest.dedup — SHA-256 hashing and duplicate detection."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autopilot.ingest.dedup import compute_hash


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
