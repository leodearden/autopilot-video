"""Deduplication — SHA-256 hashing and duplicate detection in the catalog."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Only hash the first 64 MB of each file for speed on large video files.
HASH_SIZE: int = 64 * 1024 * 1024  # 67,108,864 bytes
_CHUNK_SIZE: int = 8 * 1024  # 8 KB read chunks


def compute_hash(file_path: Path) -> str:
    """Compute SHA-256 hex digest of the first 64 MB of *file_path*.

    Reads in 8 KB chunks to keep memory usage constant regardless of file size.
    """
    h = hashlib.sha256()
    bytes_read = 0
    with open(file_path, "rb") as f:
        while bytes_read < HASH_SIZE:
            remaining = HASH_SIZE - bytes_read
            chunk = f.read(min(_CHUNK_SIZE, remaining))
            if not chunk:
                break
            h.update(chunk)
            bytes_read += len(chunk)
    return h.hexdigest()
