"""Deduplication — SHA-256 hashing and duplicate detection in the catalog."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autopilot.db import CatalogDB

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


def find_duplicates(catalog_db: CatalogDB) -> list[tuple[str, str]]:
    """Find duplicate media files by sha256_prefix.

    Returns a list of ``(kept_id, duplicate_id)`` tuples.  The first ID in each
    group (by rowid order — i.e. first ingested) is kept; subsequent ones are
    duplicates.  Groups of 3+ produce multiple pairs: ``(kept, dup1)``,
    ``(kept, dup2)``, etc.
    """
    cur = catalog_db.conn.execute(
        "SELECT sha256_prefix, GROUP_CONCAT(id) "
        "FROM media_files "
        "WHERE sha256_prefix IS NOT NULL "
        "GROUP BY sha256_prefix "
        "HAVING COUNT(*) > 1"
    )
    pairs: list[tuple[str, str]] = []
    for row in cur.fetchall():
        ids = row[1].split(",")
        kept = ids[0]
        for dup in ids[1:]:
            pairs.append((kept, dup))
    return pairs
