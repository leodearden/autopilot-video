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

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()
