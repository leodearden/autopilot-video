"""Tests for autopilot.db CatalogDB class."""

from __future__ import annotations

import sqlite3
import threading

import pytest


# -- CatalogDB class basics --------------------------------------------------


class TestCatalogDBBasics:
    """Tests for CatalogDB connection management and lifecycle."""

    def test_catalogdb_creates_connection(self, catalog_db):
        """CatalogDB with ':memory:' creates a valid connection."""
        assert catalog_db.conn is not None
        assert isinstance(catalog_db.conn, sqlite3.Connection)

    def test_catalogdb_wal_mode(self, tmp_path):
        """CatalogDB on a file-based DB enables WAL journal mode."""
        from autopilot.db import CatalogDB

        db_path = str(tmp_path / "test.db")
        db = CatalogDB(db_path)
        try:
            cur = db.conn.execute("PRAGMA journal_mode")
            mode = cur.fetchone()[0]
            assert mode == "wal", f"Expected WAL mode, got {mode}"
        finally:
            db.close()

    def test_catalogdb_thread_safety(self, catalog_db):
        """CatalogDB connection works from a different thread."""
        results: list[bool] = []

        def access_db():
            try:
                catalog_db.conn.execute("SELECT 1")
                results.append(True)
            except sqlite3.ProgrammingError:
                results.append(False)

        t = threading.Thread(target=access_db)
        t.start()
        t.join()
        assert results == [True], "Connection should be usable from another thread"

    def test_catalogdb_context_manager(self):
        """Context manager __enter__ returns self, __exit__ commits."""
        from autopilot.db import CatalogDB

        db = CatalogDB(":memory:")
        try:
            with db as ctx:
                assert ctx is db
                # Create a temp table and insert inside context manager
                ctx.conn.execute(
                    "CREATE TABLE IF NOT EXISTS _test (val TEXT)"
                )
                ctx.conn.execute("INSERT INTO _test VALUES ('hello')")
            # After __exit__, data should be committed
            cur = db.conn.execute("SELECT val FROM _test")
            assert cur.fetchone()[0] == "hello"
        finally:
            db.close()

    def test_catalogdb_close(self, catalog_db):
        """close() closes the underlying connection."""
        catalog_db.close()
        with pytest.raises(sqlite3.ProgrammingError):
            catalog_db.conn.execute("SELECT 1")
