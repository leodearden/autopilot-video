"""Tests for scripts/check_queue_health.py — fused-memory write queue diagnostic."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# Ensure scripts/ directory is on sys.path so we can import check_queue_health
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# isort: split
from check_queue_health import (  # noqa: E402  (import after sys.path mutation)
    collect_queue_stats,
    filter_by_project,
    format_report,
    main,
    summarize_health,
)

# ---------------------------------------------------------------------------
# Helper: create a minimal write_queue DB in a tmp_path file
# ---------------------------------------------------------------------------

def _make_queue_db(tmp_path: Path) -> Path:
    """Create a minimal write_queue SQLite DB mirroring the production schema.

    Returns the path to the created DB file.
    """
    db_path = tmp_path / "write_queue.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE write_queue (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id       TEXT    NOT NULL,
            operation      TEXT    NOT NULL,
            payload        TEXT,
            callback_type  TEXT,
            status         TEXT    NOT NULL,
            attempts       INTEGER NOT NULL DEFAULT 0,
            max_attempts   INTEGER NOT NULL DEFAULT 5,
            next_retry_at  REAL,
            created_at     REAL    NOT NULL,
            completed_at   REAL,
            error          TEXT
        );
        CREATE INDEX idx_wq_status_group ON write_queue (status, group_id, next_retry_at);
        """
    )
    conn.commit()
    conn.close()
    return db_path


def _insert_row(
    conn: sqlite3.Connection,
    *,
    group_id: str,
    operation: str = "add_memory_graphiti",
    status: str = "completed",
    attempts: int = 1,
    created_at: float = 1.0,
    completed_at: float | None = 2.0,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO write_queue
            (group_id, operation, status, attempts, created_at, completed_at, error)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (group_id, operation, status, attempts, created_at, completed_at, error),
    )


# ===========================================================================
# TestCollectQueueStats
# ===========================================================================


class TestCollectQueueStats:
    def test_returns_counts_grouped_by_group_and_status(self, tmp_path: Path) -> None:
        db_path = _make_queue_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            _insert_row(conn, group_id="autopilot_video", status="completed")
            _insert_row(conn, group_id="autopilot_video", status="completed")
            _insert_row(conn, group_id="mem0_autopilot_video", status="completed")
            _insert_row(conn, group_id="dark_factory", status="dead")
            conn.commit()
        finally:
            conn.close()

        result = collect_queue_stats(db_path)

        assert result == {
            ("autopilot_video", "completed"): 2,
            ("mem0_autopilot_video", "completed"): 1,
            ("dark_factory", "dead"): 1,
        }

    def test_empty_db_returns_empty_dict(self, tmp_path: Path) -> None:
        db_path = _make_queue_db(tmp_path)
        result = collect_queue_stats(db_path)
        assert result == {}

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        db_path = _make_queue_db(tmp_path)
        # Should work with a Path object, not just str
        result = collect_queue_stats(db_path)
        assert isinstance(result, dict)


# ===========================================================================
# TestSummarizeHealth
# ===========================================================================


class TestSummarizeHealth:
    def test_all_completed_is_healthy(self) -> None:
        stats = {
            ("autopilot_video", "completed"): 5,
            ("mem0_autopilot_video", "completed"): 3,
        }
        summary = summarize_health(stats)

        assert summary["healthy"] is True
        assert summary["total_rows"] == 8
        assert summary["dead"] == 0
        assert summary["retry"] == 0
        assert summary["pending"] == 0
        assert summary["unhealthy_groups"] == []

    def test_empty_stats_is_healthy(self) -> None:
        summary = summarize_health({})

        assert summary["healthy"] is True
        assert summary["total_rows"] == 0
        assert summary["dead"] == 0
        assert summary["retry"] == 0
        assert summary["pending"] == 0
        assert summary["unhealthy_groups"] == []

    def test_dead_letters_flag_unhealthy(self) -> None:
        stats = {
            ("autopilot_video", "completed"): 10,
            ("autopilot_video", "dead"): 3,
            ("mem0_autopilot_video", "retry"): 2,
        }
        summary = summarize_health(stats)

        assert summary["healthy"] is False
        assert summary["dead"] == 3
        assert summary["retry"] == 2

        # unhealthy_groups sorted by group_id
        assert summary["unhealthy_groups"] == [
            {"group_id": "autopilot_video", "dead": 3, "retry": 0, "pending": 0},
            {"group_id": "mem0_autopilot_video", "dead": 0, "retry": 2, "pending": 0},
        ]

    def test_pending_alone_is_healthy(self) -> None:
        """Pending rows are in-flight work — not an error condition."""
        stats = {
            ("autopilot_video", "pending"): 5,
            ("autopilot_video", "completed"): 10,
        }
        summary = summarize_health(stats)

        assert summary["healthy"] is True
        assert summary["pending"] == 5
        assert summary["unhealthy_groups"] == []

    def test_in_flight_rows_counted_and_arithmetic_consistent(self) -> None:
        """in_flight must appear in summary and the arithmetic invariant must hold."""
        stats = {
            ("autopilot_video", "completed"): 10,
            ("mem0_reify", "in_flight"): 1,
            ("autopilot_video", "dead"): 2,
            ("autopilot_video", "retry"): 1,
            ("autopilot_video", "pending"): 3,
        }
        summary = summarize_health(stats)

        assert summary["in_flight"] == 1
        assert summary["total_rows"] == 17
        # The arithmetic invariant: all buckets must sum to total_rows
        assert (
            summary["completed"]
            + summary["in_flight"]
            + summary["dead"]
            + summary["retry"]
            + summary["pending"]
            == summary["total_rows"]
        )

    def test_in_flight_alone_does_not_flag_unhealthy(self) -> None:
        """in_flight is transient worker activity — must NOT mark queue unhealthy."""
        stats = {
            ("autopilot_video", "completed"): 1636,
            ("mem0_reify", "in_flight"): 1,
        }
        summary = summarize_health(stats)

        assert summary["healthy"] is True
        assert summary["in_flight"] == 1
        assert summary["dead"] == 0
        assert summary["retry"] == 0
        assert summary["unhealthy_groups"] == []


# ===========================================================================
# TestFilterByProject
# ===========================================================================


class TestFilterByProject:
    def _full_stats(self) -> dict[tuple[str, str], int]:
        return {
            ("autopilot_video", "completed"): 5,
            ("mem0_autopilot_video", "completed"): 3,
            ("dark_factory", "completed"): 10,
            ("mem0_dark_factory", "dead"): 2,
            ("mem0_reify", "completed"): 7,
        }

    def test_includes_base_and_mem0_groups(self) -> None:
        stats = self._full_stats()
        filtered = filter_by_project(stats, "autopilot_video")
        assert set(filtered.keys()) == {
            ("autopilot_video", "completed"),
            ("mem0_autopilot_video", "completed"),
        }
        assert filtered[("autopilot_video", "completed")] == 5
        assert filtered[("mem0_autopilot_video", "completed")] == 3

    def test_excludes_other_projects(self) -> None:
        stats = self._full_stats()
        filtered = filter_by_project(stats, "autopilot_video")
        # dark_factory and mem0_reify must not appear
        for key in filtered:
            assert key[0] in ("autopilot_video", "mem0_autopilot_video")

    def test_exact_match_not_substring(self) -> None:
        """Filtering 'reify' must NOT return mem0_autopilot_video."""
        stats = {
            ("reify", "completed"): 5,
            ("mem0_reify", "completed"): 3,
            ("mem0_autopilot_video", "completed"): 7,  # contains "video", not "reify"
        }
        filtered = filter_by_project(stats, "reify")
        assert set(filtered.keys()) == {
            ("reify", "completed"),
            ("mem0_reify", "completed"),
        }

    def test_empty_stats_returns_empty(self) -> None:
        assert filter_by_project({}, "autopilot_video") == {}


# ===========================================================================
# TestFormatReport
# ===========================================================================


class TestFormatReport:
    def test_report_includes_totals_and_per_group_lines(self) -> None:
        stats = {
            ("autopilot_video", "completed"): 10,
            ("autopilot_video", "dead"): 1,
            ("mem0_autopilot_video", "retry"): 2,
        }
        summary = summarize_health(stats)
        report = format_report(stats, summary)

        assert "HEALTH: UNHEALTHY" in report
        assert "autopilot_video" in report
        assert "mem0_autopilot_video" in report
        assert "dead" in report
        assert "retry" in report
        assert "Unhealthy groups:" in report

    def test_healthy_report_says_healthy(self) -> None:
        stats = {
            ("autopilot_video", "completed"): 8,
            ("mem0_autopilot_video", "completed"): 4,
        }
        summary = summarize_health(stats)
        report = format_report(stats, summary)

        assert "HEALTH: HEALTHY" in report
        assert "dead=0" in report

    def test_healthy_report_has_no_unhealthy_groups_section(self) -> None:
        stats = {("autopilot_video", "completed"): 3}
        summary = summarize_health(stats)
        report = format_report(stats, summary)
        assert "Unhealthy groups:" not in report

    def test_summary_line_includes_in_flight_count(self) -> None:
        """format_report summary line must emit in_flight=N so arithmetic is visible."""
        stats = {
            ("autopilot_video", "completed"): 5,
            ("mem0_reify", "in_flight"): 2,
        }
        summary = summarize_health(stats)
        report = format_report(stats, summary)

        assert "in_flight=2" in report
        assert "total_rows=7" in report
        # All five fields present in the summary line
        lines = report.splitlines()
        summary_line = lines[1]  # second line after the HEALTH: header
        assert "completed=5" in summary_line
        assert "in_flight=2" in summary_line
        assert "dead=0" in summary_line
        assert "retry=0" in summary_line
        assert "pending=0" in summary_line

    def test_summary_line_arithmetic_sums_to_total_rows(self) -> None:
        """Parsing the summary line must yield counts that sum to total_rows."""
        import re

        stats = {
            ("autopilot_video", "completed"): 10,
            ("mem0_reify", "in_flight"): 3,
            ("dark_factory", "dead"): 2,
            ("dark_factory", "retry"): 1,
            ("autopilot_video", "pending"): 4,
        }
        summary = summarize_health(stats)
        report = format_report(stats, summary)

        summary_line = report.splitlines()[1]
        counts = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", summary_line)}
        total = counts.pop("total_rows")
        assert sum(counts.values()) == total


# ===========================================================================
# TestMain
# ===========================================================================


class TestMain:
    def test_main_healthy_exits_zero_and_prints_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        db_path = _make_queue_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            _insert_row(conn, group_id="autopilot_video", status="completed")
            conn.commit()
        finally:
            conn.close()

        rc = main(["--db-path", str(db_path)])
        captured = capsys.readouterr()

        assert rc == 0
        assert "HEALTH: HEALTHY" in captured.out

    def test_main_unhealthy_exits_nonzero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        db_path = _make_queue_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            _insert_row(conn, group_id="autopilot_video", status="dead")
            conn.commit()
        finally:
            conn.close()

        rc = main(["--db-path", str(db_path)])
        captured = capsys.readouterr()

        assert rc == 1
        assert "HEALTH: UNHEALTHY" in captured.out
        assert "autopilot_video" in captured.out

    def test_main_filter_scopes_to_project(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        db_path = _make_queue_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            _insert_row(conn, group_id="autopilot_video", status="completed")
            _insert_row(conn, group_id="dark_factory", status="completed")
            conn.commit()
        finally:
            conn.close()

        main(["--db-path", str(db_path), "--project", "autopilot_video"])
        captured = capsys.readouterr()

        assert "dark_factory" not in captured.out

    def test_main_defaults_db_path_to_dark_factory_queue(self) -> None:
        """The default --db-path should point at the dark-factory queue."""
        import argparse

        # Reconstruct just the parser to check the default
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--db-path",
            default="/home/leo/src/dark-factory/data/queue/write_queue.db",
        )
        parser.add_argument("--project", default=None)
        ns = parser.parse_args([])
        assert ns.db_path == "/home/leo/src/dark-factory/data/queue/write_queue.db"
