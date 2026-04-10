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
    _format_age,
    collect_queue_stats,
    filter_by_project,
    format_report,
    get_oldest_unhealthy_age_seconds,
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
# TestFormatAge
# ===========================================================================


class TestFormatAge:
    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (0, "0s"),
            (45, "45s"),
            (120, "2m"),
            (3600, "1h0m"),
            (3 * 3600 + 42 * 60, "3h42m"),
            (86400, "24h0m"),
        ],
    )
    def test_format_age(self, seconds: float, expected: str) -> None:
        assert _format_age(seconds) == expected


# ===========================================================================
# TestGetOldestUnhealthyAgeSeconds
# ===========================================================================


class TestGetOldestUnhealthyAgeSeconds:
    def test_returns_age_from_min_created_at_across_pending_retry_dead(
        self, tmp_path: Path
    ) -> None:
        db_path = _make_queue_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            _insert_row(conn, group_id="g", status="pending", created_at=100.0)
            _insert_row(conn, group_id="g", status="retry", created_at=200.0)
            _insert_row(conn, group_id="g", status="dead", created_at=150.0)
            _insert_row(conn, group_id="g", status="completed", created_at=50.0)
            conn.commit()
        finally:
            conn.close()

        result = get_oldest_unhealthy_age_seconds(db_path, now=1000.0)
        # min(100, 200, 150) = 100; completed@50 must not be included
        assert result == 900.0

    def test_returns_none_when_no_unhealthy_rows(self, tmp_path: Path) -> None:
        db_path = _make_queue_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            _insert_row(conn, group_id="g", status="completed", created_at=1.0)
            conn.commit()
        finally:
            conn.close()

        result = get_oldest_unhealthy_age_seconds(db_path, now=1000.0)
        assert result is None

    def test_ignores_completed_rows_in_min_computation(
        self, tmp_path: Path
    ) -> None:
        db_path = _make_queue_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            _insert_row(conn, group_id="g", status="completed", created_at=1.0)
            _insert_row(conn, group_id="g", status="pending", created_at=900.0)
            conn.commit()
        finally:
            conn.close()

        result = get_oldest_unhealthy_age_seconds(db_path, now=1000.0)
        assert result == 100.0  # 1000 - 900, not 1000 - 1


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

    def test_unhealthy_groups_entries_have_exact_key_set(self) -> None:
        """Each unhealthy_groups entry must have exactly {group_id, dead, retry, pending}."""
        stats = {
            ("g1", "dead"): 2,
            ("g1", "retry"): 1,
            ("g1", "in_flight"): 3,
            ("g1", "pending"): 4,
            ("g2", "dead"): 1,
        }
        summary = summarize_health(stats)

        for entry in summary["unhealthy_groups"]:
            assert set(entry.keys()) == {"group_id", "dead", "retry", "pending"}

    def test_surfaces_oldest_unhealthy_age_seconds_from_kwarg(self) -> None:
        stats = {("g", "dead"): 1}
        summary = summarize_health(stats, oldest_unhealthy_age_seconds=123.4)
        assert summary["oldest_unhealthy_age_seconds"] == 123.4

    def test_oldest_unhealthy_age_seconds_defaults_to_none(self) -> None:
        summary = summarize_health({})
        assert summary["oldest_unhealthy_age_seconds"] is None


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

    def test_summary_line_includes_oldest_dead_age_when_present(self) -> None:
        stats = {("g", "dead"): 1}
        summary = summarize_health(
            stats, oldest_unhealthy_age_seconds=3 * 3600 + 42 * 60
        )
        report = format_report(stats, summary)

        summary_line = report.splitlines()[1]
        assert "oldest_dead_age=3h42m" in summary_line

    def test_summary_line_omits_oldest_dead_age_when_none(self) -> None:
        stats = {("g", "completed"): 1}
        summary = summarize_health(stats)  # oldest_unhealthy_age_seconds=None
        report = format_report(stats, summary)

        assert "oldest_dead_age" not in report


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

    def test_default_db_path_does_not_contain_home_leo(self) -> None:
        """Portability guard: the script must not hardcode /home/leo/src/dark-factory."""
        import check_queue_health as cqh

        # Check the script source for the cross-project absolute path
        source_path = Path(cqh.__file__)
        source_text = source_path.read_text()
        assert "/home/leo/src/dark-factory" not in source_text, (
            "check_queue_health.py contains a hardcoded cross-project path; "
            "use FUSED_MEMORY_QUEUE_DB env var or --db-path instead"
        )

    def test_main_exits_with_code_2_on_missing_db_file(
        self, capsys: pytest.CaptureFixture
    ) -> None:
        rc = main(["--db-path", "/nonexistent/path.db"])
        captured = capsys.readouterr()

        assert rc == 2
        assert "/nonexistent/path.db" in captured.err
        # Report banner must NOT appear on diagnostic failure
        assert "HEALTH:" not in captured.out

    def test_main_exits_with_code_2_on_missing_write_queue_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        # Create a valid SQLite file but without the write_queue table
        db_path = tmp_path / "empty.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()

        rc = main(["--db-path", str(db_path)])
        captured = capsys.readouterr()

        assert rc == 2
        assert str(db_path) in captured.err
        assert "HEALTH:" not in captured.out

    def test_default_db_path_resolves_from_fused_memory_queue_db_env_var(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        db_path = _make_queue_db(tmp_path)
        monkeypatch.setenv("FUSED_MEMORY_QUEUE_DB", str(db_path))

        # No --db-path flag — should fall back to env var
        rc = main([])
        captured = capsys.readouterr()

        # 0 (healthy) or 1 (unhealthy) — NOT 2 (diagnostic failure)
        assert rc in (0, 1)
        assert "HEALTH:" in captured.out

    def test_main_exits_with_code_2_when_no_db_path_and_no_env_var(
        self,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("FUSED_MEMORY_QUEUE_DB", raising=False)

        rc = main([])
        captured = capsys.readouterr()

        assert rc == 2
        assert "--db-path" in captured.err or "FUSED_MEMORY_QUEUE_DB" in captured.err

    def test_main_integrates_oldest_dead_age_in_report(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        import re
        import time

        db_path = _make_queue_db(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            _insert_row(
                conn,
                group_id="g",
                status="dead",
                created_at=time.time() - 7200,  # 2 hours ago
            )
            conn.commit()
        finally:
            conn.close()

        rc = main(["--db-path", str(db_path)])
        captured = capsys.readouterr()

        assert rc == 1
        assert re.search(r"oldest_dead_age=\d+h\d+m", captured.out)
