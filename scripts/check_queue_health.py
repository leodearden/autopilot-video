"""Diagnostic utility for the fused-memory durable write queue.

Queries the write_queue SQLite database and emits a per-group health report
so that DLQ resurgences are caught quickly.

Usage examples::

    # Use the env var to point at the queue DB
    FUSED_MEMORY_QUEUE_DB=/path/to/write_queue.db python scripts/check_queue_health.py

    # Scope the report to a single project
    python scripts/check_queue_health.py --db-path /path/to/write_queue.db --project autopilot_video

    # Point at an alternate DB explicitly
    python scripts/check_queue_health.py --db-path /path/to/write_queue.db

Exit codes:
    0 — queue is healthy (no dead or retry rows)
    1 — queue is unhealthy
    2 — diagnostic failed (missing DB, missing write_queue table, no path configured, etc.)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Core helpers (pure functions, testable without argparse)
# ---------------------------------------------------------------------------


def _format_age(seconds: float) -> str:
    """Render a duration in seconds as a compact human-readable string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m"
    else:
        return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60)}m"


def collect_queue_stats(db_path: str | Path) -> dict[tuple[str, str], int]:
    """Query the write_queue DB and return per-(group_id, status) counts.

    Opens the database in read-only URI mode so this function never acquires
    a write lock against the live fused-memory server.

    Args:
        db_path: Path to the SQLite write_queue database file.

    Returns:
        Mapping of ``(group_id, status) -> count`` for every observed pair.
        Returns an empty dict when the table has no rows.
    """
    db_path = str(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT group_id, status, COUNT(*) FROM write_queue GROUP BY group_id, status"
        )
        return {(row[0], row[1]): row[2] for row in cursor.fetchall()}
    finally:
        conn.close()


def get_oldest_unhealthy_age_seconds(
    db_path: str | Path,
    *,
    now: float | None = None,
) -> float | None:
    """Return the age in seconds of the oldest unhealthy row (pending/retry/dead).

    Opens the database in read-only URI mode (never acquires a write lock).

    Args:
        db_path: Path to the SQLite write_queue database file.
        now: Override for the current time (seconds since epoch). Defaults to
            ``time.time()``. Useful for deterministic testing.

    Returns:
        ``(now - MIN(created_at))`` for rows with status in
        ``('pending', 'retry', 'dead')``, or ``None`` when no such rows exist.
    """
    db_path = str(db_path)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cursor = conn.execute(
            "SELECT MIN(created_at) FROM write_queue"
            " WHERE status IN ('pending', 'retry', 'dead')"
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None or row[0] is None:
        return None
    return (now if now is not None else time.time()) - row[0]


def summarize_health(
    stats: dict[tuple[str, str], int],
    *,
    oldest_unhealthy_age_seconds: float | None = None,
) -> dict[str, Any]:
    """Aggregate per-(group, status) counts into an overall health summary.

    Args:
        stats: Output of :func:`collect_queue_stats` (or a filtered subset).
        oldest_unhealthy_age_seconds: Age in seconds of the oldest unhealthy
            row, as returned by :func:`get_oldest_unhealthy_age_seconds`.
            Defaults to ``None`` (omitted from report when not provided).

    Returns:
        Dict with keys:
        - ``healthy`` (bool): True iff no dead or retry rows exist.
        - ``total_rows`` (int): Sum of all row counts.
        - ``completed`` (int): Rows in "completed" status.
        - ``dead`` (int): Rows in "dead" status.
        - ``retry`` (int): Rows in "retry" status.
        - ``pending`` (int): Rows in "pending" status.
        - ``oldest_unhealthy_age_seconds`` (float | None): Age of the oldest
          unhealthy row, or ``None`` if none exist.
        - ``unhealthy_groups`` (list[dict]): One entry per group that has any
          non-zero dead / retry / pending count, sorted by group_id.
          Each entry: ``{group_id, dead, retry, pending}``.
    """
    totals: dict[str, int] = {
        "completed": 0,
        "in_flight": 0,
        "dead": 0,
        "retry": 0,
        "pending": 0,
    }
    # Aggregate per-group counts so we can build unhealthy_groups
    group_counts: dict[str, dict[str, int]] = {}

    for (group_id, status), count in stats.items():
        # Update global totals
        if status in totals:
            totals[status] += count
        # Update per-group counters; in_flight is NOT tracked per-group —
        # global totals still aggregate it, but unhealthy_groups only needs
        # dead/retry/pending.
        g = group_counts.setdefault(group_id, {"dead": 0, "retry": 0, "pending": 0})
        if status in g:
            g[status] += count

    total_rows = sum(stats.values())
    dead = totals["dead"]
    retry = totals["retry"]
    healthy = dead == 0 and retry == 0

    unhealthy_groups: list[dict[str, Any]] = []
    for group_id in sorted(group_counts):
        g = group_counts[group_id]
        if g["dead"] > 0 or g["retry"] > 0:
            unhealthy_groups.append(
                {
                    "group_id": group_id,
                    "dead": g["dead"],
                    "retry": g["retry"],
                    "pending": g["pending"],
                }
            )

    return {
        "healthy": healthy,
        "total_rows": total_rows,
        "completed": totals["completed"],
        "in_flight": totals["in_flight"],
        "dead": dead,
        "retry": retry,
        "pending": totals["pending"],
        "oldest_unhealthy_age_seconds": oldest_unhealthy_age_seconds,
        "unhealthy_groups": unhealthy_groups,
    }


def filter_by_project(
    stats: dict[tuple[str, str], int], project_id: str
) -> dict[tuple[str, str], int]:
    """Return only the stats entries belonging to *project_id*.

    Matches exactly ``project_id`` and ``mem0_{project_id}`` — no
    substring or prefix matching so that e.g. "reify" never accidentally
    captures "mem0_autopilot_video".

    Args:
        stats: Output of :func:`collect_queue_stats`.
        project_id: The project name to filter on (e.g. "autopilot_video").

    Returns:
        Filtered dict containing only matching group_id keys.
    """
    mem0_key = f"mem0_{project_id}"
    return {
        (g, s): c
        for (g, s), c in stats.items()
        if g == project_id or g == mem0_key
    }


def format_report(
    stats: dict[tuple[str, str], int],
    summary: dict[str, Any],
) -> str:
    """Render a human-readable multi-line report.

    Args:
        stats: Output of :func:`collect_queue_stats` (possibly filtered).
        summary: Output of :func:`summarize_health`.

    Returns:
        Multi-line string suitable for printing to stdout.
    """
    lines: list[str] = []

    health_label = "HEALTHY" if summary["healthy"] else "UNHEALTHY"
    lines.append(f"HEALTH: {health_label}")
    summary_line = (
        f"total_rows={summary['total_rows']}"
        f" completed={summary['completed']}"
        f" in_flight={summary['in_flight']}"
        f" dead={summary['dead']}"
        f" retry={summary['retry']}"
        f" pending={summary['pending']}"
    )
    oldest = summary.get("oldest_unhealthy_age_seconds")
    if oldest is not None:
        summary_line += f" oldest_dead_age={_format_age(oldest)}"
    lines.append(summary_line)

    # Per-(group, status) table, sorted for stable output
    if stats:
        lines.append("")
        lines.append("Per-group breakdown:")
        for (group_id, status), count in sorted(stats.items()):
            lines.append(f"  {group_id:<40s}  {status:<12s}  {count}")

    # Unhealthy groups detail section
    if summary["unhealthy_groups"]:
        lines.append("")
        lines.append("Unhealthy groups:")
        for entry in summary["unhealthy_groups"]:
            lines.append(
                f"  {entry['group_id']}"
                f"  dead={entry['dead']}"
                f"  retry={entry['retry']}"
                f"  pending={entry['pending']}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run health check, print report, return exit code.

    Args:
        argv: Argument list (uses sys.argv when None).

    Returns:
        0 if the queue is healthy, 1 if unhealthy.
    """
    parser = argparse.ArgumentParser(
        description="Check the health of the fused-memory durable write queue."
    )
    parser.add_argument(
        "--db-path",
        default=None,
        help=(
            "Path to the write_queue SQLite database. "
            "Falls back to the FUSED_MEMORY_QUEUE_DB environment variable."
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Scope the report to a single project_id (e.g. autopilot_video).",
    )
    args = parser.parse_args(argv)

    db_path = args.db_path or os.environ.get("FUSED_MEMORY_QUEUE_DB")
    if not db_path:
        print(
            "error: --db-path is required (or set FUSED_MEMORY_QUEUE_DB env var)",
            file=sys.stderr,
        )
        return 2

    try:
        stats = collect_queue_stats(db_path)
        oldest_age = get_oldest_unhealthy_age_seconds(db_path)
    except (sqlite3.OperationalError, FileNotFoundError) as exc:
        print(
            f"error: failed to read queue DB at {db_path}: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.project:
        stats = filter_by_project(stats, args.project)
    summary = summarize_health(stats, oldest_unhealthy_age_seconds=oldest_age)
    print(format_report(stats, summary))
    return 0 if summary["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
