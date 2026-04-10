"""Diagnostic utility for the fused-memory durable write queue.

Queries the write_queue SQLite database and emits a per-group health report
so that DLQ resurgences are caught quickly.

Usage examples::

    # Check the default dark-factory queue
    python scripts/check_queue_health.py

    # Scope the report to a single project
    python scripts/check_queue_health.py --project autopilot_video

    # Point at an alternate DB
    python scripts/check_queue_health.py --db-path /path/to/write_queue.db

Exit codes:
    0 — queue is healthy (no dead or retry rows)
    1 — queue is unhealthy
    2 — diagnostic failed (missing DB, missing write_queue table, etc.)
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

_DEFAULT_DB_PATH = "/home/leo/src/dark-factory/data/queue/write_queue.db"


# ---------------------------------------------------------------------------
# Core helpers (pure functions, testable without argparse)
# ---------------------------------------------------------------------------


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


def summarize_health(stats: dict[tuple[str, str], int]) -> dict[str, Any]:
    """Aggregate per-(group, status) counts into an overall health summary.

    Args:
        stats: Output of :func:`collect_queue_stats` (or a filtered subset).

    Returns:
        Dict with keys:
        - ``healthy`` (bool): True iff no dead or retry rows exist.
        - ``total_rows`` (int): Sum of all row counts.
        - ``completed`` (int): Rows in "completed" status.
        - ``dead`` (int): Rows in "dead" status.
        - ``retry`` (int): Rows in "retry" status.
        - ``pending`` (int): Rows in "pending" status.
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
        # Update per-group counters (in_flight tracked per-group for internal
        # bookkeeping but NOT surfaced in unhealthy_groups entries)
        g = group_counts.setdefault(
            group_id, {"in_flight": 0, "dead": 0, "retry": 0, "pending": 0}
        )
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
    lines.append(
        f"total_rows={summary['total_rows']}"
        f" completed={summary['completed']}"
        f" in_flight={summary['in_flight']}"
        f" dead={summary['dead']}"
        f" retry={summary['retry']}"
        f" pending={summary['pending']}"
    )

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
        default=_DEFAULT_DB_PATH,
        help=(
            "Path to the write_queue SQLite database "
            f"(default: {_DEFAULT_DB_PATH})"
        ),
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Scope the report to a single project_id (e.g. autopilot_video).",
    )
    args = parser.parse_args(argv)

    try:
        stats = collect_queue_stats(args.db_path)
    except (sqlite3.OperationalError, FileNotFoundError) as exc:
        print(
            f"error: failed to read queue DB at {args.db_path}: {exc}",
            file=sys.stderr,
        )
        return 2
    if args.project:
        stats = filter_by_project(stats, args.project)
    summary = summarize_health(stats)
    print(format_report(stats, summary))
    return 0 if summary["healthy"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
