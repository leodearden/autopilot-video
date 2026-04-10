"""Schema integrity tests for .taskmaster/tasks/tasks.json.

Regression guard for the int→str migration of task IDs.  All task IDs and
dependency references must remain strings; every dependency reference must
resolve to an existing task ID.
"""

import json
import pathlib

import pytest


@pytest.fixture(scope="module")
def tasks_data() -> dict:
    """Load and cache the parsed tasks.json for the entire module.

    The file is ~600 KB / 7500 lines; loading it once avoids redundant I/O
    across the four test functions in this module.

    Note: cannot depend on the function-scoped ``project_root`` fixture from
    conftest.py when this fixture is module-scoped, so the project root is
    resolved directly from this file's location (same technique used in
    conftest.py for ``_PROJECT_ROOT``).
    """
    project_root = pathlib.Path(__file__).resolve().parent.parent
    tasks_path = project_root / ".taskmaster" / "tasks" / "tasks.json"
    assert tasks_path.is_file(), f"tasks.json not found at {tasks_path}"
    with open(tasks_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Structural smoke test
# ---------------------------------------------------------------------------


def test_tasks_json_loads_and_has_expected_structure(tasks_data: dict) -> None:
    """tasks.json parses as valid JSON and has the expected top-level shape."""
    assert "master" in tasks_data, (
        "tasks.json must have a top-level 'master' key"
    )
    master = tasks_data["master"]
    assert "tasks" in master, (
        "tasks.json['master'] must contain a 'tasks' key"
    )
    tasks = master["tasks"]
    assert isinstance(tasks, list), (
        f"tasks.json['master']['tasks'] must be a list, got {type(tasks).__name__}"
    )
    assert len(tasks) > 0, (
        "tasks.json['master']['tasks'] must be non-empty"
    )


# ---------------------------------------------------------------------------
# Task ID type invariant
# ---------------------------------------------------------------------------


def test_all_task_ids_are_strings(tasks_data: dict) -> None:
    """Every task.id must be a str (regression guard for int→str migration)."""
    violations = []
    for idx, task in enumerate(tasks_data["master"]["tasks"]):
        task_id = task.get("id")
        if not isinstance(task_id, str):
            violations.append(
                f"  tasks[{idx}]: id={task_id!r} (type={type(task_id).__name__})"
            )
    assert not violations, (
        f"Found {len(violations)} task(s) with non-string id:\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Dependency entry type invariant
# ---------------------------------------------------------------------------


def test_all_dependency_entries_are_strings(tasks_data: dict) -> None:
    """Every entry in each task's 'dependencies' list must be a str."""
    violations = []
    for task in tasks_data["master"]["tasks"]:
        task_id = task.get("id", "<unknown>")
        for dep in task.get("dependencies", []):
            if not isinstance(dep, str):
                violations.append(
                    f"  task id={task_id!r}: dependency={dep!r} (type={type(dep).__name__})"
                )
    assert not violations, (
        f"Found {len(violations)} non-string dependency reference(s):\n"
        + "\n".join(violations)
    )
