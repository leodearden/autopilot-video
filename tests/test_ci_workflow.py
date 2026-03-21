"""Structural validation tests for the CI/CD GitHub Actions workflow."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


@pytest.fixture
def workflow() -> dict:
    """Load and parse the CI workflow YAML."""
    assert WORKFLOW_PATH.exists(), f"Workflow file not found: {WORKFLOW_PATH}"
    data = yaml.safe_load(WORKFLOW_PATH.read_text())
    # PyYAML parses the YAML key 'on' as boolean True; normalise it.
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


class TestTriggers:
    """Verify the workflow triggers on the correct events."""

    def test_push_to_main(self, workflow: dict) -> None:
        assert "push" in workflow["on"]
        push_branches = workflow["on"]["push"]["branches"]
        assert "main" in push_branches

    def test_pull_request_to_main(self, workflow: dict) -> None:
        assert "pull_request" in workflow["on"]
        pr_branches = workflow["on"]["pull_request"]["branches"]
        assert "main" in pr_branches


class TestMatrix:
    """Verify the Python version matrix."""

    def test_python_311_in_matrix(self, workflow: dict) -> None:
        job = workflow["jobs"]["ci"]
        matrix = job["strategy"]["matrix"]
        versions = [str(v) for v in matrix["python-version"]]
        assert "3.11" in versions

    def test_python_312_in_matrix(self, workflow: dict) -> None:
        job = workflow["jobs"]["ci"]
        matrix = job["strategy"]["matrix"]
        versions = [str(v) for v in matrix["python-version"]]
        assert "3.12" in versions


class TestSteps:
    """Verify all required CI steps are present."""

    @staticmethod
    def _step_names(workflow: dict) -> list[str]:
        """Extract step names from the CI job."""
        return [s.get("name", "") for s in workflow["jobs"]["ci"]["steps"]]

    @staticmethod
    def _step_uses(workflow: dict) -> list[str]:
        """Extract 'uses' actions from the CI job."""
        return [s.get("uses", "") for s in workflow["jobs"]["ci"]["steps"]]

    @staticmethod
    def _step_runs(workflow: dict) -> list[str]:
        """Extract 'run' commands from the CI job steps."""
        return [s.get("run", "") for s in workflow["jobs"]["ci"]["steps"]]

    def test_checkout_step(self, workflow: dict) -> None:
        uses = self._step_uses(workflow)
        assert any("actions/checkout" in u for u in uses)

    def test_uv_install_step(self, workflow: dict) -> None:
        uses = self._step_uses(workflow)
        assert any("astral-sh/setup-uv" in u for u in uses)

    def test_uv_sync_step(self, workflow: dict) -> None:
        runs = self._step_runs(workflow)
        assert any("uv sync" in r for r in runs)

    def test_ruff_check_step(self, workflow: dict) -> None:
        runs = self._step_runs(workflow)
        assert any("ruff check" in r for r in runs)

    def test_pyright_step(self, workflow: dict) -> None:
        runs = self._step_runs(workflow)
        assert any("pyright" in r for r in runs)

    def test_pytest_step(self, workflow: dict) -> None:
        runs = self._step_runs(workflow)
        assert any("pytest" in r for r in runs)

    def test_pytest_flags(self, workflow: dict) -> None:
        runs = self._step_runs(workflow)
        pytest_cmds = [r for r in runs if "pytest" in r]
        assert len(pytest_cmds) > 0
        pytest_cmd = pytest_cmds[0]
        assert "-x" in pytest_cmd
        assert "--tb=short" in pytest_cmd
