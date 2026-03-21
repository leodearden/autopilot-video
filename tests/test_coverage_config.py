"""Tests verifying pytest-cov configuration in pyproject.toml."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


@pytest.fixture
def pyproject(project_root: Path) -> dict:
    """Load and parse pyproject.toml."""
    path = project_root / "pyproject.toml"
    assert path.exists(), f"pyproject.toml not found at {path}"
    with open(path, "rb") as f:
        return tomllib.load(f)


class TestCoverageDependency:
    """Verify pytest-cov is listed as a dev dependency."""

    def test_pytest_cov_in_dev_deps(self, pyproject: dict) -> None:
        dev_deps = pyproject["project"]["optional-dependencies"]["dev"]
        assert "pytest-cov" in dev_deps


class TestCoverageRunConfig:
    """Verify [tool.coverage.run] configuration."""

    def test_coverage_run_section_exists(self, pyproject: dict) -> None:
        assert "coverage" in pyproject["tool"]
        assert "run" in pyproject["tool"]["coverage"]

    def test_coverage_source(self, pyproject: dict) -> None:
        source = pyproject["tool"]["coverage"]["run"]["source"]
        assert source == ["autopilot"]

    def test_coverage_omit(self, pyproject: dict) -> None:
        omit = pyproject["tool"]["coverage"]["run"]["omit"]
        assert "*/tests/*" in omit


class TestCoverageReportConfig:
    """Verify [tool.coverage.report] configuration."""

    def test_coverage_report_section_exists(self, pyproject: dict) -> None:
        assert "report" in pyproject["tool"]["coverage"]

    def test_show_missing(self, pyproject: dict) -> None:
        assert pyproject["tool"]["coverage"]["report"]["show_missing"] is True

    def test_fail_under(self, pyproject: dict) -> None:
        assert pyproject["tool"]["coverage"]["report"]["fail_under"] == 70


class TestCICoverageFlags:
    """Verify CI workflow passes coverage flags to pytest."""

    def test_ci_pytest_has_cov_flag(self) -> None:
        """Coverage flags belong in CI (not addopts) to avoid fail_under on subset runs."""
        from pathlib import Path

        import yaml

        ci_path = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
        data = yaml.safe_load(ci_path.read_text())
        runs = [s.get("run", "") for s in data["jobs"]["ci"]["steps"]]
        pytest_cmds = [r for r in runs if "pytest" in r]
        assert len(pytest_cmds) > 0
        pytest_cmd = pytest_cmds[0]
        assert "--cov=autopilot" in pytest_cmd
        assert "--cov-report=term-missing" in pytest_cmd
