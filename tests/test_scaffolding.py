"""Tests for project scaffolding: directory structure, pyproject.toml, config.yaml."""

import pathlib


def test_directory_structure_exists(project_root: pathlib.Path) -> None:
    """All required directories exist."""
    required_dirs = [
        "autopilot",
        "autopilot/ingest",
        "autopilot/analyze",
        "autopilot/organize",
        "autopilot/plan",
        "autopilot/source",
        "autopilot/render",
        "autopilot/upload",
        "autopilot/prompts",
        "tests/fixtures",
        "scripts",
        "output",
    ]
    for d in required_dirs:
        assert (project_root / d).is_dir(), f"Directory {d}/ must exist"


def test_python_packages_have_init(project_root: pathlib.Path) -> None:
    """Every Python package has an __init__.py file."""
    python_packages = [
        "autopilot",
        "autopilot/ingest",
        "autopilot/analyze",
        "autopilot/organize",
        "autopilot/plan",
        "autopilot/source",
        "autopilot/render",
        "autopilot/upload",
    ]
    for pkg in python_packages:
        init = project_root / pkg / "__init__.py"
        assert init.is_file(), f"{pkg}/__init__.py must exist"


def test_prompts_dir_is_not_python_package(project_root: pathlib.Path) -> None:
    """The prompts directory is for .md templates, not a Python package."""
    assert not (project_root / "autopilot" / "prompts" / "__init__.py").exists(), (
        "autopilot/prompts/ should NOT have __init__.py"
    )
