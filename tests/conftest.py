"""Shared fixtures and helpers for prompt template tests."""

import json
import re
from pathlib import Path

import pytest


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def prompts_dir(project_root: Path) -> Path:
    """Return the prompts directory."""
    return project_root / "autopilot" / "prompts"


def extract_json_blocks(text: str) -> list:
    """Extract and parse all fenced ```json code blocks from markdown text.

    Finds all occurrences of ```json ... ``` in the given text,
    parses each as JSON, and returns a list of parsed objects.
    """
    pattern = r"```json\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    results = []
    for match in matches:
        parsed = json.loads(match.strip())
        results.append(parsed)
    return results
