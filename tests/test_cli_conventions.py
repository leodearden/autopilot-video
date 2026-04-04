"""Convention guard tests for test_cli.py.

Ensures test_cli.py consistently uses the Mock assertion API
(e.g. mock.assert_not_called()) rather than bare ``assert not mock.called``
which silently swallows custom messages when written as tuples.
"""

from __future__ import annotations

import re
from pathlib import Path


_TEST_CLI_PATH = Path(__file__).with_name("test_cli.py")


def test_no_bare_assert_not_called_pattern() -> None:
    """test_cli.py should use mock.assert_not_called() instead of assert not mock.called."""
    source = _TEST_CLI_PATH.read_text()
    matches = re.findall(r"assert\s+not\s+\w+\.called\b", source)
    assert matches == [], (
        f"Found {len(matches)} 'assert not <mock>.called' anti-pattern(s) in test_cli.py. "
        "Use <mock>.assert_not_called() instead."
    )
