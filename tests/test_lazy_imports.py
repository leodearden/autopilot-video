"""Tests that importing autopilot.orchestrator does NOT eagerly load stage modules.

These tests run in a subprocess to avoid contamination from the test runner's
own imports. A fresh Python process ensures sys.modules starts clean.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Stage subpackage prefixes that should NOT appear in sys.modules
# after a bare `import autopilot.orchestrator`.
STAGE_PREFIXES = (
    "autopilot.analyze.",
    "autopilot.ingest.",
    "autopilot.organize.",
    "autopilot.plan.",
    "autopilot.render.",
    "autopilot.source.",
    "autopilot.upload.",
)


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Run *code* in a fresh Python subprocess and return the result."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestLazyImportOrchestrator:
    """Verify that importing autopilot.orchestrator is lazy w.r.t. stage modules."""

    def test_no_stage_modules_on_import(self) -> None:
        """Importing orchestrator must not pull in any stage subpackage modules."""
        result = _run_in_subprocess("""\
            import sys
            import autopilot.orchestrator

            leaked = [
                m for m in sorted(sys.modules)
                if any(m.startswith(p) for p in %(prefixes)r)
            ]
            if leaked:
                print("LEAKED:" + ",".join(leaked))
                sys.exit(1)
            print("OK")
        """ % {"prefixes": STAGE_PREFIXES})

        assert result.returncode == 0, (
            f"Stage modules loaded on import:\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )

    def test_orchestrator_instantiation_no_stage_modules(self) -> None:
        """PipelineOrchestrator() must not trigger stage module imports."""
        result = _run_in_subprocess("""\
            import sys
            import autopilot.orchestrator as orch

            # Instantiate with a minimal config dict
            _obj = orch.PipelineOrchestrator({"project": "/tmp/test"})

            leaked = [
                m for m in sorted(sys.modules)
                if any(m.startswith(p) for p in %(prefixes)r)
            ]
            if leaked:
                print("LEAKED:" + ",".join(leaked))
                sys.exit(1)
            print("OK")
        """ % {"prefixes": STAGE_PREFIXES})

        assert result.returncode == 0, (
            f"Stage modules loaded on instantiation:\n"
            f"stdout: {result.stdout.strip()}\n"
            f"stderr: {result.stderr.strip()}"
        )
