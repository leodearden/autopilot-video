"""Centralized Claude CLI invocation wrapper.

Provides invoke_claude() for calling the Claude CLI as a subprocess,
replacing direct Anthropic SDK usage across all LLM-calling modules.
"""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

__all__ = ["LlmError", "invoke_claude"]

logger = logging.getLogger(__name__)


class LlmError(Exception):
    """Raised for all LLM invocation failures."""


def invoke_claude(
    *,
    prompt: str,
    system: str,
    model: str,
    max_tokens: int,
    json_schema: dict[str, Any] | None = None,
    use_api: bool = False,
    timeout: float | None = None,
) -> str | dict[str, Any]:
    """Invoke the Claude CLI and return the response text.

    Args:
        prompt: User message / prompt text.
        system: System prompt content.
        model: Model name (short CLI name like 'opus' or full ID).
        max_tokens: Maximum tokens for the response.
        json_schema: If provided, enables --json-schema structured output.
        use_api: If True, use Anthropic SDK instead of CLI.
        timeout: Subprocess timeout in seconds. None for no timeout.

    Returns:
        Response text (str) for simple calls, or parsed dict when json_schema is used.

    Raises:
        LlmError: On any invocation failure.
    """
    cmd = [
        "claude",
        "--print",
        "--output-format", "json",
        "--model", model,
        "--max-tokens", str(max_tokens),
        "--system-prompt", system,
    ]

    if json_schema is not None:
        cmd.extend(["--json-schema", json.dumps(json_schema)])

    cmd.extend(["--", prompt])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise LlmError("Claude CLI not found — is it installed?") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        raise LlmError(f"Claude CLI exited with code {exc.returncode}: {stderr}") from exc
    except subprocess.TimeoutExpired as exc:
        raise LlmError(f"Claude CLI timeout after {exc.timeout}s") from exc

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LlmError(f"Failed to parse CLI JSON output: {exc}") from exc

    # Structured output mode: return parsed dict from 'structured_output'
    if json_schema is not None:
        structured = data.get("structured_output")
        if structured is None:
            raise LlmError(
                f"Missing structured_output in CLI output (keys: {list(data.keys())})"
            )
        return structured

    # Simple text mode: return 'result' field
    result_text = data.get("result")
    if not result_text:
        raise LlmError(f"Empty or missing result in CLI output (keys: {list(data.keys())})")

    return result_text
