"""Tests for the centralized Claude CLI invocation wrapper (autopilot.llm)."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest


# -- Step 1: Basic subprocess call tests ---------------------------------------


class TestInvokeClaudeBasic:
    """Verify invoke_claude builds correct subprocess command and parses output."""

    def test_builds_correct_cli_args(self):
        """invoke_claude passes correct args: claude --print --output-format json --model ... --system-prompt ... -- <prompt>."""
        from autopilot.llm import invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "Hello from Claude"})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            invoke_claude(
                prompt="What is 2+2?",
                system="You are a math tutor.",
                model="sonnet",
                max_tokens=1024,
            )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        # Verify core structure
        assert cmd[0] == "claude"
        assert "--print" in cmd
        assert "--output-format" in cmd
        idx = cmd.index("--output-format")
        assert cmd[idx + 1] == "json"

        # Verify model
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "sonnet"

        # Verify system prompt
        assert "--system-prompt" in cmd
        idx = cmd.index("--system-prompt")
        assert cmd[idx + 1] == "You are a math tutor."

        # Verify prompt is after --
        assert "--" in cmd
        dd_idx = cmd.index("--")
        assert cmd[dd_idx + 1] == "What is 2+2?"

    def test_parses_result_field_from_json(self):
        """invoke_claude extracts and returns the 'result' field from JSON output."""
        from autopilot.llm import invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "The answer is 4."})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            text = invoke_claude(
                prompt="What is 2+2?",
                system="You are a math tutor.",
                model="sonnet",
                max_tokens=1024,
            )

        assert text == "The answer is 4."

    def test_returns_string_type(self):
        """invoke_claude returns a plain string, not bytes or dict."""
        from autopilot.llm import invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "response text"})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            text = invoke_claude(
                prompt="test",
                system="test system",
                model="sonnet",
                max_tokens=512,
            )

        assert isinstance(text, str)

    def test_subprocess_run_kwargs(self):
        """invoke_claude calls subprocess.run with capture_output=True, text=True, check=True."""
        from autopilot.llm import invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "ok"})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            invoke_claude(
                prompt="test",
                system="sys",
                model="sonnet",
                max_tokens=512,
            )

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["capture_output"] is True
        assert call_kwargs["text"] is True

    def test_passes_max_tokens_flag(self):
        """invoke_claude includes --max-tokens in the CLI command."""
        from autopilot.llm import invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "ok"})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            invoke_claude(
                prompt="test",
                system="sys",
                model="sonnet",
                max_tokens=2048,
            )

        cmd = mock_run.call_args[0][0]
        assert "--max-tokens" in cmd
        idx = cmd.index("--max-tokens")
        assert cmd[idx + 1] == "2048"


# -- Step 3: Error handling tests ----------------------------------------------


class TestInvokeClaudeErrors:
    """Verify error handling wraps all failure modes as LlmError."""

    def test_cli_not_found_raises_llm_error(self):
        """FileNotFoundError (CLI not installed) is wrapped as LlmError."""
        from autopilot.llm import LlmError, invoke_claude

        with patch("subprocess.run", side_effect=FileNotFoundError("claude")):
            with pytest.raises(LlmError, match="[Cc]laude CLI"):
                invoke_claude(prompt="test", system="sys", model="sonnet", max_tokens=512)

    def test_nonzero_exit_raises_llm_error(self):
        """CalledProcessError on non-zero exit raises LlmError with stderr."""
        from autopilot.llm import LlmError, invoke_claude

        err = subprocess.CalledProcessError(1, "claude", stderr="Rate limit exceeded")
        with patch("subprocess.run", side_effect=err):
            with pytest.raises(LlmError, match="Rate limit"):
                invoke_claude(prompt="test", system="sys", model="sonnet", max_tokens=512)

    def test_timeout_raises_llm_error(self):
        """TimeoutExpired raises LlmError."""
        from autopilot.llm import LlmError, invoke_claude

        err = subprocess.TimeoutExpired("claude", 30)
        with patch("subprocess.run", side_effect=err):
            with pytest.raises(LlmError, match="[Tt]imeout"):
                invoke_claude(
                    prompt="test", system="sys", model="sonnet", max_tokens=512, timeout=30
                )

    def test_invalid_json_raises_llm_error(self):
        """Invalid JSON in stdout raises LlmError."""
        from autopilot.llm import LlmError, invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NOT JSON AT ALL"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(LlmError, match="[Jj]son|[Pp]arse"):
                invoke_claude(prompt="test", system="sys", model="sonnet", max_tokens=512)

    def test_empty_result_raises_llm_error(self):
        """Empty 'result' field in JSON raises LlmError."""
        from autopilot.llm import LlmError, invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": ""})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(LlmError, match="[Ee]mpty"):
                invoke_claude(prompt="test", system="sys", model="sonnet", max_tokens=512)

    def test_missing_result_key_raises_llm_error(self):
        """Missing 'result' key in JSON raises LlmError."""
        from autopilot.llm import LlmError, invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"output": "something"})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(LlmError, match="[Rr]esult|[Mm]issing"):
                invoke_claude(prompt="test", system="sys", model="sonnet", max_tokens=512)


# -- Step 5: JSON schema support tests ----------------------------------------


class TestInvokeClaudeJsonSchema:
    """Verify --json-schema structured output support."""

    def test_json_schema_adds_flag(self):
        """When json_schema is passed, --json-schema flag is added to command."""
        from autopilot.llm import invoke_claude

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"structured_output": {"name": "test"}})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            invoke_claude(
                prompt="test",
                system="sys",
                model="sonnet",
                max_tokens=512,
                json_schema=schema,
            )

        cmd = mock_run.call_args[0][0]
        assert "--json-schema" in cmd
        idx = cmd.index("--json-schema")
        # The schema should be JSON-serialized
        parsed = json.loads(cmd[idx + 1])
        assert parsed == schema

    def test_json_schema_returns_structured_output(self):
        """With json_schema, return value comes from 'structured_output' field."""
        from autopilot.llm import invoke_claude

        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        expected_output = {"count": 42}

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"structured_output": expected_output})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            result = invoke_claude(
                prompt="test",
                system="sys",
                model="sonnet",
                max_tokens=512,
                json_schema=schema,
            )

        assert result == expected_output
        assert isinstance(result, dict)

    def test_no_schema_does_not_add_flag(self):
        """Without json_schema, --json-schema flag is NOT in the command."""
        from autopilot.llm import invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "text"})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            invoke_claude(prompt="test", system="sys", model="sonnet", max_tokens=512)

        cmd = mock_run.call_args[0][0]
        assert "--json-schema" not in cmd


# -- Step 7: Model name mapping tests -----------------------------------------


class TestModelMapping:
    """Verify _resolve_model maps full Anthropic IDs to CLI short names."""

    def test_opus_full_to_short(self):
        """claude-opus-4-20250514 maps to 'opus'."""
        from autopilot.llm import _resolve_model

        assert _resolve_model("claude-opus-4-20250514") == "opus"

    def test_sonnet_full_to_short(self):
        """claude-sonnet-4-20250514 maps to 'sonnet'."""
        from autopilot.llm import _resolve_model

        assert _resolve_model("claude-sonnet-4-20250514") == "sonnet"

    def test_short_name_passthrough(self):
        """Already-short names like 'opus' pass through unchanged."""
        from autopilot.llm import _resolve_model

        assert _resolve_model("opus") == "opus"
        assert _resolve_model("sonnet") == "sonnet"

    def test_unknown_model_passthrough(self):
        """Unknown model IDs pass through unchanged."""
        from autopilot.llm import _resolve_model

        assert _resolve_model("unknown-model-v3") == "unknown-model-v3"

    def test_invoke_claude_applies_mapping(self):
        """invoke_claude applies model mapping before building the CLI command."""
        from autopilot.llm import invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "ok"})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            invoke_claude(
                prompt="test",
                system="sys",
                model="claude-opus-4-20250514",
                max_tokens=512,
            )

        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "opus"
