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


# -- Step 9: API fallback tests -----------------------------------------------


class TestInvokeClaudeApiFallback:
    """Verify use_api=True routes to Anthropic SDK instead of subprocess."""

    def test_api_fallback_calls_anthropic(self):
        """use_api=True calls anthropic.Anthropic().messages.create()."""
        import sys

        from autopilot.llm import invoke_claude

        mock_content = MagicMock()
        mock_content.text = "API response text"
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = invoke_claude(
                prompt="test",
                system="sys",
                model="sonnet",
                max_tokens=512,
                use_api=True,
            )

        assert result == "API response text"
        mock_client.messages.create.assert_called_once()

    def test_api_fallback_passes_correct_params(self):
        """API fallback passes model, max_tokens, system, messages correctly."""
        import sys

        from autopilot.llm import invoke_claude

        mock_content = MagicMock()
        mock_content.text = "ok"
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            invoke_claude(
                prompt="What is 2+2?",
                system="You are a tutor.",
                model="claude-opus-4-20250514",
                max_tokens=1024,
                use_api=True,
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        # API should use the FULL model name, not the CLI short name
        assert call_kwargs["model"] == "claude-opus-4-20250514"
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["system"] == "You are a tutor."
        assert call_kwargs["messages"] == [{"role": "user", "content": "What is 2+2?"}]

    def test_api_fallback_does_not_call_subprocess(self):
        """use_api=True should NOT call subprocess.run."""
        import sys

        from autopilot.llm import invoke_claude

        mock_content = MagicMock()
        mock_content.text = "ok"
        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with (
            patch.dict(sys.modules, {"anthropic": mock_anthropic}),
            patch("subprocess.run") as mock_run,
        ):
            invoke_claude(
                prompt="test",
                system="sys",
                model="sonnet",
                max_tokens=512,
                use_api=True,
            )

        mock_run.assert_not_called()

    def test_use_api_false_does_not_import_anthropic(self):
        """use_api=False (default) should not trigger anthropic import."""
        from autopilot.llm import invoke_claude

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"result": "ok"})
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result):
            # This should work without anthropic being importable
            invoke_claude(prompt="test", system="sys", model="sonnet", max_tokens=512)


# -- Step 11: LLMConfig.use_api and --api-fallback flag tests -----------------


class TestLLMConfigUseApi:
    """Verify LLMConfig.use_api field and CLI --api-fallback flag."""

    def test_use_api_defaults_false(self):
        """LLMConfig.use_api defaults to False."""
        from autopilot.config import LLMConfig

        config = LLMConfig()
        assert config.use_api is False

    def test_use_api_can_be_set_true(self):
        """LLMConfig.use_api can be set to True."""
        from autopilot.config import LLMConfig

        config = LLMConfig(use_api=True)
        assert config.use_api is True

    def test_api_fallback_flag_on_main_group(self):
        """main Click group has --api-fallback flag."""
        from autopilot.cli import main

        param_names = [p.name for p in main.params]
        assert "api_fallback" in param_names

    def test_api_fallback_flag_stores_in_context(self):
        """--api-fallback stores api_fallback=True in click context."""
        from click.testing import CliRunner

        from autopilot.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--api-fallback", "--help"])
        assert result.exit_code == 0


# -- Step 13: classify._call_llm migration tests ------------------------------


class TestClassifyMigration:
    """Verify classify._call_llm uses invoke_claude instead of anthropic SDK."""

    def test_call_llm_uses_invoke_claude(self):
        """classify._call_llm calls invoke_claude with correct params."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import _call_llm

        config = LLMConfig()
        summary = {"time_range": "2025-01-01T10:00:00 to 2025-01-01T11:00:00"}

        llm_response = json.dumps({
            "label": "Morning hike",
            "description": "A scenic hike.",
            "split_recommended": False,
            "split_reason": None,
        })

        with patch("autopilot.organize.classify.invoke_claude", return_value=llm_response) as mock_invoke:
            result = _call_llm(summary, config)

        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["model"] == config.utility_model
        assert call_kwargs["max_tokens"] == 1024
        assert "system" in call_kwargs
        assert json.dumps(summary, indent=2) in call_kwargs["prompt"]

    def test_call_llm_returns_parsed_json(self):
        """classify._call_llm parses JSON from invoke_claude text response."""
        from autopilot.config import LLMConfig
        from autopilot.organize.classify import _call_llm

        config = LLMConfig()
        summary = {"time_range": "test"}

        llm_response = json.dumps({
            "label": "Beach day",
            "description": "Fun at the beach.",
            "split_recommended": False,
            "split_reason": None,
        })

        with patch("autopilot.organize.classify.invoke_claude", return_value=llm_response):
            result = _call_llm(summary, config)

        assert result["label"] == "Beach day"
        assert result["description"] == "Fun at the beach."

    def test_call_llm_no_anthropic_import(self):
        """classify._call_llm should NOT import anthropic directly."""
        import inspect

        from autopilot.organize import classify

        source = inspect.getsource(classify._call_llm)
        assert "import anthropic" not in source


# -- Step 15: narratives._call_llm migration tests ----------------------------


class TestNarrativesMigration:
    """Verify narratives._call_llm uses invoke_claude instead of anthropic SDK."""

    def test_call_llm_uses_invoke_claude(self):
        """narratives._call_llm calls invoke_claude with correct params."""
        from pathlib import Path

        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import _call_llm

        config = AutopilotConfig(input_dir=Path("."), output_dir=Path("."))
        storyboard = "# Master Storyboard\n\nCluster c1: Temple visit"

        llm_response = json.dumps([{
            "title": "Thailand Day",
            "activity_cluster_ids": ["c1"],
            "proposed_duration_seconds": 480,
        }])

        with patch("autopilot.organize.narratives.invoke_claude", return_value=llm_response) as mock_invoke:
            result = _call_llm(storyboard, config)

        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["model"] == config.llm.planning_model
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["prompt"] == storyboard
        assert "system" in call_kwargs

    def test_call_llm_returns_text(self):
        """narratives._call_llm returns raw text from invoke_claude."""
        from pathlib import Path

        from autopilot.config import AutopilotConfig
        from autopilot.organize.narratives import _call_llm

        config = AutopilotConfig(input_dir=Path("."), output_dir=Path("."))

        llm_response = "some response text"

        with patch("autopilot.organize.narratives.invoke_claude", return_value=llm_response):
            result = _call_llm("storyboard", config)

        assert result == "some response text"

    def test_call_llm_no_anthropic_import(self):
        """narratives._call_llm should NOT import anthropic directly."""
        import inspect

        from autopilot.organize import narratives

        source = inspect.getsource(narratives._call_llm)
        assert "import anthropic" not in source


# -- Step 17: script._call_llm migration tests --------------------------------


class TestScriptMigration:
    """Verify script._call_llm uses invoke_claude instead of anthropic SDK."""

    def test_call_llm_uses_invoke_claude(self):
        """script._call_llm calls invoke_claude with correct params."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import _call_llm

        config = LLMConfig()
        user_message = "## Approved Narrative\n\nTest storyboard"
        system_prompt = "You are a script writer."

        llm_response = json.dumps({
            "scenes": [{"scene_number": 1, "title": "Opening"}],
        })

        with patch("autopilot.plan.script.invoke_claude", return_value=llm_response) as mock_invoke:
            result = _call_llm(user_message, system_prompt, config)

        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[1]
        assert call_kwargs["model"] == config.planning_model
        assert call_kwargs["max_tokens"] == 8192
        assert call_kwargs["prompt"] == user_message
        assert call_kwargs["system"] == system_prompt

    def test_call_llm_returns_text(self):
        """script._call_llm returns raw text from invoke_claude."""
        from autopilot.config import LLMConfig
        from autopilot.plan.script import _call_llm

        config = LLMConfig()

        with patch("autopilot.plan.script.invoke_claude", return_value="script output"):
            result = _call_llm("msg", "sys", config)

        assert result == "script output"

    def test_call_llm_no_anthropic_import(self):
        """script._call_llm should NOT import anthropic directly."""
        import inspect

        from autopilot.plan import script

        source = inspect.getsource(script._call_llm)
        assert "import anthropic" not in source
