"""Tests for the FastAPI web application skeleton."""

from __future__ import annotations

import re
from pathlib import Path

import click
import pytest
from fastapi import FastAPI, Request

from autopilot.web.app import create_app


@pytest.fixture
def app(tmp_path: Path) -> FastAPI:
    """Create a FastAPI app with a temporary db path."""
    app = create_app(str(tmp_path / "catalog.db"))

    # Add a test-only route that renders base.html for template testing
    @app.get("/test-base-template")
    def _test_base_template(request: Request):
        return app.state.templates.TemplateResponse(request, "base.html", {"page_title": "Test"})

    return app


class TestCreateApp:
    """Tests for the create_app factory function."""

    def test_create_app_returns_fastapi_instance(self, tmp_path: Path) -> None:
        """create_app(db_path) returns a FastAPI instance."""
        db_path = str(tmp_path / "catalog.db")
        app = create_app(db_path)
        assert isinstance(app, FastAPI)


class TestHealthCheck:
    """Tests for the /api/health endpoint."""

    def test_health_returns_200_ok(self, client) -> None:
        """GET /api/health returns 200 with {'status': 'ok'}."""
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestStaticFiles:
    """Tests for static file serving."""

    def test_app_css_served(self, client) -> None:
        """GET /static/app.css returns 200."""
        response = client.get("/static/app.css")
        assert response.status_code == 200

    def test_app_js_served(self, client) -> None:
        """GET /static/app.js returns 200."""
        response = client.get("/static/app.js")
        assert response.status_code == 200


class TestBaseTemplate:
    """Tests for the base.html Jinja2 template."""

    def test_base_template_contains_tailwind_cdn(self, client) -> None:
        """base.html includes the Tailwind CSS CDN play script."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "tailwindcss" in response.text.lower() or "cdn.tailwindcss.com" in response.text

    def test_base_template_contains_htmx(self, client) -> None:
        """base.html includes the HTMX CDN script."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "htmx" in response.text.lower()

    def test_base_template_has_nav_items(self, client) -> None:
        """base.html nav contains all required menu items."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        html = response.text
        for item in ["Dashboard", "Media", "Pipeline", "Review", "Gates", "Settings"]:
            assert item in html, f"Nav item '{item}' not found in base.html"

    def test_base_template_has_notification_bell(self, client) -> None:
        """base.html contains a notification bell placeholder."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "notification" in response.text.lower()

    def test_base_template_has_toast_container(self, client) -> None:
        """base.html contains a toast container div."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "toast-container" in response.text

    def test_base_template_has_dark_theme(self, client) -> None:
        """base.html uses dark theme classes."""
        response = client.get("/test-base-template")
        assert response.status_code == 200
        assert "dark" in response.text


class TestServeCommand:
    """Tests for the 'autopilot serve' CLI command."""

    def test_serve_registered_in_main(self) -> None:
        """'serve' is a registered subcommand of main."""
        from autopilot.cli import main

        assert "serve" in main.commands

    def test_serve_has_host_option(self) -> None:
        """serve command has --host option with default '127.0.0.1'."""
        from autopilot.cli import main

        cmd = main.commands["serve"]
        param_names = {p.name: p for p in cmd.params}
        assert "host" in param_names
        assert param_names["host"].default == "127.0.0.1"

    def test_serve_has_port_option(self) -> None:
        """serve command has --port option with default 8080, type int."""
        from autopilot.cli import main

        cmd = main.commands["serve"]
        param_names = {p.name: p for p in cmd.params}
        assert "port" in param_names
        assert param_names["port"].default == 8080
        assert param_names["port"].type == click.INT

    def test_serve_has_output_dir_option(self) -> None:
        """serve command has --output-dir option."""
        from autopilot.cli import main

        cmd = main.commands["serve"]
        param_names = [p.name for p in cmd.params]
        assert "output_dir" in param_names

    def test_serve_invokes_uvicorn(self, tmp_path: Path) -> None:
        """serve --output-dir calls uvicorn.run with the created app."""
        from unittest.mock import MagicMock, patch

        from click.testing import CliRunner

        from autopilot.cli import main

        mock_app = MagicMock()
        mock_create_app = MagicMock(return_value=mock_app)
        mock_uvicorn_run = MagicMock()

        with (
            patch("autopilot.web.app.create_app", mock_create_app),
            patch("uvicorn.run", mock_uvicorn_run),
        ):
            runner = CliRunner()
            result = runner.invoke(main, ["serve", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0, result.output
        expected_db = str(tmp_path / "catalog.db")
        mock_create_app.assert_called_once_with(expected_db)
        mock_uvicorn_run.assert_called_once_with(mock_app, host="127.0.0.1", port=8080)

    def test_serve_passes_custom_host_port(self, tmp_path: Path) -> None:
        """serve --host 0.0.0.0 --port 9090 passes values to uvicorn.run."""
        from unittest.mock import MagicMock, patch

        from click.testing import CliRunner

        from autopilot.cli import main

        mock_app = MagicMock()
        mock_create_app = MagicMock(return_value=mock_app)
        mock_uvicorn_run = MagicMock()

        with (
            patch("autopilot.web.app.create_app", mock_create_app),
            patch("uvicorn.run", mock_uvicorn_run),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["serve", "--host", "0.0.0.0", "--port", "9090", "--output-dir", str(tmp_path)],
            )

        assert result.exit_code == 0, result.output
        mock_uvicorn_run.assert_called_once_with(mock_app, host="0.0.0.0", port=9090)


def _read_app_js() -> str:
    """Read the app.js source file."""
    js_path = Path(__file__).resolve().parent.parent / "autopilot" / "web" / "static" / "app.js"
    return js_path.read_text()


def _brace_match_from(source: str, start: int, label: str) -> str:
    """Extract a brace-delimited block from *source* starting at index *start*.

    *start* must point at an opening ``{``.  The function walks forward counting
    brace depth and returns ``source[start:end]`` (braces included) once the
    matching closing ``}`` is found.

    Raises :class:`AssertionError` with a message containing *label* if
    ``source[start]`` is not ``{`` or if the braces are unbalanced.
    """
    assert 0 <= start < len(source), f"start {start} out of range for {label}"
    assert source[start] == "{", f"expected '{{' at position {start} in {label}"
    brace_depth = 0
    i = start
    while i < len(source):
        ch = source[i]
        if ch in ("'", '"', "`"):
            # Skip over the entire JS string literal, honoring backslash escapes.
            quote_char = ch
            i += 1
            while i < len(source):
                if source[i] == "\\":
                    i += 2  # skip escape sequence
                elif source[i] == quote_char:
                    i += 1  # consume closing quote
                    break
                else:
                    i += 1
            continue
        if ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(
        f"unbalanced braces in {label} (depth {brace_depth} after scan)"
    )


def _extract_listener_body(js_source: str, event_type: str) -> str:
    """Extract the body of a source.addEventListener callback by brace-matching.

    Returns the outermost ``{ ... }`` block of the callback function for the
    given *event_type*.  Raises :class:`AssertionError` if the listener is not
    found or if braces are not balanced (malformed JS).

    .. note::

       This helper only works for **inline function** listeners, e.g.
       ``source.addEventListener('event', function(e) { ... })``.  For
       factory-pattern listeners (e.g. ``makeStageHandler(...)``), a clear
       :class:`AssertionError` is raised — it will **not** silently return
       a subsequent listener's body.
    """
    marker = f"addEventListener('{event_type}'"
    listener_start = js_source.find(marker)
    assert listener_start != -1, f"{event_type} event listener not found in app.js"

    # Bound the search for 'function' to this listener's extent only — stop
    # before the next addEventListener( so a factory-pattern callback does not
    # silently fall through to a subsequent inline listener.
    next_listener = js_source.find("addEventListener(", listener_start + 1)
    search_end = next_listener if next_listener != -1 else len(js_source)

    func_start = js_source.find("function", listener_start, search_end)
    assert func_start != -1, f"function keyword not found in {event_type} listener"

    body_start = js_source.find("{", func_start)
    assert body_start != -1, f"opening brace not found in {event_type} listener"

    return _brace_match_from(js_source, body_start, f"{event_type} listener")


def _extract_function_body(js_source: str, func_name: str) -> str:
    """Extract the body of a named ``function`` declaration by brace-matching.

    Locates ``function <func_name>`` in *js_source*, finds its opening ``{``,
    and brace-matches to the corresponding closing ``}``.  Returns the
    outermost ``{ ... }`` block (braces included).

    Raises :class:`AssertionError` if *func_name* is not found or if the
    braces are not balanced (malformed JS).
    """
    match = re.search(rf'\bfunction\s+{re.escape(func_name)}\b', js_source)
    assert match is not None, f"{func_name} function not found in source"

    body_start = js_source.find("{", match.end())
    assert body_start != -1, f"opening brace not found for {func_name}"

    return _brace_match_from(js_source, body_start, func_name)


@pytest.fixture(scope="module")
def app_js_source() -> str:
    """Read app.js once and cache for the entire test module."""
    return _read_app_js()


@pytest.fixture(scope="class")
def make_stage_handler_body(app_js_source: str) -> str:
    """Extract makeStageHandler body once per class that needs it."""
    return _extract_function_body(app_js_source, "makeStageHandler")


class TestExtractFunctionBody:
    """Tests for the _extract_function_body helper."""

    def test_extracts_simple_function_body(self) -> None:
        """Extracts the body of a simple named function."""
        js = "function greet() { return 'hello'; }"
        body = _extract_function_body(js, "greet")
        assert body == "{ return 'hello'; }"

    def test_handles_nested_braces(self) -> None:
        """Correctly brace-matches when there are nested braces."""
        js = "function calc() { if (true) { return 1; } else { return 2; } }"
        body = _extract_function_body(js, "calc")
        assert body == "{ if (true) { return 1; } else { return 2; } }"

    def test_raises_when_function_not_found(self) -> None:
        """Raises AssertionError when the function name is not found."""
        js = "function otherFunc() { return 1; }"
        with pytest.raises(AssertionError, match="nonexistent"):
            _extract_function_body(js, "nonexistent")

    def test_raises_on_unclosed_braces(self) -> None:
        """Raises AssertionError when braces are not balanced (malformed JS)."""
        js = "function broken() { if (true) { return 1; }"
        with pytest.raises(AssertionError, match='unbalanced'):
            _extract_function_body(js, "broken")

    def test_does_not_match_prefix_name_collision(self) -> None:
        """Searching for 'makeStageHandler' must not match 'makeStageHandlerV2'."""
        js = (
            "function makeStageHandlerV2() { return 2; }\n"
            "function makeStageHandler() { return 1; }"
        )
        body = _extract_function_body(js, "makeStageHandler")
        assert body == "{ return 1; }"

    def test_targets_correct_function_among_multiple(self) -> None:
        """Each function in a multi-function source can be individually targeted."""
        js = (
            "function alpha() { return 1; }\n"
            "function beta() { return 2; }\n"
            "function gamma() { return 3; }"
        )
        assert _extract_function_body(js, "beta") == "{ return 2; }"
        assert _extract_function_body(js, "alpha") == "{ return 1; }"
        assert _extract_function_body(js, "gamma") == "{ return 3; }"


class TestExtractListenerBody:
    """Tests for the _extract_listener_body helper hardening."""

    def test_extracts_simple_listener_body(self) -> None:
        """Extracts the body of an inline addEventListener callback."""
        js = "source.addEventListener('myEvent', function(e) { doThing(); })"
        body = _extract_listener_body(js, "myEvent")
        assert body == "{ doThing(); }"

    def test_raises_on_unclosed_braces(self) -> None:
        """_extract_listener_body raises AssertionError on malformed JS with unclosed braces."""
        malformed_js = (
            "source.addEventListener('broken_event', function(event) "
            "{ if (true) { doStuff(); }"
        )
        with pytest.raises(AssertionError, match='unbalanced'):
            _extract_listener_body(malformed_js, "broken_event")

    def test_extract_listener_body_ignores_function_keyword_from_next_listener(self) -> None:
        """factory-pattern listener must not silently return the next listener's body.

        When the target listener uses a factory callback (no 'function' keyword),
        the unbounded find('function', ...) would walk into a subsequent inline
        listener and return the wrong body.  After the fix, an AssertionError
        referencing both 'function' and the event type must be raised instead.
        """
        js = (
            "source.addEventListener('first_event', makeStageHandler('first_event')); "
            "source.addEventListener('second_event', function(e) { doThing(); })"
        )
        with pytest.raises(AssertionError, match=r"function.*first_event|first_event.*function"):
            _extract_listener_body(js, "first_event")

    def test_extract_listener_body_finds_inline_listener_after_factory_listener(self) -> None:
        """Inline listener following a factory-pattern listener still extracts correctly.

        After bounding the search, an inline listener that comes *after* a factory
        listener must still be extractable by its own event type.
        """
        js = (
            "source.addEventListener('factory_event', makeStageHandler('factory_event')); "
            "source.addEventListener('target_event', function(e) { targetThing(); })"
        )
        body = _extract_listener_body(js, "target_event")
        assert body == "{ targetThing(); }"


class TestBraceMatchFrom:
    """Tests for the _brace_match_from helper."""

    def test_extracts_simple_braced_block(self) -> None:
        """Extracts a simple braced block from a known start index."""
        result = _brace_match_from("x { a; }", 2, "simple")
        assert result == "{ a; }"

    def test_handles_nested_braces(self) -> None:
        """Correctly brace-matches when there are nested braces."""
        source = "{ if (x) { y; } }"
        result = _brace_match_from(source, 0, "nested")
        assert result == source

    def test_raises_when_start_not_at_brace(self) -> None:
        """Raises AssertionError when source[start] is not '{'."""
        with pytest.raises(AssertionError, match=r"expected '\{'"):
            _brace_match_from("abc", 1, "no-brace")

    def test_raises_on_unbalanced_braces(self) -> None:
        """Raises AssertionError for shallow (depth-1) unbalanced braces."""
        with pytest.raises(AssertionError, match=r"unbalanced.*depth 1"):
            _brace_match_from("{ unclosed", 0, "shallow")

    def test_label_appears_in_error_message(self) -> None:
        """Error message includes the label for deeply-nested (depth-3) unclosed braces."""
        with pytest.raises(AssertionError, match=r"deep-block.*depth 3"):
            _brace_match_from("{ { { deep", 0, "deep-block")

    def test_unbalanced_error_includes_depth(self) -> None:
        """Error message from unbalanced braces includes 'depth' with a number."""
        with pytest.raises(AssertionError, match=r"depth \d+"):
            _brace_match_from("{ open { but no close", 0, "depth-check")

    def test_raises_on_out_of_range_start(self) -> None:
        """Raises AssertionError with 'out of range' for invalid start indices."""
        source = "{ ok }"
        # start == len(source)
        with pytest.raises(AssertionError, match="out of range"):
            _brace_match_from(source, len(source), "at-len")
        # start > len(source)
        with pytest.raises(AssertionError, match="out of range"):
            _brace_match_from(source, len(source) + 5, "beyond-len")
        # negative start
        with pytest.raises(AssertionError, match="out of range"):
            _brace_match_from(source, -1, "negative")
        # empty source
        with pytest.raises(AssertionError, match="out of range"):
            _brace_match_from("", 0, "empty")

    def test_skips_braces_inside_single_quoted_string(self) -> None:
        """A closing '}' inside a single-quoted JS string does not end the block."""
        source = "{ x = '}'; }"
        result = _brace_match_from(source, 0, "sq")
        assert result == source

    def test_skips_braces_inside_double_quoted_string(self) -> None:
        """A closing '}' inside a double-quoted JS string does not end the block."""
        source = '{ x = "}"; }'
        result = _brace_match_from(source, 0, "dq")
        assert result == source

    def test_skips_braces_inside_template_literal(self) -> None:
        """A closing '}' inside a backtick template literal does not end the block."""
        source = "{ x = `}`; }"
        result = _brace_match_from(source, 0, "tl")
        assert result == source

    def test_skips_opening_braces_inside_strings(self) -> None:
        """An opening '{' inside a string does not increment depth."""
        # Without string-literal skipping, the '{' at position 7 would push
        # depth to 2, and the lone '}' at the end would leave depth at 1,
        # raising an unbalanced error.
        source = "{ x = '{'; }"
        result = _brace_match_from(source, 0, "open-in-str")
        assert result == source

    def test_respects_backslash_escape_in_string_literals(self) -> None:
        """An escaped quote (\\') does not close the string; the following '}' stays inside."""
        # JS: { x = '\'}'; }
        # \\' is an escaped single-quote (keeps the string open), so the '}' at
        # position 9 is still inside the JS string.  The real closing brace is
        # the last character.
        source = "{ x = '\\'}'; }"
        result = _brace_match_from(source, 0, "escape")
        assert result == source


class TestFixtureSanity:
    """Sanity checks for module-scoped caching fixtures."""

    def test_app_js_source_returns_string(self, app_js_source: str) -> None:
        """app_js_source fixture returns a non-empty string containing 'function'."""
        assert isinstance(app_js_source, str)
        assert len(app_js_source) > 0
        assert "function" in app_js_source

    def test_make_stage_handler_body_contains_try_catch(self, make_stage_handler_body: str) -> None:
        """make_stage_handler_body fixture returns a string containing 'try' and 'catch'."""
        assert isinstance(make_stage_handler_body, str)
        assert "try" in make_stage_handler_body
        assert "catch" in make_stage_handler_body


class TestSSEErrorHandling:
    """Tests for SSE notification error handling in app.js."""

    def test_sse_notification_listener_has_try_catch(self, app_js_source: str) -> None:
        """The SSE notification listener wraps JSON.parse in try/catch."""
        listener_body = _extract_listener_body(app_js_source, "notification")

        # Assert try/catch wraps the JSON.parse
        assert "try" in listener_body, "try block not found in notification listener"
        assert "catch" in listener_body, "catch block not found in notification listener"
        assert "console.error" in listener_body, (
            "console.error not found in notification listener catch block"
        )


class TestSSEHandlerFactory:
    """Tests for the makeStageHandler factory function in app.js."""

    def test_make_stage_handler_defined(self, app_js_source: str) -> None:
        """makeStageHandler function is defined in app.js."""
        assert "function makeStageHandler" in app_js_source, (
            "makeStageHandler factory function not found in app.js"
        )

    def test_make_stage_handler_has_try_catch(self, make_stage_handler_body: str) -> None:
        """makeStageHandler contains try/catch error handling."""
        assert "try" in make_stage_handler_body, "try block not found in makeStageHandler"
        assert "catch" in make_stage_handler_body, "catch block not found in makeStageHandler"
        assert "console.error" in make_stage_handler_body, (
            "console.error not found in makeStageHandler catch block"
        )

    def test_make_stage_handler_has_console_warn_for_missing_stage(
        self, make_stage_handler_body: str,
    ) -> None:
        """makeStageHandler logs console.warn when data.stage is falsy."""
        assert "console.warn" in make_stage_handler_body, (
            "console.warn for missing stage field not found in makeStageHandler"
        )

    def test_make_stage_handler_calls_refresh_stage_card(
        self, make_stage_handler_body: str,
    ) -> None:
        """makeStageHandler calls debouncedRefreshStageCard when stage is present."""
        assert "debouncedRefreshStageCard" in make_stage_handler_body, (
            "debouncedRefreshStageCard call not found in makeStageHandler"
        )
        assert "if (refreshCard)" in make_stage_handler_body, (
            "conditional guard 'if (refreshCard)' not found in makeStageHandler. "
            "debouncedRefreshStageCard must only be called when the refreshCard param is true."
        )

    def test_setup_dashboard_sse_uses_factory(self, app_js_source: str) -> None:
        """setupDashboardSSE uses makeStageHandler for all 3 event types."""
        dashboard_body = _extract_function_body(app_js_source, "setupDashboardSSE")

        for event_type in ("stage_started", "stage_completed", "job_progress"):
            assert f"makeStageHandler('{event_type}'" in dashboard_body, (
                f"setupDashboardSSE does not use makeStageHandler for '{event_type}'"
            )

    def test_dashboard_factory_calls_use_default_refresh_card(self, app_js_source: str) -> None:
        """setupDashboardSSE factory calls do not explicitly disable refreshCard.

        makeStageHandler has exactly one boolean positional parameter (refreshCard,
        5th position).  Any literal ``false`` token in a call's argument list would
        mean refreshCard was explicitly set to false, which would silently break
        dashboard card refresh for events whose entire purpose is to refresh the
        stage card.  This test asserts that no such token is present in any of the
        three dashboard-wiring calls.
        """
        dashboard_body = _extract_function_body(app_js_source, "setupDashboardSSE")

        for event_type in ("stage_started", "stage_completed", "job_progress"):
            match = re.search(
                rf"makeStageHandler\('{event_type}'[^)]*\)", dashboard_body
            )
            assert match is not None, (
                f"makeStageHandler('{event_type}' ...) call not found in setupDashboardSSE"
            )
            call_text = match.group(0)
            assert "false" not in call_text, (
                f"makeStageHandler('{event_type}') call contains literal 'false', "
                f"which means refreshCard is explicitly disabled for this event: {call_text!r}"
            )

    def test_stage_error_handled_via_notification(self, app_js_source: str) -> None:
        """stage_error is handled server-side via notification events, not client-side."""
        # The unified notification handler in setupNotificationSSE handles all
        # user-facing notifications including stage errors.
        func_start = app_js_source.find("function setupNotificationSSE")
        assert func_start != -1
        func_body = app_js_source[func_start:]
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]
        assert "addEventListener('notification'" in func_body, (
            "setupNotificationSSE should handle unified notification events"
        )


    def test_make_stage_handler_shows_toast_on_missing_stage(
        self, make_stage_handler_body: str,
    ) -> None:
        """makeStageHandler shows a fallback toast when stage is missing and toastMsg is provided.

        This catches the regression where stage_error events with no stage field
        silently dropped the error toast from the UI.  The else branch (missing-stage
        path) must call showToast with '{stage}' replaced by 'unknown' so that
        error-level events always surface to the user.
        """
        factory_body = make_stage_handler_body

        # Locate the else branch (the missing-stage path)
        else_pos = factory_body.find("} else {")
        assert else_pos != -1, "else branch not found in makeStageHandler"

        # Extract the else block body via brace-matching
        else_brace_start = factory_body.find("{", else_pos + 1)
        assert else_brace_start != -1, "opening brace of else block not found"

        else_body = _brace_match_from(factory_body, else_brace_start, "else block")

        # The else branch must call showToast for fallback toast on missing stage
        assert "showToast" in else_body, (
            "showToast call not found in makeStageHandler else branch (missing-stage path). "
            "Error-level SSE events with no stage field should still show a fallback toast."
        )
        assert ".replace('{stage}', 'unknown')" in else_body, (
            "makeStageHandler else branch must call .replace('{stage}', 'unknown') to substitute "
            "the placeholder — a bare 'unknown' substring does not prove the replacement contract."
        )


class TestSSERunHandlerRobustness:
    """Tests that run_completed and run_failed handlers have try/catch."""

    def test_notification_handler_has_try_catch(self, app_js_source: str) -> None:
        """The unified notification handler wraps its body in try/catch.

        run_completed and run_failed are now handled server-side via notification
        events. The notification handler in setupNotificationSSE must have
        error handling to avoid silently dropping events.
        """
        body = _extract_listener_body(app_js_source, "notification")
        assert "try" in body, "try block not found in notification handler"
        assert "catch" in body, "catch block not found in notification handler"
        assert "console.error" in body, "console.error not found in notification catch block"


class TestDashboardSSEEventCoverage:
    """Comprehensive test ensuring all dashboard-relevant event types are handled."""

    # Job-level detail events are intentionally unhandled at dashboard level —
    # they update individual job rows, not stage cards.
    _JOB_LEVEL_EVENTS = {"job_started", "job_completed", "job_error"}

    # Server-only events that are in VALID_EVENT_TYPES but intentionally have no
    # source.addEventListener handler in app.js. These events are emitted by the
    # backend but the frontend does not act on them at the dashboard level.
    _SERVER_ONLY_EVENTS = {
        "gate_waiting",
        "gate_approved",
        "gate_skipped",
        "stage_error",
        "run_completed",
        "run_failed",
    }

    # Legacy SSE listeners in app.js that are not in VALID_EVENT_TYPES.
    # Currently empty — 'notification' was promoted to VALID_EVENT_TYPES.
    _LEGACY_SSE_LISTENERS: frozenset[str] = frozenset()

    def test_all_dashboard_event_types_handled(self, app_js_source: str) -> None:
        """Every VALID_EVENT_TYPE (except job-level and server-only) has a listener in app.js."""
        from autopilot.web.routes.sse import VALID_EVENT_TYPES

        dashboard_event_types = (
            set(VALID_EVENT_TYPES) - self._JOB_LEVEL_EVENTS - self._SERVER_ONLY_EVENTS
        )
        missing = []
        for event_type in sorted(dashboard_event_types):
            marker = f"source.addEventListener('{event_type}'"
            if marker not in app_js_source:
                missing.append(event_type)

        assert not missing, (
            f"Dashboard-relevant event types missing addEventListener in app.js: {missing}"
        )

    def test_no_orphaned_sse_listeners(self, app_js_source: str) -> None:
        """Every source.addEventListener in app.js corresponds to a known event type."""
        from autopilot.web.routes.sse import VALID_EVENT_TYPES

        # Extract event types from source.addEventListener('...') calls only —
        # this excludes DOM-level listeners like document.addEventListener('DOMContentLoaded').
        sse_listeners = set(re.findall(r"source\.addEventListener\('([^']+)'", app_js_source))

        allowed = set(VALID_EVENT_TYPES) | self._LEGACY_SSE_LISTENERS
        orphaned = sorted(sse_listeners - allowed)
        assert not orphaned, (
            f"SSE listeners in app.js not in VALID_EVENT_TYPES or _LEGACY_SSE_LISTENERS: {orphaned}"
        )

    def test_legacy_listeners_exist_in_app_js(self, app_js_source: str) -> None:
        """Every _LEGACY_SSE_LISTENERS entry has a source.addEventListener in app.js."""
        sse_listeners = set(re.findall(r"source\.addEventListener\('([^']+)'", app_js_source))

        stale = sorted(self._LEGACY_SSE_LISTENERS - sse_listeners)
        assert not stale, (
            f"_LEGACY_SSE_LISTENERS entries not found in app.js: {stale}. "
            "Remove them from the allowlist."
        )

    def test_legacy_and_valid_sets_disjoint(self) -> None:
        """_LEGACY_SSE_LISTENERS and VALID_EVENT_TYPES must not overlap."""
        from autopilot.web.routes.sse import VALID_EVENT_TYPES

        overlap = sorted(self._LEGACY_SSE_LISTENERS & set(VALID_EVENT_TYPES))
        assert not overlap, (
            f"Event types in both _LEGACY_SSE_LISTENERS and VALID_EVENT_TYPES: {overlap}. "
            "Promote to VALID_EVENT_TYPES and remove from legacy, or vice versa."
        )
