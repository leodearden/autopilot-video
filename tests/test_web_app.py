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


@pytest.fixture
def client(app: FastAPI):
    """Create a TestClient for the app."""
    from starlette.testclient import TestClient

    return TestClient(app)


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


def _extract_listener_body(js_source: str, event_type: str) -> str:
    """Extract the body of a source.addEventListener callback by brace-matching.

    Returns the outermost { ... } block of the callback function for the given
    event type.  Raises AssertionError if the listener is not found.
    """
    marker = f"addEventListener('{event_type}'"
    listener_start = js_source.find(marker)
    assert listener_start != -1, f"{event_type} event listener not found in app.js"

    func_start = js_source.find("function", listener_start)
    assert func_start != -1, f"function keyword not found in {event_type} listener"

    body_start = js_source.find("{", func_start)
    assert body_start != -1, f"opening brace not found in {event_type} listener"

    brace_depth = 0
    body_end = body_start
    for i in range(body_start, len(js_source)):
        if js_source[i] == "{":
            brace_depth += 1
        elif js_source[i] == "}":
            brace_depth -= 1
            if brace_depth == 0:
                body_end = i + 1
                break

    return js_source[body_start:body_end]


class TestSSEErrorHandling:
    """Tests for SSE notification error handling in app.js."""

    def test_sse_notification_listener_has_try_catch(self) -> None:
        """The SSE notification listener wraps JSON.parse in try/catch."""
        js_path = Path(__file__).resolve().parent.parent / "autopilot" / "web" / "static" / "app.js"
        js_source = js_path.read_text()

        # Find the notification listener block
        listener_start = js_source.find("addEventListener('notification'")
        assert listener_start != -1, "notification event listener not found in app.js"

        # Extract the listener function body (from the opening brace after 'function'
        # to its matching closing brace)
        func_start = js_source.find("function", listener_start)
        assert func_start != -1, "function keyword not found in notification listener"

        body_start = js_source.find("{", func_start)
        assert body_start != -1, "opening brace not found in notification listener"

        # Extract the listener body
        brace_depth = 0
        body_end = body_start
        for i in range(body_start, len(js_source)):
            if js_source[i] == "{":
                brace_depth += 1
            elif js_source[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    body_end = i + 1
                    break

        listener_body = js_source[body_start:body_end]

        # Assert try/catch wraps the JSON.parse
        assert "try" in listener_body, "try block not found in notification listener"
        assert "catch" in listener_body, "catch block not found in notification listener"
        assert "console.error" in listener_body, (
            "console.error not found in notification listener catch block"
        )


class TestSSEHandlerFactory:
    """Tests for the makeStageHandler factory function in app.js."""

    def test_make_stage_handler_defined(self) -> None:
        """makeStageHandler function is defined in app.js."""
        js_source = _read_app_js()
        assert "function makeStageHandler" in js_source, (
            "makeStageHandler factory function not found in app.js"
        )

    def test_make_stage_handler_has_try_catch(self) -> None:
        """makeStageHandler contains try/catch error handling."""
        js_source = _read_app_js()
        # Extract the makeStageHandler function body
        func_start = js_source.find("function makeStageHandler")
        assert func_start != -1, "makeStageHandler not found"

        body_start = js_source.find("{", func_start)
        brace_depth = 0
        body_end = body_start
        for i in range(body_start, len(js_source)):
            if js_source[i] == "{":
                brace_depth += 1
            elif js_source[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    body_end = i + 1
                    break

        factory_body = js_source[body_start:body_end]
        assert "try" in factory_body, "try block not found in makeStageHandler"
        assert "catch" in factory_body, "catch block not found in makeStageHandler"
        assert "console.error" in factory_body, (
            "console.error not found in makeStageHandler catch block"
        )

    def test_make_stage_handler_has_console_warn_for_missing_stage(self) -> None:
        """makeStageHandler logs console.warn when data.stage is falsy."""
        js_source = _read_app_js()
        func_start = js_source.find("function makeStageHandler")
        assert func_start != -1, "makeStageHandler not found"

        body_start = js_source.find("{", func_start)
        brace_depth = 0
        body_end = body_start
        for i in range(body_start, len(js_source)):
            if js_source[i] == "{":
                brace_depth += 1
            elif js_source[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    body_end = i + 1
                    break

        factory_body = js_source[body_start:body_end]
        assert "console.warn" in factory_body, (
            "console.warn for missing stage field not found in makeStageHandler"
        )

    def test_make_stage_handler_calls_refresh_stage_card(self) -> None:
        """makeStageHandler calls refreshStageCard when stage is present."""
        js_source = _read_app_js()
        func_start = js_source.find("function makeStageHandler")
        assert func_start != -1, "makeStageHandler not found"

        body_start = js_source.find("{", func_start)
        brace_depth = 0
        body_end = body_start
        for i in range(body_start, len(js_source)):
            if js_source[i] == "{":
                brace_depth += 1
            elif js_source[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    body_end = i + 1
                    break

        factory_body = js_source[body_start:body_end]
        assert "refreshStageCard" in factory_body, (
            "refreshStageCard call not found in makeStageHandler"
        )

    def test_setup_dashboard_sse_uses_factory(self) -> None:
        """setupDashboardSSE uses makeStageHandler for all 3 event types."""
        js_source = _read_app_js()

        # Extract setupDashboardSSE body
        func_start = js_source.find("function setupDashboardSSE")
        assert func_start != -1, "setupDashboardSSE not found"

        body_start = js_source.find("{", func_start)
        brace_depth = 0
        body_end = body_start
        for i in range(body_start, len(js_source)):
            if js_source[i] == "{":
                brace_depth += 1
            elif js_source[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    body_end = i + 1
                    break

        dashboard_body = js_source[body_start:body_end]

        for event_type in ("stage_started", "stage_completed", "job_progress"):
            assert f"makeStageHandler('{event_type}'" in dashboard_body, (
                f"setupDashboardSSE does not use makeStageHandler for '{event_type}'"
            )

    def test_stage_error_uses_factory(self) -> None:
        """stage_error handler uses makeStageHandler with toast parameters."""
        js_source = _read_app_js()
        assert "makeStageHandler('stage_error'" in js_source, (
            "stage_error handler does not use makeStageHandler"
        )


    def test_make_stage_handler_shows_toast_on_missing_stage(self) -> None:
        """makeStageHandler shows a fallback toast when stage is missing and toastMsg is provided.

        This catches the regression where stage_error events with no stage field
        silently dropped the error toast from the UI.  The else branch (missing-stage
        path) must call showToast with '{stage}' replaced by 'unknown' so that
        error-level events always surface to the user.
        """
        js_source = _read_app_js()

        # Extract the full makeStageHandler function body
        func_start = js_source.find("function makeStageHandler")
        assert func_start != -1, "makeStageHandler not found"

        body_start = js_source.find("{", func_start)
        brace_depth = 0
        body_end = body_start
        for i in range(body_start, len(js_source)):
            if js_source[i] == "{":
                brace_depth += 1
            elif js_source[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    body_end = i + 1
                    break

        factory_body = js_source[body_start:body_end]

        # Locate the else branch (the missing-stage path)
        else_pos = factory_body.find("} else {")
        assert else_pos != -1, "else branch not found in makeStageHandler"

        # Extract the else block body via brace-matching
        else_brace_start = factory_body.find("{", else_pos + 1)
        assert else_brace_start != -1, "opening brace of else block not found"

        brace_depth = 0
        else_end = else_brace_start
        for i in range(else_brace_start, len(factory_body)):
            if factory_body[i] == "{":
                brace_depth += 1
            elif factory_body[i] == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    else_end = i + 1
                    break

        else_body = factory_body[else_brace_start:else_end]

        # The else branch must call showToast for fallback toast on missing stage
        assert "showToast" in else_body, (
            "showToast call not found in makeStageHandler else branch (missing-stage path). "
            "Error-level SSE events with no stage field should still show a fallback toast."
        )


class TestSSERunHandlerRobustness:
    """Tests that run_completed and run_failed handlers have try/catch."""

    def test_run_completed_has_try_catch(self) -> None:
        """run_completed handler wraps its body in try/catch."""
        js_source = _read_app_js()
        body = _extract_listener_body(js_source, "run_completed")
        assert "try" in body, "try block not found in run_completed handler"
        assert "catch" in body, "catch block not found in run_completed handler"
        assert "console.error" in body, "console.error not found in run_completed catch block"

    def test_run_failed_has_try_catch(self) -> None:
        """run_failed handler wraps its body in try/catch."""
        js_source = _read_app_js()
        body = _extract_listener_body(js_source, "run_failed")
        assert "try" in body, "try block not found in run_failed handler"
        assert "catch" in body, "catch block not found in run_failed handler"
        assert "console.error" in body, "console.error not found in run_failed catch block"


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
    # 'notification' handler exists but the server never emits this event type.
    _LEGACY_SSE_LISTENERS: frozenset[str] = frozenset({"notification"})

    def test_all_dashboard_event_types_handled(self) -> None:
        """Every VALID_EVENT_TYPE (except job-level and server-only) has a listener in app.js."""
        from autopilot.web.routes.sse import VALID_EVENT_TYPES

        js_source = _read_app_js()

        dashboard_event_types = (
            set(VALID_EVENT_TYPES) - self._JOB_LEVEL_EVENTS - self._SERVER_ONLY_EVENTS
        )
        missing = []
        for event_type in sorted(dashboard_event_types):
            marker = f"source.addEventListener('{event_type}'"
            if marker not in js_source:
                missing.append(event_type)

        assert not missing, (
            f"Dashboard-relevant event types missing addEventListener in app.js: {missing}"
        )

    def test_no_orphaned_sse_listeners(self) -> None:
        """Every source.addEventListener in app.js corresponds to a known event type."""
        from autopilot.web.routes.sse import VALID_EVENT_TYPES

        js_source = _read_app_js()

        # Extract event types from source.addEventListener('...') calls only —
        # this excludes DOM-level listeners like document.addEventListener('DOMContentLoaded').
        sse_listeners = set(re.findall(r"source\.addEventListener\('([^']+)'", js_source))

        allowed = set(VALID_EVENT_TYPES) | self._LEGACY_SSE_LISTENERS
        orphaned = sorted(sse_listeners - allowed)
        assert not orphaned, (
            f"SSE listeners in app.js not in VALID_EVENT_TYPES or _LEGACY_SSE_LISTENERS: {orphaned}"
        )

    def test_legacy_listeners_exist_in_app_js(self) -> None:
        """Every _LEGACY_SSE_LISTENERS entry has a source.addEventListener in app.js."""
        js_source = _read_app_js()
        sse_listeners = set(re.findall(r"source\.addEventListener\('([^']+)'", js_source))

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
