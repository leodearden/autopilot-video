"""Tests for the FastAPI web application skeleton."""

from __future__ import annotations

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


def _extract_listener_body(js_source: str, event_type: str) -> str:
    """Extract the function body of an addEventListener(event_type, ...) block.

    Returns the full brace-delimited body of the listener callback.
    Raises AssertionError if the listener is not found.
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


def _read_app_js() -> str:
    """Read the app.js source file."""
    js_path = Path(__file__).resolve().parent.parent / "autopilot" / "web" / "static" / "app.js"
    return js_path.read_text()


class TestDashboardSSEGateHandlers:
    """Tests for gate event SSE handlers in app.js."""

    def test_gate_waiting_handler_exists_with_try_catch_and_refresh(self) -> None:
        """app.js has a gate_waiting listener with try/catch and refreshStageCard."""
        js_source = _read_app_js()
        body = _extract_listener_body(js_source, "gate_waiting")

        assert "try" in body, "try block not found in gate_waiting listener"
        assert "catch" in body, "catch block not found in gate_waiting listener"
        assert "refreshStageCard" in body, "refreshStageCard not found in gate_waiting listener"
        assert "showToast" in body, "showToast not found in gate_waiting listener"

    def test_gate_approved_handler_exists_with_try_catch_and_refresh(self) -> None:
        """app.js has a gate_approved listener with try/catch and refreshStageCard."""
        js_source = _read_app_js()
        body = _extract_listener_body(js_source, "gate_approved")

        assert "try" in body, "try block not found in gate_approved listener"
        assert "catch" in body, "catch block not found in gate_approved listener"
        assert "refreshStageCard" in body, "refreshStageCard not found in gate_approved listener"
        assert "showToast" in body, "showToast not found in gate_approved listener"

    def test_gate_skipped_handler_exists_with_try_catch_and_refresh(self) -> None:
        """app.js has a gate_skipped listener with try/catch and refreshStageCard."""
        js_source = _read_app_js()
        body = _extract_listener_body(js_source, "gate_skipped")

        assert "try" in body, "try block not found in gate_skipped listener"
        assert "catch" in body, "catch block not found in gate_skipped listener"
        assert "refreshStageCard" in body, "refreshStageCard not found in gate_skipped listener"
        assert "showToast" in body, "showToast not found in gate_skipped listener"


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


class TestDashboardSSEErrorHandlers:
    """Tests for stage_error, run_completed, run_failed SSE handlers in app.js."""

    def test_stage_error_handler_exists_with_try_catch_and_toast(self) -> None:
        """app.js has a stage_error listener with try/catch, refreshStageCard, and showToast."""
        js_source = _read_app_js()
        body = _extract_listener_body(js_source, "stage_error")

        assert "try" in body, "try block not found in stage_error listener"
        assert "catch" in body, "catch block not found in stage_error listener"
        assert "refreshStageCard" in body, "refreshStageCard not found in stage_error listener"
        assert "showToast" in body, "showToast not found in stage_error listener"

    def test_run_completed_handler_exists_with_try_catch_and_toast(self) -> None:
        """app.js has a run_completed listener with try/catch and showToast."""
        js_source = _read_app_js()
        body = _extract_listener_body(js_source, "run_completed")

        assert "try" in body, "try block not found in run_completed listener"
        assert "catch" in body, "catch block not found in run_completed listener"
        assert "showToast" in body, "showToast not found in run_completed listener"

    def test_run_failed_handler_exists_with_try_catch_and_toast(self) -> None:
        """app.js has a run_failed listener with try/catch and showToast."""
        js_source = _read_app_js()
        body = _extract_listener_body(js_source, "run_failed")

        assert "try" in body, "try block not found in run_failed listener"
        assert "catch" in body, "catch block not found in run_failed listener"
        assert "showToast" in body, "showToast not found in run_failed listener"
