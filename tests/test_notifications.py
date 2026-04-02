"""Tests for notification bell wiring in app.js."""

from __future__ import annotations

from pathlib import Path

_APP_JS = Path(__file__).resolve().parent.parent / "autopilot" / "web" / "static" / "app.js"


class TestNotificationSSESetup:
    """Verify app.js wires up SSE for notification bell on all pages."""

    def test_sse_connects_on_all_pages(self) -> None:
        """connectSSE is called unconditionally (not only on dashboard)."""
        content = _APP_JS.read_text()
        # The SSE source should be created before the dashboard-grid check
        dom_ready_idx = content.find("DOMContentLoaded")
        assert dom_ready_idx != -1
        body = content[dom_ready_idx:]
        connect_idx = body.find("connectSSE")
        dashboard_check_idx = body.find("dashboard-grid")
        assert connect_idx != -1, "connectSSE not called in DOMContentLoaded"
        assert connect_idx < dashboard_check_idx, (
            "connectSSE should be called before dashboard-grid check"
        )

    def test_has_gate_waiting_handler(self) -> None:
        """app.js listens for gate_waiting SSE events."""
        content = _APP_JS.read_text()
        assert "gate_waiting" in content

    def test_has_gate_approved_handler(self) -> None:
        """app.js listens for gate_approved SSE events."""
        content = _APP_JS.read_text()
        assert "gate_approved" in content

    def test_has_gate_skipped_handler(self) -> None:
        """app.js listens for gate_skipped SSE events."""
        content = _APP_JS.read_text()
        assert "gate_skipped" in content

    def test_has_update_notification_badge(self) -> None:
        """app.js has an updateNotificationBadge function."""
        content = _APP_JS.read_text()
        assert "updateNotificationBadge" in content

    def test_notification_badge_shows_and_hides(self) -> None:
        """updateNotificationBadge references hidden class for toggling."""
        content = _APP_JS.read_text()
        func_start = content.find("function updateNotificationBadge")
        assert func_start != -1
        func_body = content[func_start:content.find("\n}", func_start) + 2]
        assert "hidden" in func_body
        assert "notification-badge" in func_body

    def test_stage_error_shows_toast(self) -> None:
        """stage_error events trigger a toast notification."""
        content = _APP_JS.read_text()
        error_idx = content.find("stage_error")
        assert error_idx != -1
        # After stage_error listener, showToast should be called
        block_end = content.find("});", error_idx)
        block = content[error_idx:block_end]
        assert "showToast" in block

    def test_run_completed_shows_toast(self) -> None:
        """run_completed events trigger a toast notification."""
        content = _APP_JS.read_text()
        assert "run_completed" in content
        completed_idx = content.find("run_completed")
        block_end = content.find("});", completed_idx)
        block = content[completed_idx:block_end]
        assert "showToast" in block
