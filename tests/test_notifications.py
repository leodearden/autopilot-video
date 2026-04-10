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

    def test_setup_listens_for_notification_event(self) -> None:
        """setupNotificationSSE listens for the unified 'notification' event type."""
        content = _APP_JS.read_text()
        func_start = content.find("function setupNotificationSSE")
        assert func_start != -1, "setupNotificationSSE function not found"
        func_body = content[func_start:]
        # Find the end of the function (next top-level function or end of file)
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]
        assert "addEventListener('notification'" in func_body or \
               'addEventListener("notification"' in func_body, \
               "setupNotificationSSE should listen for 'notification' event"

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

    def test_notification_handler_calls_show_toast(self) -> None:
        """The notification event handler inside setupNotificationSSE calls showToast."""
        content = _APP_JS.read_text()
        func_start = content.find("function setupNotificationSSE")
        assert func_start != -1
        func_body = content[func_start:]
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]
        # Find the notification listener and verify showToast is called within it
        notif_idx = func_body.find("notification")
        assert notif_idx != -1, "No 'notification' listener in setupNotificationSSE"
        after_notif = func_body[notif_idx:]
        assert "showToast" in after_notif, (
            "showToast should be called within notification handler"
        )

    def test_bell_click_clears_badge(self) -> None:
        """setupNotificationSSE wires bell click to clear the unread count."""
        content = _APP_JS.read_text()
        func_start = content.find("function setupNotificationSSE")
        assert func_start != -1
        func_body = content[func_start:]
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]
        assert "notification-bell" in func_body, (
            "setupNotificationSSE should reference 'notification-bell' element"
        )
        assert "addEventListener('click'" in func_body or \
               'addEventListener("click"' in func_body, \
               "setupNotificationSSE should add a click handler for the bell"
        assert "updateNotificationBadge(0)" in func_body, (
            "Bell click handler should reset badge to 0"
        )

    def test_notification_handler_guards_data_message(self) -> None:
        """The notification handler must guard data.message with a typeof check
        and provide a '(no message)' fallback so that undefined/null values
        never reach showToast or Notification."""
        content = _APP_JS.read_text()
        func_start = content.find("function setupNotificationSSE")
        assert func_start != -1
        func_body = content[func_start:]
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]

        # There must be a typeof check on data.message
        assert "typeof data.message" in func_body, (
            "notification handler should check typeof data.message"
        )
        # There must be a '(no message)' fallback string
        assert "(no message)" in func_body, (
            "notification handler should have a '(no message)' fallback"
        )
        # showToast should NOT be called with raw data.message
        # Instead it should use the guarded variable (msg)
        notif_listener = func_body[func_body.find("notification"):]
        assert "showToast(data.message" not in notif_listener, (
            "showToast should use guarded msg variable, not raw data.message"
        )

    def test_request_permission_chains_then_with_granted_check(self) -> None:
        """requestPermission() must be chained with .then() that creates a
        Notification only when permission is 'granted'."""
        content = _APP_JS.read_text()
        func_start = content.find("function setupNotificationSSE")
        assert func_start != -1
        func_body = content[func_start:]
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]

        # requestPermission must be followed by .then(
        assert "requestPermission().then(" in func_body, (
            "requestPermission should be chained with .then() callback"
        )
        # The .then callback must check for 'granted'
        perm_idx = func_body.find("requestPermission().then(")
        assert perm_idx != -1
        after_perm = func_body[perm_idx:]
        assert "'granted'" in after_perm or '"granted"' in after_perm, (
            ".then() callback should check permission === 'granted'"
        )

    def test_permission_requested_flag_guards_request_permission(self) -> None:
        """A module-level _permissionRequested flag must exist before
        setupNotificationSSE and be checked before calling requestPermission()
        to prevent stacking permission dialogs on rapid SSE events."""
        content = _APP_JS.read_text()

        # _permissionRequested must be declared at module level,
        # before setupNotificationSSE
        flag_idx = content.find("_permissionRequested")
        assert flag_idx != -1, "_permissionRequested flag not found in app.js"
        func_idx = content.find("function setupNotificationSSE")
        assert flag_idx < func_idx, (
            "_permissionRequested should be declared before setupNotificationSSE"
        )

        # Inside setupNotificationSSE, the flag must guard requestPermission
        func_body = content[func_idx:]
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]
        assert "_permissionRequested" in func_body, (
            "setupNotificationSSE should check _permissionRequested flag"
        )
        # The flag should be set to true before/after calling requestPermission
        assert "_permissionRequested = true" in func_body, (
            "setupNotificationSSE should set _permissionRequested = true"
        )

    def test_sse_reconnect_gap_documented(self) -> None:
        """connectSSE should have a comment documenting the SSE reconnect gap
        and mentioning Last-Event-ID as the robust fix."""
        content = _APP_JS.read_text()
        func_start = content.find("function connectSSE")
        assert func_start != -1
        func_body = content[func_start:]
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]

        assert "reconnect" in func_body.lower() or "reconnection" in func_body.lower(), (
            "connectSSE should document the reconnect gap"
        )
        assert "Last-Event-ID" in func_body, (
            "connectSSE should mention Last-Event-ID as the robust fix"
        )

    def test_notification_handler_guards_data_object_before_side_effects(self) -> None:
        """The notification handler must guard that data is a non-null object
        immediately after JSON.parse, BEFORE unreadCount++ or updateNotificationBadge.
        This prevents inconsistent UI state when JSON.parse yields null
        (e.g. server sends 'data: null')."""
        content = _APP_JS.read_text()
        func_start = content.find("function setupNotificationSSE")
        assert func_start != -1
        func_body = content[func_start:]
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]

        # There must be a typeof data !== 'object' guard
        assert "typeof data !== 'object'" in func_body or \
               'typeof data !== "object"' in func_body, (
            "notification handler should check typeof data !== 'object'"
        )
        # There must be a !data check (for null)
        assert "!data" in func_body, (
            "notification handler should check !data for null"
        )
        # The guard must appear BEFORE unreadCount++ to prevent side effects
        guard_idx = func_body.find("typeof data")
        unread_idx = func_body.find("unreadCount++")
        assert guard_idx != -1 and unread_idx != -1, (
            "Both guard and unreadCount++ should exist"
        )
        assert guard_idx < unread_idx, (
            "data object guard must appear BEFORE unreadCount++ "
            "to prevent badge increment on invalid payloads"
        )

    def test_setup_notification_sse_jsdoc_reflects_unified_model(self) -> None:
        """JSDoc for setupNotificationSSE must describe the unified notification
        model and must not contain stale gate_waiting / gates-are-approved references."""
        content = _APP_JS.read_text()
        func_idx = content.find("function setupNotificationSSE")
        assert func_idx != -1, "setupNotificationSSE not found in app.js"
        # Walk backward from the function declaration to isolate its JSDoc block
        jsdoc_end = func_idx  # exclusive upper bound (the function line itself)
        jsdoc_start = content.rfind("/**", 0, jsdoc_end)
        assert jsdoc_start != -1, "No JSDoc block found before setupNotificationSSE"
        jsdoc_block = content[jsdoc_start:jsdoc_end]

        # Stale phrases must be absent
        assert "gate_waiting" not in jsdoc_block, (
            "JSDoc should not reference stale gate_waiting event"
        )
        assert "gates are approved or skipped" not in jsdoc_block, (
            "JSDoc should not describe stale gate approval/skip behavior"
        )

        # Phrases describing the unified notification model must be present
        assert "unread" in jsdoc_block.lower(), (
            "JSDoc should mention unread badge behavior"
        )
        assert "notification" in jsdoc_block.lower(), (
            "JSDoc should mention the unified notification event"
        )
        assert "bell" in jsdoc_block.lower(), (
            "JSDoc should mention bell click resetting the count"
        )

    def test_connectsse_no_duplicate_notification_listener(self) -> None:
        """connectSSE must not duplicate the notification listener."""
        content = _APP_JS.read_text()
        func_start = content.find("function connectSSE")
        assert func_start != -1
        # Find end of connectSSE function
        func_body = content[func_start:]
        next_func = func_body.find("\nfunction ", 1)
        if next_func != -1:
            func_body = func_body[:next_func]
        has_listener = (
            "addEventListener('notification'" in func_body
            or 'addEventListener("notification"' in func_body
        )
        assert not has_listener, (
            "connectSSE should NOT contain a notification"
            " listener (handled by setupNotificationSSE)"
        )
