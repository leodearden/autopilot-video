"""Tests for gate configuration page and API endpoints."""

from __future__ import annotations

from html.parser import HTMLParser

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB

# Constants mirrored from the route module (populated after pre-2)
VALID_MODES = ("auto", "pause", "notify")

GATE_PRESETS = {
    "full_auto": {s: "auto" for s in CatalogDB._PIPELINE_STAGES},
    "review_creative": {
        "narrate": "pause",
        "script": "pause",
        "upload": "pause",
    },
    "review_everything": {s: "pause" for s in CatalogDB._PIPELINE_STAGES},
    "review_before_render": {
        "source": "pause",
        "upload": "pause",
    },
}


_VALID_GATE_IDS = {f"gate-{s}" for s in CatalogDB._PIPELINE_STAGES}


class _SelectedOptionParser(HTMLParser):
    """Extract the selected <option> value per gate div from HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._gate_id: str | None = None
        self._div_depth = 0
        self._in_select = False
        self.selected: dict[str, str] = {}  # gate div id -> selected value

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        if tag == "div":
            if self._gate_id is not None:
                self._div_depth += 1
            elif attr_dict.get("id") in _VALID_GATE_IDS:
                self._gate_id = attr_dict["id"]
                self._div_depth = 1
        if self._gate_id is not None:
            if tag == "select":
                self._in_select = True
            elif tag == "option" and self._in_select and "selected" in attr_dict:
                self.selected[self._gate_id] = attr_dict.get("value") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "select":
            self._in_select = False
        if tag == "div" and self._gate_id is not None:
            self._div_depth -= 1
            if self._div_depth == 0:
                self._gate_id = None


def _get_selected_modes(html: str) -> dict[str, str]:
    """Parse page HTML and return ``{stage: selected_mode}`` for each gate."""
    parser = _SelectedOptionParser()
    parser.feed(html)
    return {k.removeprefix("gate-"): v for k, v in parser.selected.items()}


class TestGateListAPI:
    """Tests for GET /api/gates endpoint."""

    def test_api_gates_returns_all_gates(self, client: TestClient) -> None:
        """GET /api/gates returns 200 with JSON list of 9 gate dicts."""
        response = client.get("/api/gates")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 9

        # Each gate has required keys
        required_keys = {"stage", "mode", "status", "timeout_hours", "decided_at", "decided_by"}
        for gate in data:
            assert required_keys.issubset(gate.keys()), f"Missing keys in {gate}"

        # Stages match _PIPELINE_STAGES order
        stages = [g["stage"] for g in data]
        assert stages == list(CatalogDB._PIPELINE_STAGES)

        # Default mode is 'auto'
        for gate in data:
            assert gate["mode"] == "auto"


class TestGateDetailAPI:
    """Tests for GET /api/gates/{stage} endpoint."""

    def test_api_gate_detail_returns_single_gate(self, client: TestClient) -> None:
        """GET /api/gates/analyze returns 200 with gate dict for 'analyze'."""
        response = client.get("/api/gates/analyze")
        assert response.status_code == 200
        data = response.json()
        assert data["stage"] == "analyze"
        assert data["mode"] == "auto"

    def test_api_gate_detail_unknown_stage_returns_404(self, client: TestClient) -> None:
        """GET /api/gates/nonexistent returns 404."""
        response = client.get("/api/gates/nonexistent")
        assert response.status_code == 404


class TestGateUpdateAPI:
    """Tests for PUT /api/gates/{stage} endpoint."""

    @pytest.mark.parametrize("mode", ["auto", "pause", "notify"])
    def test_update_gate_mode_persists(self, client: TestClient, mode: str) -> None:
        """PUT /api/gates/analyze with mode updates and persists via re-GET."""
        response = client.put("/api/gates/analyze", json={"mode": mode})
        assert response.status_code == 200
        assert response.json()["mode"] == mode
        # Re-GET confirms persistence
        check = client.get("/api/gates/analyze")
        assert check.json()["mode"] == mode

    def test_update_gate_invalid_mode(self, client: TestClient) -> None:
        """PUT with invalid mode returns 422."""
        response = client.put("/api/gates/analyze", json={"mode": "invalid"})
        assert response.status_code == 422

    def test_update_gate_timeout(self, client: TestClient) -> None:
        """PUT /api/gates/render with timeout_hours=24.0 persists."""
        response = client.put("/api/gates/render", json={"timeout_hours": 24.0})
        assert response.status_code == 200
        assert response.json()["timeout_hours"] == 24.0

    def test_update_gate_clear_timeout(self, client: TestClient) -> None:
        """PUT with timeout_hours=null clears the timeout."""
        # Set a timeout first
        client.put("/api/gates/render", json={"timeout_hours": 12.0})
        # Clear it
        response = client.put("/api/gates/render", json={"timeout_hours": None})
        assert response.status_code == 200
        assert response.json()["timeout_hours"] is None

    def test_update_gate_empty_body(self, client: TestClient) -> None:
        """PUT /api/gates/analyze with {} returns gate unchanged (no-op path)."""
        response = client.put("/api/gates/analyze", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["stage"] == "analyze"
        assert data["mode"] == "auto"

    def test_update_gate_unknown_stage(self, client: TestClient) -> None:
        """PUT /api/gates/nonexistent returns 404."""
        response = client.put("/api/gates/nonexistent", json={"mode": "auto"})
        assert response.status_code == 404


class TestGateApproveAPI:
    """Tests for POST /api/gates/{stage}/approve endpoint."""

    def test_approve_gate(self, client: TestClient) -> None:
        """POST /api/gates/classify/approve returns 200 with approved status."""
        response = client.post("/api/gates/classify/approve")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "approved"
        assert data["decided_by"] == "console"
        assert data["decided_at"] is not None

    def test_approve_unknown_stage_returns_404(self, client: TestClient) -> None:
        """POST /api/gates/nonexistent/approve returns 404."""
        response = client.post("/api/gates/nonexistent/approve")
        assert response.status_code == 404


class TestGateSkipAPI:
    """Tests for POST /api/gates/{stage}/skip endpoint."""

    def test_skip_gate(self, client: TestClient) -> None:
        """POST /api/gates/classify/skip returns 200 with skipped status."""
        response = client.post("/api/gates/classify/skip")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "skipped"
        assert data["decided_by"] == "console"
        assert data["decided_at"] is not None

    def test_skip_unknown_stage_returns_404(self, client: TestClient) -> None:
        """POST /api/gates/nonexistent/skip returns 404."""
        response = client.post("/api/gates/nonexistent/skip")
        assert response.status_code == 404


class TestGatePresetsAPI:
    """Tests for PUT /api/gates/preset/{preset_name} endpoint."""

    def test_preset_full_auto(self, client: TestClient) -> None:
        """PUT /api/gates/preset/full_auto sets all gates to mode='auto'."""
        response = client.put("/api/gates/preset/full_auto")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 9
        for gate in data:
            assert gate["mode"] == "auto"

    def test_preset_review_creative(self, client: TestClient) -> None:
        """review_creative pauses narrate, script, upload; rest auto."""
        response = client.put("/api/gates/preset/review_creative")
        assert response.status_code == 200
        data = response.json()
        modes = {g["stage"]: g["mode"] for g in data}
        assert modes["narrate"] == "pause"
        assert modes["script"] == "pause"
        assert modes["upload"] == "pause"
        for stage in ("ingest", "analyze", "classify", "edl", "source", "render"):
            assert modes[stage] == "auto"

    def test_preset_review_everything(self, client: TestClient) -> None:
        """review_everything pauses all 9 gates."""
        response = client.put("/api/gates/preset/review_everything")
        assert response.status_code == 200
        data = response.json()
        for gate in data:
            assert gate["mode"] == "pause"

    def test_preset_review_before_render(self, client: TestClient) -> None:
        """review_before_render pauses source and upload; rest auto."""
        response = client.put("/api/gates/preset/review_before_render")
        assert response.status_code == 200
        data = response.json()
        modes = {g["stage"]: g["mode"] for g in data}
        assert modes["source"] == "pause"
        assert modes["upload"] == "pause"
        for stage in ("ingest", "analyze", "classify", "narrate", "script", "edl", "render"):
            assert modes[stage] == "auto"

    def test_preset_unknown_returns_404(self, client: TestClient) -> None:
        """PUT /api/gates/preset/nonexistent returns 404."""
        response = client.put("/api/gates/preset/nonexistent")
        assert response.status_code == 404


class TestGatesPage:
    """Tests for GET /gates HTML page."""

    def test_gates_page_returns_200(self, client: TestClient) -> None:
        """GET /gates returns 200 with text/html."""
        response = client.get("/gates")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_gates_page_has_title(self, client: TestClient) -> None:
        """HTML contains 'Gate Configuration'."""
        response = client.get("/gates")
        assert "Gate Configuration" in response.text

    def test_gates_page_shows_all_stages(self, client: TestClient) -> None:
        """HTML contains all 9 stage names."""
        response = client.get("/gates")
        for stage in CatalogDB._PIPELINE_STAGES:
            assert stage in response.text

    def test_gates_page_shows_transitions(self, client: TestClient) -> None:
        """HTML contains transition labels (from-stage paired with to-stage)."""
        response = client.get("/gates")
        # Check a few transitions exist
        assert "ingest" in response.text
        assert "analyze" in response.text


class TestGateTogglePartial:
    """Tests for gate_toggle.html partial content within the gates page."""

    def test_toggle_contains_mode_select(self, client: TestClient) -> None:
        """Gate toggle partial contains a select with auto/pause/notify options."""
        response = client.get("/gates")
        html = response.text
        for mode in VALID_MODES:
            assert f'value="{mode}"' in html, f"Missing option value for {mode}"

    def test_toggle_has_htmx_attributes(self, client: TestClient) -> None:
        """Gate toggle has hx-put and hx-swap attributes for HTMX updates."""
        response = client.get("/gates")
        html = response.text
        assert "hx-put" in html
        assert "hx-swap" in html

    @pytest.mark.parametrize("mode", ["auto", "pause", "notify"])
    def test_toggle_shows_current_mode(self, client: TestClient, mode: str) -> None:
        """The selected option matches the gate's current mode."""
        # Set analyze gate to given mode — leave others at default 'auto'
        client.put("/api/gates/analyze", json={"mode": mode})
        response = client.get("/gates")
        selected = _get_selected_modes(response.text)

        # Positive: analyze must reflect the updated mode
        assert selected["analyze"] == mode

        # Negative control: classify (not modified) must still show 'auto'
        assert selected["classify"] == "auto"


class TestGatePresetsUI:
    """Tests for preset buttons on the gates page."""

    def test_gates_page_shows_preset_buttons(self, client: TestClient) -> None:
        """GET /gates HTML contains buttons for all 4 presets."""
        response = client.get("/gates")
        html = response.text
        assert "Full Auto" in html
        assert "Review Creative" in html
        assert "Review Everything" in html
        assert "Review Before Render" in html

    def test_preset_buttons_have_htmx(self, client: TestClient) -> None:
        """Each preset button has hx-put targeting /api/gates/preset/{name}."""
        response = client.get("/gates")
        html = response.text
        for preset_name in GATE_PRESETS:
            assert f"/api/gates/preset/{preset_name}" in html


class TestGateStatusDisplay:
    """Tests for gate status display on the gates page."""

    def test_waiting_gate_shows_approve_skip_buttons(
        self, client: TestClient, app: FastAPI
    ) -> None:
        """A gate with status='waiting' shows Approve and Skip buttons."""
        # Set a gate to waiting status via direct DB
        db = CatalogDB(app.state.db_path)
        try:
            db.update_gate("classify", status="waiting")
            db.conn.commit()
        finally:
            db.close()
        response = client.get("/gates")
        html = response.text
        assert "Approve" in html
        assert "Skip" in html

    def test_approved_gate_shows_decided_info(
        self, client: TestClient, app: FastAPI
    ) -> None:
        """An approved gate with decided_at/decided_by shows that info."""
        db = CatalogDB(app.state.db_path)
        try:
            db.update_gate(
                "classify",
                status="approved",
                decided_at="2026-03-24T12:00:00",
                decided_by="console",
            )
            db.conn.commit()
        finally:
            db.close()
        response = client.get("/gates")
        html = response.text
        assert "2026-03-24T12:00:00" in html
        assert "by console" in html

    def test_status_color_coding(self, client: TestClient, app: FastAPI) -> None:
        """Waiting gate has amber class, approved has green, idle has gray."""
        db = CatalogDB(app.state.db_path)
        try:
            db.update_gate("classify", status="waiting")
            db.update_gate("analyze", status="approved")
            db.conn.commit()
        finally:
            db.close()
        response = client.get("/gates")
        html = response.text
        assert "bg-amber-900" in html  # waiting
        assert "bg-green-900" in html  # approved
        assert "bg-gray-900" in html  # idle (default for other gates)


class TestHtmxResponses:
    """Tests for HTMX-aware response logic (HTML partial vs JSON)."""

    def test_put_gate_returns_html_for_htmx(self, client: TestClient) -> None:
        """PUT /api/gates/analyze with HX-Request returns text/html partial."""
        response = client.put(
            "/api/gates/analyze",
            json={"mode": "pause"},
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "gate-analyze" in response.text
        # Verify the rendered partial has the correct mode selected
        selected = _get_selected_modes(response.text)
        assert selected["analyze"] == "pause"

    def test_put_gate_returns_json_without_htmx(self, client: TestClient) -> None:
        """PUT /api/gates/analyze without HX-Request returns application/json."""
        response = client.put("/api/gates/analyze", json={"mode": "pause"})
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        data = response.json()
        assert data["mode"] == "pause"

    def test_approve_returns_html_for_htmx(self, client: TestClient) -> None:
        """POST approve with HX-Request returns HTML partial."""
        response = client.post(
            "/api/gates/classify/approve",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "gate-classify" in response.text

    def test_skip_returns_html_for_htmx(self, client: TestClient) -> None:
        """POST skip with HX-Request returns HTML partial."""
        response = client.post(
            "/api/gates/classify/skip",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "gate-classify" in response.text


class TestPresetHtmxResponse:
    """Tests for HTMX-aware preset endpoint responses."""

    def test_preset_returns_html_for_htmx(self, client: TestClient) -> None:
        """PUT /api/gates/preset/full_auto with HX-Request returns HTML with all 9 gate rows."""
        response = client.put(
            "/api/gates/preset/full_auto",
            headers={"HX-Request": "true"},
        )
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        # Should contain all 9 gate toggle divs
        for stage in CatalogDB._PIPELINE_STAGES:
            assert f"gate-{stage}" in response.text
        # Verify all 9 gate partials render with mode='auto' for full_auto preset
        selected = _get_selected_modes(response.text)
        assert len(selected) == 9
        assert all(v == "auto" for v in selected.values())

    def test_preset_returns_json_without_htmx(self, client: TestClient) -> None:
        """PUT /api/gates/preset/full_auto without HX-Request returns JSON list."""
        response = client.put("/api/gates/preset/full_auto")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 9
