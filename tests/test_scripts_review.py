"""Tests for the /review/scripts, /review/edit-plans, and /review/renders routes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from autopilot.db import CatalogDB
from autopilot.web.app import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "catalog.db")


@pytest.fixture
def db(db_path: str) -> CatalogDB:
    db = CatalogDB(db_path)
    db.conn.isolation_level = None
    return db


@pytest.fixture
def seeded_db(db: CatalogDB) -> CatalogDB:
    """Seed narratives, scripts, and edit plans."""
    db.init_default_gates()
    db.insert_narrative(
        "n-1", title="Morning Walk", status="scripted",
    )
    db.insert_narrative(
        "n-2", title="Sunset Drive", status="approved",
    )
    db.upsert_narrative_script(
        "n-1",
        json.dumps({
            "scenes": [
                {"title": "Opening", "description": "Wide shot of park"},
                {"title": "Walk", "description": "Follow subject"},
            ],
        }),
    )
    db.upsert_narrative_script(
        "n-2",
        json.dumps([
            {"title": "Scene 1", "description": "Car interior"},
        ]),
    )
    # Edit plan for n-1 only
    db.upsert_edit_plan("n-1", edl_json='{"clips": []}')
    return db


@pytest.fixture
def app(seeded_db: CatalogDB, db_path: str) -> FastAPI:
    return create_app(db_path)


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def empty_app(db: CatalogDB, db_path: str) -> FastAPI:
    db.init_default_gates()
    return create_app(db_path)


@pytest.fixture
def empty_client(empty_app: FastAPI) -> TestClient:
    return TestClient(empty_app)


# ---------------------------------------------------------------------------
# GET /review/scripts
# ---------------------------------------------------------------------------


class TestScriptsPage:
    """Tests for the script review page."""

    def test_returns_200_html(self, client: TestClient) -> None:
        resp = client.get("/review/scripts")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_shows_narrative_titles(self, client: TestClient) -> None:
        resp = client.get("/review/scripts")
        html = resp.text
        assert "Morning Walk" in html
        assert "Sunset Drive" in html

    def test_shows_scene_info(self, client: TestClient) -> None:
        resp = client.get("/review/scripts")
        html = resp.text
        assert "Opening" in html
        assert "Wide shot of park" in html
        assert "Scene 1" in html

    def test_empty_state(self, empty_client: TestClient) -> None:
        resp = empty_client.get("/review/scripts")
        assert resp.status_code == 200
        assert "No scripts" in resp.text

    def test_extends_base_template(self, client: TestClient) -> None:
        resp = client.get("/review/scripts")
        assert "Autopilot Video" in resp.text


# ---------------------------------------------------------------------------
# GET /review/edit-plans (redirect to /review/render)
# ---------------------------------------------------------------------------


class TestEditPlansRedirect:
    """Tests for /review/edit-plans redirecting to /review/render."""

    def test_redirects_to_render_index(self, client: TestClient) -> None:
        resp = client.get("/review/edit-plans", follow_redirects=False)
        assert resp.status_code == 307
        assert "/review/render" in resp.headers["location"]


# ---------------------------------------------------------------------------
# GET /review/renders (redirect to /review/render)
# ---------------------------------------------------------------------------


class TestRendersRedirect:
    """Tests for /review/renders redirecting to /review/render."""

    def test_redirects_to_render(self, client: TestClient) -> None:
        resp = client.get("/review/renders", follow_redirects=False)
        assert resp.status_code == 307
        assert "/review/render" in resp.headers["location"]
