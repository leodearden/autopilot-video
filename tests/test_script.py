"""Tests for script generation (autopilot.plan.script) and narrative_scripts DB CRUD."""

from __future__ import annotations

import json

import pytest


# -- Pre-1: narrative_scripts DB CRUD tests ------------------------------------


class TestDBNarrativeScripts:
    """Tests for narrative_scripts table and CRUD methods in CatalogDB."""

    def test_upsert_and_get_narrative_script(self, catalog_db):
        """Upsert stores a script and get retrieves it."""
        # Need a narrative row first (FK)
        catalog_db.insert_narrative("n1", title="Test Narrative")

        script_data = json.dumps({"scenes": [], "broll_needs": [], "quality_flags": []})
        catalog_db.upsert_narrative_script("n1", script_data)

        result = catalog_db.get_narrative_script("n1")
        assert result is not None
        assert result["narrative_id"] == "n1"
        assert result["script_json"] == script_data
        assert "created_at" in result

    def test_get_nonexistent_returns_none(self, catalog_db):
        """get_narrative_script returns None for non-existent narrative."""
        result = catalog_db.get_narrative_script("nonexistent")
        assert result is None

    def test_upsert_overwrites(self, catalog_db):
        """Second upsert overwrites the first script_json."""
        catalog_db.insert_narrative("n1", title="Test Narrative")

        script_v1 = json.dumps({"scenes": [{"scene_number": 1}]})
        script_v2 = json.dumps({"scenes": [{"scene_number": 1}, {"scene_number": 2}]})

        catalog_db.upsert_narrative_script("n1", script_v1)
        catalog_db.upsert_narrative_script("n1", script_v2)

        result = catalog_db.get_narrative_script("n1")
        assert result is not None
        assert result["script_json"] == script_v2
