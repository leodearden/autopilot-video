"""Tests for script generation (autopilot.plan.script) and narrative_scripts DB CRUD."""

from __future__ import annotations

import inspect
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


# -- Step 1: Public API surface tests -----------------------------------------


class TestPublicAPI:
    """Verify ScriptError and public API surface."""

    def test_script_error_importable(self):
        """ScriptError is importable from script module."""
        from autopilot.plan.script import ScriptError

        assert ScriptError is not None

    def test_script_error_is_exception(self):
        """ScriptError is a subclass of Exception with message."""
        from autopilot.plan.script import ScriptError

        assert issubclass(ScriptError, Exception)
        err = ScriptError("test message")
        assert str(err) == "test message"

    def test_build_narrative_storyboard_signature(self):
        """build_narrative_storyboard has narrative_id and db params, returns str."""
        from autopilot.plan.script import build_narrative_storyboard

        sig = inspect.signature(build_narrative_storyboard)
        params = list(sig.parameters.keys())
        assert "narrative_id" in params
        assert "db" in params
        assert sig.return_annotation in (str, "str")

    def test_generate_script_signature(self):
        """generate_script has narrative_id, db, config params, returns dict."""
        from autopilot.plan.script import generate_script

        sig = inspect.signature(generate_script)
        params = list(sig.parameters.keys())
        assert "narrative_id" in params
        assert "db" in params
        assert "config" in params
        assert sig.return_annotation in (dict, "dict")

    def test_all_exports(self):
        """__all__ includes ScriptError, build_narrative_storyboard, generate_script."""
        from autopilot.plan import script

        assert "ScriptError" in script.__all__
        assert "build_narrative_storyboard" in script.__all__
        assert "generate_script" in script.__all__


# -- Step 3: build_narrative_storyboard basic tests ----------------------------


class TestBuildStoryboardBasic:
    """Tests for build_narrative_storyboard: not found and empty clusters."""

    def test_narrative_not_found_raises_script_error(self, catalog_db):
        """Raises ScriptError when narrative_id does not exist in DB."""
        from autopilot.plan.script import ScriptError, build_narrative_storyboard

        with pytest.raises(ScriptError, match="[Nn]arrative.*not found"):
            build_narrative_storyboard("nonexistent", catalog_db)

    def test_empty_cluster_list_returns_minimal_storyboard(self, catalog_db):
        """Narrative with no activity clusters returns minimal storyboard text."""
        from autopilot.plan.script import build_narrative_storyboard

        catalog_db.insert_narrative(
            "n1",
            title="Empty Narrative",
            activity_cluster_ids_json=json.dumps([]),
        )

        result = build_narrative_storyboard("n1", catalog_db)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should mention the narrative title or indicate it's empty
        assert "Empty Narrative" in result or "no" in result.lower()
