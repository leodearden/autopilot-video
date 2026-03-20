"""Tests for EDL asset resolution orchestrator (autopilot.source.resolve)."""

from __future__ import annotations

import inspect
import json
from unittest.mock import MagicMock, patch

from autopilot.source import BrollRequest, MusicRequest, VoiceoverRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edl(
    music: list[dict] | None = None,
    voiceovers: list[dict] | None = None,
    broll_requests: list[dict] | None = None,
) -> dict:
    """Create a minimal EDL dict for testing."""
    return {
        "clips": [],
        "transitions": [],
        "crop_modes": [],
        "titles": [],
        "audio_settings": [],
        "music": music or [],
        "voiceovers": voiceovers or [],
        "broll_requests": broll_requests or [],
    }


def _make_config(
    music_engine: str = "musicgen",
    tts_engine: str = "kokoro",
) -> MagicMock:
    """Create a mock ModelConfig."""
    config = MagicMock()
    config.music_engine = music_engine
    config.tts_engine = tts_engine
    return config


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------

class TestResolvePublicAPI:
    """Verify resolve_edl_assets surface."""

    def test_resolve_edl_assets_importable(self):
        """resolve_edl_assets is importable."""
        from autopilot.source.resolve import resolve_edl_assets

        assert resolve_edl_assets is not None

    def test_resolve_edl_assets_signature(self):
        """resolve_edl_assets has edl, config, output_dir, db params."""
        from autopilot.source.resolve import resolve_edl_assets

        sig = inspect.signature(resolve_edl_assets)
        params = list(sig.parameters.keys())
        assert "edl" in params
        assert "config" in params
        assert "output_dir" in params
        assert "db" in params

    def test_all_exports(self):
        """__all__ includes resolve_edl_assets."""
        from autopilot.source import resolve

        assert "resolve_edl_assets" in resolve.__all__


# ---------------------------------------------------------------------------
# Request extraction tests
# ---------------------------------------------------------------------------

class TestRequestExtraction:
    """Tests that resolve extracts requests from EDL correctly."""

    def test_extracts_music_requests(self, tmp_path):
        """Music entries in EDL become MusicRequest objects."""
        from autopilot.source.resolve import resolve_edl_assets

        edl = _make_edl(music=[
            {"mood": "upbeat", "duration": 30.0, "start_time": "00:01:00.000"},
        ])
        config = _make_config(music_engine="fetch_list_only")
        db = MagicMock()

        result = resolve_edl_assets(edl, config, tmp_path, db)

        # With fetch_list_only, all music stays unresolved
        assert "unresolved" in result
        assert any(
            isinstance(r, MusicRequest) and r.mood == "upbeat"
            for r in result["unresolved"]
        )

    def test_extracts_broll_requests(self, tmp_path):
        """Broll entries in EDL become BrollRequest objects."""
        from autopilot.source.resolve import resolve_edl_assets

        edl = _make_edl(broll_requests=[
            {"description": "mountain vista", "duration": 5.0, "start_time": "00:02:00.000"},
        ])
        config = _make_config()
        db = MagicMock()

        # No API keys set → broll stays unresolved
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("PEXELS_API_KEY", None)
            os.environ.pop("PIXABAY_API_KEY", None)
            result = resolve_edl_assets(edl, config, tmp_path, db)

        assert any(
            isinstance(r, BrollRequest) and r.description == "mountain vista"
            for r in result["unresolved"]
        )

    def test_extracts_voiceover_requests(self, tmp_path):
        """Voiceover entries in EDL become VoiceoverRequest objects."""
        from autopilot.source.resolve import resolve_edl_assets

        edl = _make_edl(voiceovers=[
            {"text": "Hello world", "start_time": "00:00:05.000", "duration": 3.0},
        ])
        config = _make_config(tts_engine="kokoro")
        db = MagicMock()

        # Kokoro not installed → raises VoiceoverError → stays unresolved
        result = resolve_edl_assets(edl, config, tmp_path, db)

        assert any(
            isinstance(r, VoiceoverRequest) and r.text == "Hello world"
            for r in result["unresolved"]
        )


# ---------------------------------------------------------------------------
# Resolution pipeline tests
# ---------------------------------------------------------------------------

class TestResolutionPipeline:
    """Tests for the full resolution pipeline."""

    def test_resolved_assets_get_paths_in_edl(self, tmp_path):
        """Resolved assets have resolved_path set in the EDL entries."""
        edl = _make_edl(
            music=[
                {"mood": "chill", "duration": 20.0, "start_time": "00:00:00.000"},
            ],
        )
        config = _make_config(music_engine="musicgen")
        db = MagicMock()

        fake_music_path = tmp_path / "music.wav"
        fake_music_path.write_bytes(b"audio")

        with patch("autopilot.source.resolve.source_music", return_value=fake_music_path):
            from autopilot.source.resolve import resolve_edl_assets

            result = resolve_edl_assets(edl, config, tmp_path, db)

        # The EDL music entry should have resolved_path
        assert result["edl"]["music"][0].get("resolved_path") == str(fake_music_path)

    def test_unresolved_collected(self, tmp_path):
        """Unresolved assets are collected in the result."""
        edl = _make_edl(
            music=[
                {"mood": "epic", "duration": 60.0, "start_time": "00:05:00.000"},
            ],
        )
        config = _make_config(music_engine="musicgen")
        db = MagicMock()

        with patch("autopilot.source.resolve.source_music", return_value=None):
            from autopilot.source.resolve import resolve_edl_assets

            result = resolve_edl_assets(edl, config, tmp_path, db)

        assert len(result["unresolved"]) == 1
        assert isinstance(result["unresolved"][0], MusicRequest)

    def test_fetch_list_generated_for_unresolved(self, tmp_path):
        """A fetch_list.md is generated when there are unresolved assets."""
        edl = _make_edl(
            music=[
                {"mood": "epic", "duration": 60.0, "start_time": "00:05:00.000"},
            ],
        )
        config = _make_config(music_engine="fetch_list_only")
        db = MagicMock()

        from autopilot.source.resolve import resolve_edl_assets

        resolve_edl_assets(edl, config, tmp_path, db)

        fetch_list_path = tmp_path / "fetch_list.md"
        assert fetch_list_path.exists()

    def test_empty_edl_produces_no_unresolved(self, tmp_path):
        """Empty EDL arrays produce no unresolved items."""
        edl = _make_edl()
        config = _make_config()
        db = MagicMock()

        from autopilot.source.resolve import resolve_edl_assets

        result = resolve_edl_assets(edl, config, tmp_path, db)

        assert result["unresolved"] == []


# ---------------------------------------------------------------------------
# Integration with DB tests
# ---------------------------------------------------------------------------

class TestResolveDBIntegration:
    """Tests for DB interaction in resolve pipeline."""

    def test_updates_edl_in_db(self, tmp_path, catalog_db):
        """Resolved EDL is stored back to DB via upsert_edit_plan."""
        # Seed a narrative and edit plan
        catalog_db.conn.execute(
            "INSERT INTO narratives (narrative_id, title, status) VALUES (?, ?, ?)",
            ("narr-1", "Test Narrative", "planned"),
        )
        catalog_db.upsert_edit_plan("narr-1", json.dumps(_make_edl()))

        edl = _make_edl(
            music=[
                {"mood": "upbeat", "duration": 30.0, "start_time": "00:01:00.000"},
            ],
        )
        config = _make_config(music_engine="fetch_list_only")

        from autopilot.source.resolve import resolve_edl_assets

        resolve_edl_assets(
            edl, config, tmp_path, catalog_db, narrative_id="narr-1"
        )

        # Verify the edit plan was updated
        plan = catalog_db.get_edit_plan("narr-1")
        assert plan is not None
        stored_edl = json.loads(plan["edl_json"])
        assert "music" in stored_edl

    def test_full_pipeline_with_mocked_sources(self, tmp_path):
        """Full pipeline: extract → source → update EDL → write fetch_list."""
        edl = _make_edl(
            music=[
                {"mood": "upbeat", "duration": 30.0, "start_time": "00:01:00.000"},
                {"mood": "ambient", "duration": 60.0, "start_time": "00:03:00.000"},
            ],
            voiceovers=[
                {"text": "Welcome", "start_time": "00:00:00.000", "duration": 2.0},
            ],
            broll_requests=[
                {"description": "sunset", "duration": 5.0, "start_time": "00:02:00.000"},
            ],
        )

        config = _make_config()
        db = MagicMock()

        music_path = tmp_path / "resolved_music.wav"
        music_path.write_bytes(b"audio")
        vo_path = tmp_path / "resolved_vo.wav"
        vo_path.write_bytes(b"audio")
        broll_paths = [tmp_path / "resolved_broll.mp4"]
        broll_paths[0].write_bytes(b"video")

        with patch("autopilot.source.resolve.source_music") as mock_sm, \
             patch("autopilot.source.resolve.generate_voiceover") as mock_gv, \
             patch("autopilot.source.resolve.source_broll") as mock_sb:

            # First music resolves, second doesn't
            mock_sm.side_effect = [music_path, None]
            mock_gv.return_value = vo_path
            mock_sb.return_value = broll_paths

            from autopilot.source.resolve import resolve_edl_assets

            result = resolve_edl_assets(edl, config, tmp_path, db)

        # 1 music resolved, 1 unresolved
        assert result["edl"]["music"][0].get("resolved_path") == str(music_path)
        assert result["edl"]["music"][1].get("resolved_path") is None

        # Voiceover resolved
        assert result["edl"]["voiceovers"][0].get("resolved_path") == str(vo_path)

        # B-roll resolved (first of the list)
        assert result["edl"]["broll_requests"][0].get("resolved_path") is not None

        # 1 unresolved music request
        assert len(result["unresolved"]) == 1
        assert isinstance(result["unresolved"][0], MusicRequest)

        # Fetch list generated
        assert (tmp_path / "fetch_list.md").exists()
