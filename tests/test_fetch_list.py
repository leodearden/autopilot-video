"""Tests for fetch list generation (autopilot.source.fetch_list)."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from autopilot.source import BrollRequest, MusicRequest, VoiceoverRequest


class TestFetchListPublicAPI:
    """Verify generate_fetch_list surface."""

    def test_generate_fetch_list_importable(self):
        """generate_fetch_list is importable from fetch_list module."""
        from autopilot.source.fetch_list import generate_fetch_list

        assert generate_fetch_list is not None

    def test_generate_fetch_list_signature(self):
        """generate_fetch_list has unresolved, output_path params."""
        from autopilot.source.fetch_list import generate_fetch_list

        sig = inspect.signature(generate_fetch_list)
        params = list(sig.parameters.keys())
        assert "unresolved" in params
        assert "output_path" in params

    def test_all_exports(self):
        """__all__ includes generate_fetch_list."""
        from autopilot.source import fetch_list

        assert "generate_fetch_list" in fetch_list.__all__


class TestFetchListOutput:
    """Tests for fetch list content and format."""

    def test_writes_markdown_file(self, tmp_path):
        """generate_fetch_list writes a .md file to the specified path."""
        from autopilot.source.fetch_list import generate_fetch_list

        output_path = tmp_path / "fetch_list.md"
        unresolved = [
            MusicRequest(mood="upbeat", duration=30.0, start_time="00:01:00.000"),
        ]
        generate_fetch_list(unresolved, output_path)
        assert output_path.exists()

    def test_markdown_has_table_header(self, tmp_path):
        """Output markdown contains a table with type/description/suggested_sources/priority columns."""
        from autopilot.source.fetch_list import generate_fetch_list

        output_path = tmp_path / "fetch_list.md"
        unresolved = [
            MusicRequest(mood="ambient", duration=60.0, start_time="00:00:00.000"),
        ]
        generate_fetch_list(unresolved, output_path)
        content = output_path.read_text()
        # Should have table headers
        assert "Type" in content
        assert "Description" in content
        assert "Suggested Sources" in content
        assert "Priority" in content

    def test_music_request_in_table(self, tmp_path):
        """MusicRequest entries appear in the fetch list."""
        from autopilot.source.fetch_list import generate_fetch_list

        output_path = tmp_path / "fetch_list.md"
        unresolved = [
            MusicRequest(mood="gentle piano", duration=45.0, start_time="00:02:30.000"),
        ]
        generate_fetch_list(unresolved, output_path)
        content = output_path.read_text()
        assert "Music" in content
        assert "gentle piano" in content

    def test_broll_request_in_table(self, tmp_path):
        """BrollRequest entries appear in the fetch list."""
        from autopilot.source.fetch_list import generate_fetch_list

        output_path = tmp_path / "fetch_list.md"
        unresolved = [
            BrollRequest(
                description="aerial view of rice terraces",
                duration=5.0,
                start_time="00:03:00.000",
            ),
        ]
        generate_fetch_list(unresolved, output_path)
        content = output_path.read_text()
        assert "B-Roll" in content
        assert "aerial view of rice terraces" in content

    def test_voiceover_request_in_table(self, tmp_path):
        """VoiceoverRequest entries appear in the fetch list."""
        from autopilot.source.fetch_list import generate_fetch_list

        output_path = tmp_path / "fetch_list.md"
        unresolved = [
            VoiceoverRequest(
                text="Welcome to the journey",
                start_time="00:00:05.000",
                duration=3.0,
            ),
        ]
        generate_fetch_list(unresolved, output_path)
        content = output_path.read_text()
        assert "Voiceover" in content
        assert "Welcome to the journey" in content

    def test_empty_list_writes_header_only(self, tmp_path):
        """Empty unresolved list writes just the header."""
        from autopilot.source.fetch_list import generate_fetch_list

        output_path = tmp_path / "fetch_list.md"
        generate_fetch_list([], output_path)
        content = output_path.read_text()
        assert "Type" in content
        # No data rows after header
        lines = [l for l in content.strip().split("\n") if l.startswith("|")]
        # Header row + separator row = 2 lines
        assert len(lines) == 2

    def test_mixed_request_types(self, tmp_path):
        """Handles a mix of all three request types."""
        from autopilot.source.fetch_list import generate_fetch_list

        output_path = tmp_path / "fetch_list.md"
        unresolved = [
            MusicRequest(mood="chill", duration=20.0, start_time="00:00:00.000"),
            BrollRequest(description="sunset", duration=4.0, start_time="00:01:00.000"),
            VoiceoverRequest(text="Intro text", start_time="00:00:10.000", duration=5.0),
        ]
        generate_fetch_list(unresolved, output_path)
        content = output_path.read_text()
        assert "Music" in content
        assert "B-Roll" in content
        assert "Voiceover" in content

    def test_file_written_to_specified_path(self, tmp_path):
        """Output file is written to the exact path specified."""
        from autopilot.source.fetch_list import generate_fetch_list

        subdir = tmp_path / "output" / "assets"
        output_path = subdir / "my_fetch_list.md"
        unresolved = [
            MusicRequest(mood="rock", duration=60.0, start_time="00:05:00.000"),
        ]
        generate_fetch_list(unresolved, output_path)
        assert output_path.exists()
        assert output_path.read_text().strip() != ""

    def test_suggested_sources_per_type(self, tmp_path):
        """Each request type has appropriate suggested sources."""
        from autopilot.source.fetch_list import generate_fetch_list

        output_path = tmp_path / "fetch_list.md"
        unresolved = [
            MusicRequest(mood="jazz", duration=30.0, start_time="00:00:00.000"),
            BrollRequest(description="ocean", duration=5.0, start_time="00:01:00.000"),
            VoiceoverRequest(text="Hello", start_time="00:02:00.000", duration=2.0),
        ]
        generate_fetch_list(unresolved, output_path)
        content = output_path.read_text()
        # Music sources should mention Freesound or similar
        # B-roll sources should mention Pexels or Pixabay
        # Voiceover should mention TTS engines
        lines = content.split("\n")
        for line in lines:
            if "Music" in line and "jazz" in line:
                assert any(s in line for s in ["Freesound", "MusicGen", "AudioJungle"])
            if "B-Roll" in line and "ocean" in line:
                assert any(s in line for s in ["Pexels", "Pixabay"])
            if "Voiceover" in line and "Hello" in line:
                assert any(s in line for s in ["Kokoro", "ElevenLabs", "TTS"])
