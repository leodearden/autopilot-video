"""Tests for autopilot/prompts/script_writer.md prompt template."""

import re
from pathlib import Path


class TestScriptWriterPromptExists:
    """Verify the script writer prompt file exists and has substantial content."""

    def test_file_exists(self, prompts_dir: Path):
        path = prompts_dir / "script_writer.md"
        assert path.exists(), f"Expected prompt file at {path}"

    def test_file_is_non_empty(self, prompts_dir: Path):
        path = prompts_dir / "script_writer.md"
        content = path.read_text()
        assert len(content) > 200, (
            f"script_writer.md should have substantial content (>200 chars), "
            f"got {len(content)} chars"
        )


class TestScriptWriterInputSection:
    """Verify input section references required elements."""

    def _read(self, prompts_dir: Path) -> str:
        return (prompts_dir / "script_writer.md").read_text().lower()

    def test_references_narrative(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "narrative" in content, (
            "Prompt must reference 'narrative' in input section"
        )

    def test_references_storyboard(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "storyboard" in content or "l-storyboard" in content, (
            "Prompt must reference 'storyboard' or 'L-Storyboard' in input section"
        )


class TestScriptWriterOutputFormat:
    """Verify output format mentions required elements."""

    def _read(self, prompts_dir: Path) -> str:
        return (prompts_dir / "script_writer.md").read_text().lower()

    def test_output_mentions_scenes(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "scene" in content or "scenes" in content, (
            "Output format must reference 'scene'/'scenes'"
        )

    def test_output_mentions_voiceover(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "voiceover" in content, (
            "Output format must reference 'voiceover'"
        )

    def test_output_mentions_titles(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "title" in content or "titles" in content, (
            "Output format must reference 'title'/'titles'"
        )

    def test_output_mentions_music(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "music" in content or "music_mood" in content, (
            "Output format must reference 'music'/'music_mood'"
        )

    def test_output_mentions_broll(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert (
            "broll" in content
            or "b-roll" in content
            or "b_roll" in content
        ), "Output format must reference B-roll (broll/b-roll/b_roll)"

    def test_output_mentions_quality_flag(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "quality_flag" in content or "quality flag" in content, (
            "Prompt must reference 'quality_flag'/'quality flag' for flagging "
            "source footage issues"
        )

    def test_output_mentions_json(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "json" in content, (
            "Output format must mention JSON for structured output"
        )
