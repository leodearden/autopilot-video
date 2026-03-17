"""Tests for autopilot/prompts/narrative_planner.md prompt template."""

import re
from pathlib import Path


class TestNarrativePlannerPromptExists:
    """Verify the narrative planner prompt file exists and has substantial content."""

    def test_file_exists(self, prompts_dir: Path):
        path = prompts_dir / "narrative_planner.md"
        assert path.exists(), f"Expected prompt file at {path}"

    def test_file_is_non_empty(self, prompts_dir: Path):
        path = prompts_dir / "narrative_planner.md"
        content = path.read_text()
        assert len(content) > 200, (
            f"narrative_planner.md should have substantial content (>200 chars), "
            f"got {len(content)} chars"
        )


class TestNarrativePlannerCreatorPlaceholders:
    """Verify all 6 creator profile placeholders are present."""

    PLACEHOLDERS = [
        "{creator_name}",
        "{channel_style}",
        "{target_audience}",
        "{default_video_duration}",
        "{narration_style}",
        "{music_preference}",
    ]

    def test_all_creator_placeholders_present(self, prompts_dir: Path):
        content = (prompts_dir / "narrative_planner.md").read_text()
        for placeholder in self.PLACEHOLDERS:
            assert placeholder in content, (
                f"Missing creator profile placeholder: {placeholder}"
            )


class TestNarrativePlannerInputSection:
    """Verify input section references required elements."""

    def test_references_storyboard(self, prompts_dir: Path):
        content = (prompts_dir / "narrative_planner.md").read_text().lower()
        assert "storyboard" in content or "master storyboard" in content, (
            "Prompt must reference 'storyboard' or 'master storyboard' in input section"
        )


class TestNarrativePlannerOutputFormat:
    """Verify output format has required JSON fields."""

    def _read(self, prompts_dir: Path) -> str:
        return (prompts_dir / "narrative_planner.md").read_text().lower()

    def test_output_mentions_title(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "title" in content, "Output format must reference 'title' field"

    def test_output_mentions_activity_cluster_ids(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "activity_cluster_ids" in content, (
            "Output format must reference 'activity_cluster_ids' field"
        )

    def test_output_mentions_duration(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "proposed_duration" in content or "duration" in content, (
            "Output format must reference 'proposed_duration'/'duration' field"
        )

    def test_output_mentions_arc(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "arc" in content, "Output format must reference 'arc' field"

    def test_output_mentions_emotional_journey(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "emotional_journey" in content, (
            "Output format must reference 'emotional_journey' field"
        )

    def test_output_mentions_reasoning(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "reasoning" in content, (
            "Output format must reference 'reasoning' field"
        )

    def test_output_mentions_json(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "json" in content, (
            "Output format must mention JSON for structured output"
        )
