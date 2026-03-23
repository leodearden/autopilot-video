"""Tests for autopilot/prompts/activity_label.md prompt template."""

from pathlib import Path


class TestActivityLabelPromptExists:
    """Verify the activity label prompt file exists and has substantial content."""

    def test_file_exists(self, prompts_dir: Path):
        path = prompts_dir / "activity_label.md"
        assert path.exists(), f"Expected prompt file at {path}"

    def test_file_is_non_empty(self, prompts_dir: Path):
        path = prompts_dir / "activity_label.md"
        content = path.read_text()
        assert len(content) > 100, (
            f"activity_label.md should have substantial content (>100 chars), "
            f"got {len(content)} chars"
        )


class TestActivityLabelInputSignals:
    """Verify the prompt references all required input signals."""

    def _read(self, prompts_dir: Path) -> str:
        return (prompts_dir / "activity_label.md").read_text().lower()

    def test_references_transcript(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "transcript" in content or "transcription" in content, (
            "Prompt must reference transcript/transcription input signal"
        )

    def test_references_yolo_detection(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "yolo" in content or "object detect" in content, (
            "Prompt must reference YOLO/object detection input signal"
        )

    def test_references_audio_events(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "audio event" in content or "audio_event" in content, (
            "Prompt must reference audio event input signal"
        )

    def test_references_gps_location(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "gps" in content or "location" in content, (
            "Prompt must reference GPS/location input signal"
        )

    def test_references_time_range(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "time range" in content or "time_range" in content or "temporal" in content, (
            "Prompt must reference time range/temporal input signal"
        )


class TestActivityLabelOutputFormat:
    """Verify output format section has required JSON fields."""

    def _read(self, prompts_dir: Path) -> str:
        return (prompts_dir / "activity_label.md").read_text().lower()

    def test_output_mentions_label_field(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "label" in content, "Output format must reference 'label' field"

    def test_output_mentions_description_field(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "description" in content, "Output format must reference 'description' field"

    def test_output_mentions_split_field(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "split_recommended" in content or "split" in content, (
            "Output format must reference 'split_recommended'/'split' field"
        )

    def test_output_mentions_json(self, prompts_dir: Path):
        content = self._read(prompts_dir)
        assert "json" in content, "Output format must mention JSON for structured output"
