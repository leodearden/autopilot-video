"""Tests for autopilot.ingest.normalizer — EBU R128 audio normalization."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from autopilot.ingest.normalizer import normalize_audio


class TestNormalizeAudio:
    """Tests for normalize_audio ffmpeg command construction."""

    def test_normalize_builds_correct_ffmpeg_command(self, tmp_path: Path) -> None:
        """normalize_audio should call ffmpeg with EBU R128 loudnorm filter."""
        media_path = tmp_path / "clip.mp4"
        media_path.write_bytes(b"\x00")
        output_dir = tmp_path / "normalized"

        with patch("subprocess.run") as mock_run:
            normalize_audio(media_path, output_dir)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(media_path) in cmd
        assert "-af" in cmd
        af_idx = cmd.index("-af")
        assert "loudnorm=I=-16:TP=-1.5:LRA=11" in cmd[af_idx + 1]
        assert "-ar" in cmd
        assert "48000" in cmd
        assert "-ac" in cmd
        assert "2" in cmd
        assert "-y" in cmd
        # Output should be <stem>.wav in output_dir
        output_path_str = cmd[-1]
        assert output_path_str == str(output_dir / "clip.wav")

    def test_normalize_returns_output_path(self, tmp_path: Path) -> None:
        """normalize_audio should return the output wav file path."""
        media_path = tmp_path / "clip.mp4"
        media_path.write_bytes(b"\x00")
        output_dir = tmp_path / "normalized"

        with patch("subprocess.run"):
            result = normalize_audio(media_path, output_dir)

        assert result == output_dir / "clip.wav"

    def test_normalize_creates_output_dir(self, tmp_path: Path) -> None:
        """normalize_audio should create output_dir if it doesn't exist."""
        media_path = tmp_path / "clip.mp4"
        media_path.write_bytes(b"\x00")
        output_dir = tmp_path / "normalized" / "deep"

        assert not output_dir.exists()
        with patch("subprocess.run"):
            normalize_audio(media_path, output_dir)
        assert output_dir.exists()

    def test_normalize_audio_file(self, tmp_path: Path) -> None:
        """normalize_audio should handle .wav audio input correctly."""
        media_path = tmp_path / "audio.wav"
        media_path.write_bytes(b"\x00")
        output_dir = tmp_path / "normalized"

        with patch("subprocess.run") as mock_run:
            result = normalize_audio(media_path, output_dir)

        assert result == output_dir / "audio.wav"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert str(media_path) in cmd
