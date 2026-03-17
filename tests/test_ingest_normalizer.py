"""Tests for autopilot.ingest.normalizer — EBU R128 audio normalization."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

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


class TestNormalizeSkip:
    """Tests for normalizer skip logic and error handling."""

    def test_normalize_skips_existing_output(self, tmp_path: Path) -> None:
        """normalize_audio should skip ffmpeg when output already exists and is non-empty."""
        media_path = tmp_path / "clip.mp4"
        media_path.write_bytes(b"\x00")
        output_dir = tmp_path / "normalized"
        output_dir.mkdir()
        existing = output_dir / "clip.wav"
        existing.write_bytes(b"\x00\x01\x02")  # non-empty

        with patch("subprocess.run") as mock_run:
            result = normalize_audio(media_path, output_dir)

        assert result == existing
        mock_run.assert_not_called()

    def test_normalize_does_not_skip_if_missing(self, tmp_path: Path) -> None:
        """normalize_audio should run ffmpeg when output file doesn't exist."""
        media_path = tmp_path / "clip.mp4"
        media_path.write_bytes(b"\x00")
        output_dir = tmp_path / "normalized"

        with patch("subprocess.run") as mock_run:
            normalize_audio(media_path, output_dir)

        mock_run.assert_called_once()

    def test_normalize_ffmpeg_failure(self, tmp_path: Path) -> None:
        """normalize_audio should propagate CalledProcessError from ffmpeg."""
        media_path = tmp_path / "clip.mp4"
        media_path.write_bytes(b"\x00")
        output_dir = tmp_path / "normalized"

        with (
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
            ),
            pytest.raises(subprocess.CalledProcessError),
        ):
            normalize_audio(media_path, output_dir)


class TestNormalizeCollisionFix:
    """Tests for output filename collision prevention via root_dir parameter."""

    def test_normalize_disambiguates_same_stem_different_dirs(
        self, tmp_path: Path
    ) -> None:
        """Two input files with the same stem in different subdirs
        should produce different outputs."""
        day1 = tmp_path / "day1"
        day2 = tmp_path / "day2"
        day1.mkdir()
        day2.mkdir()
        file1 = day1 / "DJI_0001.mp4"
        file2 = day2 / "DJI_0001.mp4"
        file1.write_bytes(b"\x00")
        file2.write_bytes(b"\x00")

        output_dir = tmp_path / "normalized"

        with patch("subprocess.run"):
            path1 = normalize_audio(file1, output_dir, root_dir=tmp_path)
            path2 = normalize_audio(file2, output_dir, root_dir=tmp_path)

        assert path1 != path2, (
            f"Same stem from different dirs should produce different outputs, "
            f"but both resolved to {path1}"
        )

    def test_normalize_with_root_dir_mirrors_structure(self, tmp_path: Path) -> None:
        """With root_dir, output should mirror input directory structure."""
        day1 = tmp_path / "day1"
        day1.mkdir()
        clip = day1 / "clip.mp4"
        clip.write_bytes(b"\x00")

        output_dir = tmp_path / "normalized"

        with patch("subprocess.run"):
            result = normalize_audio(clip, output_dir, root_dir=tmp_path)

        expected = output_dir / "day1" / "clip.wav"
        assert result == expected
        assert (output_dir / "day1").exists()

    def test_normalize_without_root_dir_uses_stem(self, tmp_path: Path) -> None:
        """Without root_dir, output should use plain stem (backward compatible)."""
        clip = tmp_path / "day1" / "clip.mp4"
        clip.parent.mkdir()
        clip.write_bytes(b"\x00")

        output_dir = tmp_path / "normalized"

        with patch("subprocess.run"):
            result = normalize_audio(clip, output_dir)

        assert result == output_dir / "clip.wav"
