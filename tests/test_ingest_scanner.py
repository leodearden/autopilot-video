"""Tests for autopilot.ingest.scanner — media file discovery and metadata extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autopilot.ingest.scanner import MediaFile, scan_directory


class TestMediaFile:
    """Tests for the MediaFile dataclass."""

    def test_mediafile_fields(self) -> None:
        """All fields should be stored correctly when provided."""
        mf = MediaFile(
            file_path=Path("/footage/clip.mp4"),
            codec="h264",
            resolution_w=3840,
            resolution_h=2160,
            fps=30.0,
            duration_seconds=120.5,
            created_at="2026-01-15T10:30:00",
            gps_lat=13.7563,
            gps_lon=100.5018,
            audio_channels=2,
            sha256_prefix="abc123def456",
            metadata_json='{"format": "mp4"}',
        )
        assert mf.file_path == Path("/footage/clip.mp4")
        assert mf.codec == "h264"
        assert mf.resolution_w == 3840
        assert mf.resolution_h == 2160
        assert mf.fps == 30.0
        assert mf.duration_seconds == 120.5
        assert mf.created_at == "2026-01-15T10:30:00"
        assert mf.gps_lat == 13.7563
        assert mf.gps_lon == 100.5018
        assert mf.audio_channels == 2
        assert mf.sha256_prefix == "abc123def456"
        assert mf.metadata_json == '{"format": "mp4"}'

    def test_mediafile_defaults(self) -> None:
        """Optional fields should default to None when only file_path is given."""
        mf = MediaFile(file_path=Path("/footage/clip.mp4"))
        assert mf.file_path == Path("/footage/clip.mp4")
        assert mf.codec is None
        assert mf.resolution_w is None
        assert mf.resolution_h is None
        assert mf.fps is None
        assert mf.duration_seconds is None
        assert mf.created_at is None
        assert mf.gps_lat is None
        assert mf.gps_lon is None
        assert mf.audio_channels is None
        assert mf.sha256_prefix is None
        assert mf.metadata_json is None

    def test_mediafile_file_path_is_path(self) -> None:
        """file_path should be stored as a Path object."""
        mf = MediaFile(file_path=Path("/footage/clip.mp4"))
        assert isinstance(mf.file_path, Path)


class TestScanDirectory:
    """Tests for scan_directory file discovery."""

    def test_scan_finds_video_files(self, tmp_path: Path) -> None:
        """scan_directory should discover .mp4 and .mov video files."""
        (tmp_path / "clip1.mp4").write_bytes(b"\x00")
        (tmp_path / "clip2.mov").write_bytes(b"\x00")
        with patch("autopilot.ingest.scanner._probe_file") as mock_probe:
            mock_probe.side_effect = lambda p: MediaFile(file_path=p)
            results = scan_directory(tmp_path)
        paths = {mf.file_path.name for mf in results}
        assert "clip1.mp4" in paths
        assert "clip2.mov" in paths
        assert len(results) == 2

    def test_scan_finds_audio_files(self, tmp_path: Path) -> None:
        """scan_directory should discover .wav, .mp3, and .aac audio files."""
        (tmp_path / "audio.wav").write_bytes(b"\x00")
        (tmp_path / "audio.mp3").write_bytes(b"\x00")
        (tmp_path / "audio.aac").write_bytes(b"\x00")
        with patch("autopilot.ingest.scanner._probe_file") as mock_probe:
            mock_probe.side_effect = lambda p: MediaFile(file_path=p)
            results = scan_directory(tmp_path)
        paths = {mf.file_path.name for mf in results}
        assert "audio.wav" in paths
        assert "audio.mp3" in paths
        assert "audio.aac" in paths
        assert len(results) == 3

    def test_scan_finds_image_files(self, tmp_path: Path) -> None:
        """scan_directory should discover .jpg, .png, and .tiff image files."""
        (tmp_path / "photo.jpg").write_bytes(b"\x00")
        (tmp_path / "photo.png").write_bytes(b"\x00")
        (tmp_path / "photo.tiff").write_bytes(b"\x00")
        with patch("autopilot.ingest.scanner._probe_file") as mock_probe:
            mock_probe.side_effect = lambda p: MediaFile(file_path=p)
            results = scan_directory(tmp_path)
        paths = {mf.file_path.name for mf in results}
        assert "photo.jpg" in paths
        assert "photo.png" in paths
        assert "photo.tiff" in paths
        assert len(results) == 3

    def test_scan_ignores_unsupported_extensions(self, tmp_path: Path) -> None:
        """scan_directory should skip .txt, .pdf, and .py files."""
        (tmp_path / "readme.txt").write_bytes(b"\x00")
        (tmp_path / "doc.pdf").write_bytes(b"\x00")
        (tmp_path / "script.py").write_bytes(b"\x00")
        (tmp_path / "clip.mp4").write_bytes(b"\x00")
        with patch("autopilot.ingest.scanner._probe_file") as mock_probe:
            mock_probe.side_effect = lambda p: MediaFile(file_path=p)
            results = scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0].file_path.name == "clip.mp4"

    def test_scan_walks_subdirectories(self, tmp_path: Path) -> None:
        """scan_directory should recursively find media in nested directories."""
        subdir = tmp_path / "day1" / "morning"
        subdir.mkdir(parents=True)
        (tmp_path / "top.mp4").write_bytes(b"\x00")
        (subdir / "nested.mov").write_bytes(b"\x00")
        with patch("autopilot.ingest.scanner._probe_file") as mock_probe:
            mock_probe.side_effect = lambda p: MediaFile(file_path=p)
            results = scan_directory(tmp_path)
        names = {mf.file_path.name for mf in results}
        assert "top.mp4" in names
        assert "nested.mov" in names
        assert len(results) == 2

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """scan_directory should return an empty list for an empty directory."""
        with patch("autopilot.ingest.scanner._probe_file") as mock_probe:
            results = scan_directory(tmp_path)
        assert results == []
        mock_probe.assert_not_called()
