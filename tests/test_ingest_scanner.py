"""Tests for autopilot.ingest.scanner — media file discovery and metadata extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autopilot.ingest.scanner import MediaFile


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
