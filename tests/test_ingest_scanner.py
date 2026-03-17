"""Tests for autopilot.ingest.scanner — media file discovery and metadata extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from autopilot.ingest.scanner import (
    MediaFile,
    _run_exiftool,
    _run_ffprobe,
    probe_file,
    scan_directory,
)


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


def _sync_executor_mock(probe_fn=None):
    """Build a mock ProcessPoolExecutor that runs synchronously.

    If *probe_fn* is given, executor.map will call it for each item.
    Otherwise it calls the function argument passed to map().
    """
    mock_executor = MagicMock()
    mock_executor.__enter__ = MagicMock(return_value=mock_executor)
    mock_executor.__exit__ = MagicMock(return_value=False)

    def _sync_map(fn, items):
        f = probe_fn if probe_fn is not None else fn
        return [f(item) for item in items]

    mock_executor.map.side_effect = _sync_map
    return mock_executor


class TestScanDirectory:
    """Tests for scan_directory file discovery."""

    def test_scan_finds_video_files(self, tmp_path: Path) -> None:
        """scan_directory should discover .mp4 and .mov video files."""
        (tmp_path / "clip1.mp4").write_bytes(b"\x00")
        (tmp_path / "clip2.mov").write_bytes(b"\x00")
        probe = lambda p: MediaFile(file_path=p)  # noqa: E731
        executor = _sync_executor_mock(probe)
        with patch("autopilot.ingest.scanner.ProcessPoolExecutor", return_value=executor):
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
        probe = lambda p: MediaFile(file_path=p)  # noqa: E731
        executor = _sync_executor_mock(probe)
        with patch("autopilot.ingest.scanner.ProcessPoolExecutor", return_value=executor):
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
        probe = lambda p: MediaFile(file_path=p)  # noqa: E731
        executor = _sync_executor_mock(probe)
        with patch("autopilot.ingest.scanner.ProcessPoolExecutor", return_value=executor):
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
        probe = lambda p: MediaFile(file_path=p)  # noqa: E731
        executor = _sync_executor_mock(probe)
        with patch("autopilot.ingest.scanner.ProcessPoolExecutor", return_value=executor):
            results = scan_directory(tmp_path)
        assert len(results) == 1
        assert results[0].file_path.name == "clip.mp4"

    def test_scan_walks_subdirectories(self, tmp_path: Path) -> None:
        """scan_directory should recursively find media in nested directories."""
        subdir = tmp_path / "day1" / "morning"
        subdir.mkdir(parents=True)
        (tmp_path / "top.mp4").write_bytes(b"\x00")
        (subdir / "nested.mov").write_bytes(b"\x00")
        probe = lambda p: MediaFile(file_path=p)  # noqa: E731
        executor = _sync_executor_mock(probe)
        with patch("autopilot.ingest.scanner.ProcessPoolExecutor", return_value=executor):
            results = scan_directory(tmp_path)
        names = {mf.file_path.name for mf in results}
        assert "top.mp4" in names
        assert "nested.mov" in names
        assert len(results) == 2

    def test_scan_empty_directory(self, tmp_path: Path) -> None:
        """scan_directory should return an empty list for an empty directory."""
        results = scan_directory(tmp_path)
        assert results == []


class TestRunFfprobe:
    """Tests for _run_ffprobe metadata extraction."""

    def _make_ffprobe_output(
        self,
        *,
        video_codec: str | None = "h264",
        width: int | None = 3840,
        height: int | None = 2160,
        r_frame_rate: str = "30/1",
        duration: str = "120.5",
        audio_channels: int | None = 2,
    ) -> str:
        """Build a realistic ffprobe JSON output string."""
        streams = []
        if video_codec is not None:
            streams.append(
                {
                    "codec_type": "video",
                    "codec_name": video_codec,
                    "width": width,
                    "height": height,
                    "r_frame_rate": r_frame_rate,
                }
            )
        if audio_channels is not None:
            streams.append(
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": audio_channels,
                }
            )
        return json.dumps(
            {
                "streams": streams,
                "format": {"duration": duration, "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
            }
        )

    def test_parse_ffprobe_video(self) -> None:
        """ffprobe JSON for a video file should yield codec, resolution, fps, duration, channels."""
        output = self._make_ffprobe_output()
        mock_result = MagicMock()
        mock_result.stdout = output
        with patch("subprocess.run", return_value=mock_result):
            result = _run_ffprobe(Path("/footage/clip.mp4"))
        assert result["codec"] == "h264"
        assert result["resolution_w"] == 3840
        assert result["resolution_h"] == 2160
        assert result["fps"] == 30.0
        assert result["duration_seconds"] == 120.5
        assert result["audio_channels"] == 2

    def test_parse_ffprobe_audio_only(self) -> None:
        """ffprobe JSON for an audio-only file should have no resolution/fps."""
        output = self._make_ffprobe_output(video_codec=None, width=None, height=None)
        mock_result = MagicMock()
        mock_result.stdout = output
        with patch("subprocess.run", return_value=mock_result):
            result = _run_ffprobe(Path("/footage/audio.wav"))
        assert result.get("resolution_w") is None
        assert result.get("resolution_h") is None
        assert result.get("fps") is None
        assert result["duration_seconds"] == 120.5
        assert result["audio_channels"] == 2
        # Audio-only: codec should come from audio stream
        assert result["codec"] == "aac"

    def test_parse_ffprobe_image(self) -> None:
        """ffprobe JSON for an image should have codec and resolution but no duration."""
        output = self._make_ffprobe_output(
            video_codec="mjpeg",
            width=4032,
            height=3024,
            duration="0",
            audio_channels=None,
        )
        mock_result = MagicMock()
        mock_result.stdout = output
        with patch("subprocess.run", return_value=mock_result):
            result = _run_ffprobe(Path("/footage/photo.jpg"))
        assert result["codec"] == "mjpeg"
        assert result["resolution_w"] == 4032
        assert result["resolution_h"] == 3024
        assert result.get("audio_channels") is None

    def test_ffprobe_failure(self) -> None:
        """_run_ffprobe should return empty dict on subprocess failure."""
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "ffprobe"),
        ):
            result = _run_ffprobe(Path("/footage/corrupt.mp4"))
        assert result == {}


class TestRunExiftool:
    """Tests for _run_exiftool metadata extraction."""

    def test_parse_exiftool_dji_mp4(self) -> None:
        """Exiftool JSON for a DJI file should yield created_at and GPS coords."""
        output = json.dumps(
            [
                {
                    "SourceFile": "/footage/DJI_0001.mp4",
                    "CreateDate": "2026:01:15 10:30:00",
                    "GPSLatitude": 13.7563,
                    "GPSLongitude": 100.5018,
                }
            ]
        )
        mock_result = MagicMock()
        mock_result.stdout = output
        with patch("subprocess.run", return_value=mock_result):
            result = _run_exiftool(Path("/footage/DJI_0001.mp4"))
        assert result["created_at"] == "2026-01-15T10:30:00"
        assert result["gps_lat"] == 13.7563
        assert result["gps_lon"] == 100.5018

    def test_parse_exiftool_no_gps(self) -> None:
        """Exiftool JSON without GPS fields should yield None for gps_lat/gps_lon."""
        output = json.dumps(
            [
                {
                    "SourceFile": "/footage/clip.mp4",
                    "CreateDate": "2026:01:15 10:30:00",
                }
            ]
        )
        mock_result = MagicMock()
        mock_result.stdout = output
        with patch("subprocess.run", return_value=mock_result):
            result = _run_exiftool(Path("/footage/clip.mp4"))
        assert result["created_at"] == "2026-01-15T10:30:00"
        assert result.get("gps_lat") is None
        assert result.get("gps_lon") is None

    def test_parse_exiftool_no_date(self) -> None:
        """Exiftool JSON without date fields should yield None for created_at."""
        output = json.dumps(
            [
                {
                    "SourceFile": "/footage/clip.mp4",
                    "GPSLatitude": 13.7563,
                    "GPSLongitude": 100.5018,
                }
            ]
        )
        mock_result = MagicMock()
        mock_result.stdout = output
        with patch("subprocess.run", return_value=mock_result):
            result = _run_exiftool(Path("/footage/clip.mp4"))
        assert result.get("created_at") is None
        assert result["gps_lat"] == 13.7563

    def test_exiftool_failure(self) -> None:
        """_run_exiftool should return empty dict on subprocess failure."""
        import subprocess

        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "exiftool"),
        ):
            result = _run_exiftool(Path("/footage/corrupt.mp4"))
        assert result == {}


class TestProbeFile:
    """Tests for probe_file combining ffprobe + exiftool."""

    def test_probe_file_combines_metadata(self) -> None:
        """probe_file should merge ffprobe and exiftool metadata into a MediaFile."""
        ffprobe_data = {
            "codec": "h264",
            "resolution_w": 3840,
            "resolution_h": 2160,
            "fps": 30.0,
            "duration_seconds": 120.5,
            "audio_channels": 2,
            "_raw": {"streams": [], "format": {}},
        }
        exiftool_data = {
            "created_at": "2026-01-15T10:30:00",
            "gps_lat": 13.7563,
            "gps_lon": 100.5018,
        }
        with (
            patch("autopilot.ingest.scanner._run_ffprobe", return_value=ffprobe_data),
            patch("autopilot.ingest.scanner._run_exiftool", return_value=exiftool_data),
        ):
            mf = probe_file(Path("/footage/clip.mp4"))
        assert mf.file_path == Path("/footage/clip.mp4")
        assert mf.codec == "h264"
        assert mf.resolution_w == 3840
        assert mf.resolution_h == 2160
        assert mf.fps == 30.0
        assert mf.duration_seconds == 120.5
        assert mf.audio_channels == 2
        assert mf.created_at == "2026-01-15T10:30:00"
        assert mf.gps_lat == 13.7563
        assert mf.gps_lon == 100.5018

    def test_probe_file_ffprobe_only(self) -> None:
        """probe_file should handle empty exiftool result gracefully."""
        ffprobe_data = {
            "codec": "h264",
            "resolution_w": 1920,
            "resolution_h": 1080,
            "fps": 24.0,
            "duration_seconds": 60.0,
            "audio_channels": 2,
            "_raw": {"streams": [], "format": {}},
        }
        with (
            patch("autopilot.ingest.scanner._run_ffprobe", return_value=ffprobe_data),
            patch("autopilot.ingest.scanner._run_exiftool", return_value={}),
        ):
            mf = probe_file(Path("/footage/clip.mp4"))
        assert mf.codec == "h264"
        assert mf.created_at is None
        assert mf.gps_lat is None
        assert mf.gps_lon is None

    def test_probe_file_stores_raw_metadata(self) -> None:
        """probe_file should store raw ffprobe JSON in metadata_json."""
        raw = {"streams": [{"codec_type": "video"}], "format": {"duration": "10"}}
        ffprobe_data = {
            "codec": "h264",
            "_raw": raw,
        }
        with (
            patch("autopilot.ingest.scanner._run_ffprobe", return_value=ffprobe_data),
            patch("autopilot.ingest.scanner._run_exiftool", return_value={}),
        ):
            mf = probe_file(Path("/footage/clip.mp4"))
        assert mf.metadata_json is not None
        parsed = json.loads(mf.metadata_json)
        assert parsed == raw


class TestScanDirectoryParallel:
    """Tests for scan_directory parallel execution."""

    def test_scan_uses_process_pool(self, tmp_path: Path) -> None:
        """scan_directory should use ProcessPoolExecutor with specified max_workers."""
        for i in range(5):
            (tmp_path / f"clip{i}.mp4").write_bytes(b"\x00")

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.return_value = [
            MediaFile(file_path=tmp_path / f"clip{i}.mp4") for i in range(5)
        ]

        with patch(
            "autopilot.ingest.scanner.ProcessPoolExecutor",
            return_value=mock_executor,
        ) as mock_pool_cls:
            results = scan_directory(tmp_path, max_workers=2)

        mock_pool_cls.assert_called_once_with(max_workers=2)
        assert mock_executor.map.called
        assert len(results) == 5

    def test_scan_default_workers(self, tmp_path: Path) -> None:
        """scan_directory without max_workers should use None (CPU count default)."""
        (tmp_path / "clip.mp4").write_bytes(b"\x00")

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.return_value = [
            MediaFile(file_path=tmp_path / "clip.mp4"),
        ]

        with patch(
            "autopilot.ingest.scanner.ProcessPoolExecutor",
            return_value=mock_executor,
        ) as mock_pool_cls:
            results = scan_directory(tmp_path)

        mock_pool_cls.assert_called_once_with(max_workers=None)
        assert len(results) == 1

    def test_scan_handles_probe_failure(self, tmp_path: Path) -> None:
        """scan_directory should not crash when executor.map raises mid-iteration."""
        (tmp_path / "aaa_good.mp4").write_bytes(b"\x00")
        (tmp_path / "zzz_bad.mp4").write_bytes(b"\x00")

        def mock_map(fn, files):
            """Yield one success then raise on the second (sorted: aaa first, zzz second)."""
            for f in files:
                if f.name == "zzz_bad.mp4":
                    raise Exception("probe failed")
                yield MediaFile(file_path=f)

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.side_effect = mock_map

        with patch(
            "autopilot.ingest.scanner.ProcessPoolExecutor",
            return_value=mock_executor,
        ):
            results = scan_directory(tmp_path)

        # aaa_good.mp4 yielded before the exception, so partial results returned
        assert len(results) == 1
        assert results[0].file_path.name == "aaa_good.mp4"

    def test_scan_returns_all_non_failing_files(self, tmp_path: Path) -> None:
        """Probe failure for one file should NOT drop subsequent non-failing files.

        With executor.map, the entire iteration aborts when one future raises,
        silently dropping all files after the failing one.  The desired behavior
        is per-future fault isolation: only the failing file is skipped.
        """
        (tmp_path / "aaa_good.mp4").write_bytes(b"\x00")
        (tmp_path / "mmm_bad.mp4").write_bytes(b"\x00")
        (tmp_path / "zzz_good.mp4").write_bytes(b"\x00")

        def _make_result(file_path):
            if file_path.name == "mmm_bad.mp4":
                raise Exception("probe failed for mmm_bad")
            return MediaFile(file_path=file_path)

        def mock_map(fn, files):
            """Simulates executor.map — raises mid-iteration, aborting the rest."""
            for f in files:
                yield _make_result(f)

        mock_executor = MagicMock()
        mock_executor.__enter__ = MagicMock(return_value=mock_executor)
        mock_executor.__exit__ = MagicMock(return_value=False)
        mock_executor.map.side_effect = mock_map

        with patch(
            "autopilot.ingest.scanner.ProcessPoolExecutor",
            return_value=mock_executor,
        ):
            results = scan_directory(tmp_path)

        # Both aaa_good and zzz_good should be returned; only mmm_bad is skipped
        names = {mf.file_path.name for mf in results}
        assert len(results) == 2, (
            f"Expected 2 results (aaa_good + zzz_good), got {len(results)}: {names}"
        )
        assert "aaa_good.mp4" in names
        assert "zzz_good.mp4" in names
        assert "mmm_bad.mp4" not in names


class TestIngestPackage:
    """Tests verifying the ingest package exports."""

    def test_scanner_exports(self) -> None:
        """Scanner public API should be importable from autopilot.ingest.scanner."""
        from autopilot.ingest.scanner import MediaFile as MF
        from autopilot.ingest.scanner import probe_file as pf
        from autopilot.ingest.scanner import scan_directory as sd

        assert callable(pf)
        assert callable(sd)
        assert MF is not None

    def test_normalizer_exports(self) -> None:
        """Normalizer public API should be importable from autopilot.ingest.normalizer."""
        from autopilot.ingest.normalizer import normalize_audio as na

        assert callable(na)

    def test_dedup_exports(self) -> None:
        """Dedup public API should be importable from autopilot.ingest.dedup."""
        from autopilot.ingest.dedup import compute_hash as ch
        from autopilot.ingest.dedup import find_duplicates as fd
        from autopilot.ingest.dedup import mark_duplicates as md

        assert callable(ch)
        assert callable(fd)
        assert callable(md)
