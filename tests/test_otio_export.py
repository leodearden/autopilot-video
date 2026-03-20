"""Tests for OTIO export (autopilot.plan.otio_export)."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# -- Step 1: Public API surface tests -----------------------------------------


class TestOtioExportPublicAPI:
    """Verify OtioExportError, export_otio, detect_otio_changes surface."""

    def test_otio_export_error_importable(self):
        """OtioExportError is importable from otio_export module."""
        from autopilot.plan.otio_export import OtioExportError

        assert OtioExportError is not None

    def test_otio_export_error_is_exception(self):
        """OtioExportError is a subclass of Exception with message."""
        from autopilot.plan.otio_export import OtioExportError

        assert issubclass(OtioExportError, Exception)
        err = OtioExportError("test message")
        assert str(err) == "test message"

    def test_export_otio_signature(self):
        """export_otio has edl, output_path, db params and returns Path."""
        from autopilot.plan.otio_export import export_otio

        sig = inspect.signature(export_otio)
        params = list(sig.parameters.keys())
        assert "edl" in params
        assert "output_path" in params
        assert "db" in params

    def test_detect_otio_changes_signature(self):
        """detect_otio_changes has otio_path, original_edl params, returns dict."""
        from autopilot.plan.otio_export import detect_otio_changes

        sig = inspect.signature(detect_otio_changes)
        params = list(sig.parameters.keys())
        assert "otio_path" in params
        assert "original_edl" in params
        assert sig.return_annotation in (dict, "dict")

    def test_all_exports(self):
        """__all__ includes OtioExportError, export_otio, detect_otio_changes."""
        from autopilot.plan import otio_export

        assert "OtioExportError" in otio_export.__all__
        assert "export_otio" in otio_export.__all__
        assert "detect_otio_changes" in otio_export.__all__


# -- Step 3: Timecode-to-RationalTime conversion tests ------------------------

import opentimelineio as otio


class TestTcToRationalTime:
    """Verify _tc_to_rational_time converts HH:MM:SS.mmm to RationalTime."""

    def test_zero_timecode_30fps(self):
        """00:00:00.000 at 30fps -> RationalTime(0, 30)."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("00:00:00.000", 30.0)
        assert rt == otio.opentime.RationalTime(0, 30)

    def test_one_minute_thirty_seconds_30fps(self):
        """00:01:30.500 at 30fps -> 90.5 seconds."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("00:01:30.500", 30.0)
        expected_seconds = 90.5
        assert abs(otio.opentime.to_seconds(rt) - expected_seconds) < 0.001

    def test_one_hour_24fps(self):
        """01:00:00.000 at 24fps -> 3600 seconds."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("01:00:00.000", 24.0)
        assert rt.rate == 24.0
        assert abs(otio.opentime.to_seconds(rt) - 3600.0) < 0.001

    def test_millisecond_precision_60fps(self):
        """00:00:05.123 at 60fps -> 5.123 seconds."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("00:00:05.123", 60.0)
        assert rt.rate == 60.0
        assert abs(otio.opentime.to_seconds(rt) - 5.123) < 0.001

    def test_ten_seconds_30fps(self):
        """00:00:10.000 at 30fps -> 10.0 seconds."""
        from autopilot.plan.otio_export import _tc_to_rational_time

        rt = _tc_to_rational_time("00:00:10.000", 30.0)
        assert abs(otio.opentime.to_seconds(rt) - 10.0) < 0.001


# -- Step 5: Basic clip conversion tests --------------------------------------


def _mock_db_for_clips():
    """Create a mock CatalogDB that returns media for known clip_ids."""
    db = MagicMock()

    def get_media(media_id):
        media_map = {
            "v1": {
                "id": "v1",
                "file_path": "/media/clip_v1.mp4",
                "fps": 30.0,
                "duration_seconds": 120.0,
            },
            "v2": {
                "id": "v2",
                "file_path": "/media/clip_v2.mp4",
                "fps": 24.0,
                "duration_seconds": 60.0,
            },
        }
        return media_map.get(media_id)

    db.get_media = MagicMock(side_effect=get_media)
    return db


def _minimal_edl(clips=None, transitions=None):
    """Build a minimal EDL dict with defaults."""
    return {
        "clips": clips or [],
        "transitions": transitions or [],
        "crop_modes": [],
        "titles": [],
        "audio_settings": [],
        "music": [],
        "voiceovers": [],
        "broll_requests": [],
    }


class TestBasicClipConversion:
    """Verify export_otio creates valid .otio with clips."""

    def test_single_clip_produces_otio_file(self, tmp_path):
        """Single clip EDL exports to a .otio file that exists."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(clips=[{
            "clip_id": "v1",
            "in_timecode": "00:00:05.000",
            "out_timecode": "00:00:15.000",
            "track": 1,
        }])
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()

        result = export_otio(edl, output, db)
        assert result == output
        assert output.exists()

    def test_single_clip_round_trips(self, tmp_path):
        """Single clip .otio can be read back by OTIO."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(clips=[{
            "clip_id": "v1",
            "in_timecode": "00:00:05.000",
            "out_timecode": "00:00:15.000",
            "track": 1,
        }])
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        assert tl is not None
        assert isinstance(tl, otio.schema.Timeline)

    def test_single_clip_has_video_track(self, tmp_path):
        """Timeline has 1 video track with 1 clip."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(clips=[{
            "clip_id": "v1",
            "in_timecode": "00:00:05.000",
            "out_timecode": "00:00:15.000",
            "track": 1,
        }])
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [
            t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video
        ]
        assert len(video_tracks) == 1
        clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        assert len(clips) == 1

    def test_clip_has_correct_source_range(self, tmp_path):
        """Clip source_range covers in_timecode to out_timecode (10s duration)."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(clips=[{
            "clip_id": "v1",
            "in_timecode": "00:00:05.000",
            "out_timecode": "00:00:15.000",
            "track": 1,
        }])
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [
            t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video
        ]
        clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        clip = clips[0]

        # Source range start should be at 5.0 seconds
        start_sec = otio.opentime.to_seconds(clip.source_range.start_time)
        assert abs(start_sec - 5.0) < 0.01

        # Duration should be 10.0 seconds
        dur_sec = otio.opentime.to_seconds(clip.source_range.duration)
        assert abs(dur_sec - 10.0) < 0.01

    def test_clip_has_external_reference(self, tmp_path):
        """Clip media_reference is ExternalReference with correct target_url."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(clips=[{
            "clip_id": "v1",
            "in_timecode": "00:00:05.000",
            "out_timecode": "00:00:15.000",
            "track": 1,
        }])
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [
            t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video
        ]
        clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        clip = clips[0]

        assert isinstance(clip.media_reference, otio.schema.ExternalReference)
        assert clip.media_reference.target_url == "/media/clip_v1.mp4"


# -- Step 7: Multi-track support tests ----------------------------------------


class TestMultiTrackSupport:
    """Verify export_otio creates separate tracks for different track numbers."""

    def test_two_tracks_created(self, tmp_path):
        """EDL with clips on track 1 and 2 creates 2 video tracks."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(clips=[
            {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            },
            {
                "clip_id": "v2",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:05.000",
                "track": 2,
            },
        ])
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [
            t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video
        ]
        assert len(video_tracks) == 2

    def test_clips_on_correct_tracks(self, tmp_path):
        """Each track contains only its clips."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(clips=[
            {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            },
            {
                "clip_id": "v2",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:05.000",
                "track": 2,
            },
        ])
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [
            t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video
        ]

        # Track 1 (V1) has v1
        t1_clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        assert len(t1_clips) == 1
        assert t1_clips[0].name == "v1"

        # Track 2 (V2) has v2
        t2_clips = [c for c in video_tracks[1] if isinstance(c, otio.schema.Clip)]
        assert len(t2_clips) == 1
        assert t2_clips[0].name == "v2"

    def test_clips_ordered_by_timecode_within_track(self, tmp_path):
        """Clips on the same track are ordered by in_timecode."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(clips=[
            {
                "clip_id": "v2",
                "in_timecode": "00:00:20.000",
                "out_timecode": "00:00:30.000",
                "track": 1,
            },
            {
                "clip_id": "v1",
                "in_timecode": "00:00:05.000",
                "out_timecode": "00:00:15.000",
                "track": 1,
            },
        ])
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [
            t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video
        ]
        t1_clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        assert len(t1_clips) == 2
        # v1 comes first (in_timecode 00:00:05) then v2 (00:00:20)
        assert t1_clips[0].name == "v1"
        assert t1_clips[1].name == "v2"
