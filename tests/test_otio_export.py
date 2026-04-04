"""Tests for OTIO export (autopilot.plan.otio_export)."""

from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import opentimelineio as otio
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

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:05.000",
                    "out_timecode": "00:00:15.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()

        result = export_otio(edl, output, db)
        assert result == output
        assert output.exists()

    def test_single_clip_round_trips(self, tmp_path):
        """Single clip .otio can be read back by OTIO."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:05.000",
                    "out_timecode": "00:00:15.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        assert tl is not None
        assert isinstance(tl, otio.schema.Timeline)

    def test_single_clip_has_video_track(self, tmp_path):
        """Timeline has 1 video track with 1 clip."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:05.000",
                    "out_timecode": "00:00:15.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        assert len(video_tracks) == 1
        clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        assert len(clips) == 1

    def test_clip_has_correct_source_range(self, tmp_path):
        """Clip source_range covers in_timecode to out_timecode (10s duration)."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:05.000",
                    "out_timecode": "00:00:15.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
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

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:05.000",
                    "out_timecode": "00:00:15.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
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

        edl = _minimal_edl(
            clips=[
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
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        assert len(video_tracks) == 2

    def test_clips_on_correct_tracks(self, tmp_path):
        """Each track contains only its clips."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
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
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]

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

        edl = _minimal_edl(
            clips=[
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
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        t1_clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        assert len(t1_clips) == 2
        # v1 comes first (in_timecode 00:00:05) then v2 (00:00:20)
        assert t1_clips[0].name == "v1"
        assert t1_clips[1].name == "v2"


# -- Step 9: Transition mapping tests -----------------------------------------


class TestTransitionMapKeys:
    """Verify _TRANSITION_TYPE_MAP keys match the prompt schema."""

    def test_map_keys_match_prompt_schema(self):
        """_TRANSITION_TYPE_MAP keys exactly match prompt schema non-cut types.

        The edit_planner.md prompt schema defines: crossfade, cut, fade_in,
        fade_out, dissolve.  'cut' is implicit (no Transition object), so the
        map must contain exactly {crossfade, dissolve, fade_in, fade_out}.
        """
        from autopilot.plan.otio_export import _TRANSITION_TYPE_MAP

        expected_keys = {"crossfade", "dissolve", "fade_in", "fade_out"}
        assert set(_TRANSITION_TYPE_MAP.keys()) == expected_keys


class TestTransitionMapping:
    """Verify EDL transitions map to OTIO Transition objects."""

    @staticmethod
    def _assert_clip_transition_clip(track):
        """Assert track has Clip/Transition/Clip structure and return items list."""
        items = list(track)
        assert len(items) == 3
        assert isinstance(items[0], otio.schema.Clip)
        assert isinstance(items[1], otio.schema.Transition)
        assert isinstance(items[2], otio.schema.Clip)
        return items

    def test_crossfade_creates_smpte_dissolve(self, tmp_path):
        """EDL transition type 'crossfade' creates SMPTE_Dissolve between clips."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            transitions=[
                {
                    "type": "crossfade",
                    "duration": 1.0,
                    "position": 0,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]

        # Verify Clip/Transition/Clip structure and derive transitions
        track_items = self._assert_clip_transition_clip(video_tracks[0])
        transitions = [item for item in track_items if isinstance(item, otio.schema.Transition)]
        assert len(transitions) == 1
        assert transitions[0].transition_type == otio.schema.Transition.Type.SMPTE_Dissolve

    def test_crossfade_duration_correct(self, tmp_path):
        """Transition duration matches EDL duration (1.0 second)."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            transitions=[
                {
                    "type": "crossfade",
                    "duration": 1.0,
                    "position": 0,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]

        # Verify transition is positioned between clips
        track_items = list(video_tracks[0])
        assert isinstance(track_items[1], otio.schema.Transition)

        # Transition has in_offset + out_offset = total duration
        trans = track_items[1]
        total_dur_sec = otio.opentime.to_seconds(trans.in_offset) + otio.opentime.to_seconds(
            trans.out_offset
        )
        assert abs(total_dur_sec - 1.0) < 0.01

    def test_cut_type_produces_no_transition(self, tmp_path):
        """EDL transition type 'cut' produces no explicit Transition object."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            transitions=[
                {
                    "type": "cut",
                    "duration": 0,
                    "position": 1,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        transitions = [item for item in video_tracks[0] if isinstance(item, otio.schema.Transition)]
        assert len(transitions) == 0

    def test_fade_in_creates_smpte_dissolve(self, tmp_path):
        """EDL transition type 'fade_in' creates SMPTE_Dissolve with name='fade_in'."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            transitions=[
                {
                    "type": "fade_in",
                    "duration": 1.0,
                    "position": 0,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]

        # Verify Clip/Transition/Clip structure and derive transitions
        track_items = self._assert_clip_transition_clip(video_tracks[0])
        transitions = [item for item in track_items if isinstance(item, otio.schema.Transition)]
        assert len(transitions) == 1
        assert transitions[0].transition_type == otio.schema.Transition.Type.SMPTE_Dissolve
        assert transitions[0].name == "fade_in"

    def test_fade_out_creates_smpte_dissolve(self, tmp_path):
        """EDL transition type 'fade_out' creates SMPTE_Dissolve with name='fade_out'."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            transitions=[
                {
                    "type": "fade_out",
                    "duration": 1.0,
                    "position": 0,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]

        # Verify Clip/Transition/Clip structure and derive transitions
        track_items = self._assert_clip_transition_clip(video_tracks[0])
        transitions = [item for item in track_items if isinstance(item, otio.schema.Transition)]
        assert len(transitions) == 1
        assert transitions[0].transition_type == otio.schema.Transition.Type.SMPTE_Dissolve
        assert transitions[0].name == "fade_out"

    def test_dissolve_creates_smpte_dissolve(self, tmp_path):
        """EDL transition type 'dissolve' creates SMPTE_Dissolve with name='dissolve'."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            transitions=[
                {
                    "type": "dissolve",
                    "duration": 1.0,
                    "position": 0,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]

        # Verify Clip/Transition/Clip structure and derive transitions
        track_items = self._assert_clip_transition_clip(video_tracks[0])
        transitions = [item for item in track_items if isinstance(item, otio.schema.Transition)]
        assert len(transitions) == 1
        assert transitions[0].transition_type == otio.schema.Transition.Type.SMPTE_Dissolve
        assert transitions[0].name == "dissolve"

    def test_unknown_transition_type_falls_back_and_warns(self, tmp_path, caplog):
        """Unrecognized transition type (e.g. 'wipe') falls back to SMPTE_Dissolve and warns."""
        import logging

        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            transitions=[
                {
                    "type": "wipe",
                    "duration": 1.0,
                    "position": 0,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()

        with caplog.at_level(logging.WARNING, logger="autopilot.plan.otio_export"):
            export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]

        # Use structural helper to verify Clip/Transition/Clip and get items
        track_items = self._assert_clip_transition_clip(video_tracks[0])

        # Derive transitions from the single iteration
        transitions = [item for item in track_items if isinstance(item, otio.schema.Transition)]
        assert len(transitions) == 1
        assert transitions[0].transition_type == otio.schema.Transition.Type.SMPTE_Dissolve
        assert transitions[0].name == "wipe"

        # Verify WARNING log was emitted for unrecognized type
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("wipe" in msg for msg in warning_messages), (
            f"Expected a WARNING containing 'wipe', got: {warning_messages}"
        )

    def test_unknown_transition_type_falls_back_to_smpte_dissolve(self, tmp_path):
        """Unrecognized transition type (e.g. 'wipe') falls back to SMPTE_Dissolve."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            transitions=[
                {
                    "type": "wipe",
                    "duration": 1.0,
                    "position": 0,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        transitions = [item for item in video_tracks[0] if isinstance(item, otio.schema.Transition)]
        assert len(transitions) == 1
        assert transitions[0].transition_type == otio.schema.Transition.Type.SMPTE_Dissolve
        assert transitions[0].name == "wipe"

        # Verify structural ordering: Clip / Transition / Clip
        track_items = list(video_tracks[0])
        assert isinstance(track_items[0], otio.schema.Clip)
        assert isinstance(track_items[1], otio.schema.Transition)
        assert isinstance(track_items[2], otio.schema.Clip)

    def test_unknown_transition_type_emits_warning(self, tmp_path, caplog):
        """Unrecognized transition type emits a WARNING log message."""
        import logging

        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
            ],
            transitions=[
                {
                    "type": "wipe",
                    "duration": 1.0,
                    "position": 0,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()

        with caplog.at_level(logging.WARNING, logger="autopilot.plan.otio_export"):
            export_otio(edl, output, db)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("wipe" in msg for msg in warning_messages), (
            f"Expected a WARNING containing 'wipe', got: {warning_messages}"
        )


# -- Step 20: Multi-track transition isolation tests ---------------------------


class TestMultiTrackTransitionIsolation:
    """Verify transitions only apply to their designated track."""

    def test_transition_only_on_target_track(self, tmp_path):
        """Crossfade on track 1 does not bleed into track 2."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:10.000",
                    "out_timecode": "00:00:20.000",
                    "track": 1,
                },
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 2,
                },
            ],
            transitions=[
                {
                    "type": "crossfade",
                    "duration": 0.5,
                    "position": 0,
                    "track": 1,
                },
            ],
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        assert len(video_tracks) == 2

        # Track 1 (V1) should have exactly 1 transition between its 2 clips
        t1_transitions = [
            item for item in video_tracks[0] if isinstance(item, otio.schema.Transition)
        ]
        assert len(t1_transitions) == 1

        # Track 2 (V2) should have 0 transitions
        t2_transitions = [
            item for item in video_tracks[1] if isinstance(item, otio.schema.Transition)
        ]
        assert len(t2_transitions) == 0


# -- Step 11: Metadata preservation tests -------------------------------------


def _edl_with_metadata():
    """Build an EDL dict with all metadata fields populated."""
    return {
        "clips": [
            {
                "clip_id": "v1",
                "in_timecode": "00:00:00.000",
                "out_timecode": "00:00:10.000",
                "track": 1,
            },
        ],
        "transitions": [],
        "crop_modes": [
            {"clip_id": "v1", "mode": "16:9"},
        ],
        "titles": [
            {"text": "My Video Title", "position": "lower_third", "start": 0, "duration": 3},
        ],
        "audio_settings": [
            {"clip_id": "v1", "level_db": -6.0},
        ],
        "music": [
            {"track": "background_music.mp3", "level_db": -12.0},
        ],
        "voiceovers": [
            {"text": "Welcome to the show", "start": 0.0, "duration": 5.0},
        ],
        "broll_requests": [
            {"description": "aerial city shot", "duration": 3.0},
        ],
        "target_duration_seconds": 120,
    }


class TestMetadataPreservation:
    """Verify metadata is preserved on clips and timeline."""

    def test_clip_level_crop_mode(self, tmp_path):
        """Clip metadata includes crop_mode from EDL crop_modes."""
        from autopilot.plan.otio_export import export_otio

        edl = _edl_with_metadata()
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        clip = clips[0]

        assert "autopilot" in clip.metadata
        assert clip.metadata["autopilot"]["crop_mode"] == "16:9"

    def test_clip_level_audio_setting(self, tmp_path):
        """Clip metadata includes audio level_db from EDL audio_settings."""
        from autopilot.plan.otio_export import export_otio

        edl = _edl_with_metadata()
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        clip = clips[0]

        assert clip.metadata["autopilot"]["level_db"] == -6.0

    def test_timeline_level_titles(self, tmp_path):
        """Timeline metadata includes titles list."""
        from autopilot.plan.otio_export import export_otio

        edl = _edl_with_metadata()
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        assert "autopilot" in tl.metadata
        titles = tl.metadata["autopilot"]["titles"]
        assert len(titles) == 1
        assert titles[0]["text"] == "My Video Title"

    def test_timeline_level_music(self, tmp_path):
        """Timeline metadata includes music list."""
        from autopilot.plan.otio_export import export_otio

        edl = _edl_with_metadata()
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        music = tl.metadata["autopilot"]["music"]
        assert len(music) == 1
        assert music[0]["track"] == "background_music.mp3"

    def test_timeline_level_voiceovers(self, tmp_path):
        """Timeline metadata includes voiceovers list."""
        from autopilot.plan.otio_export import export_otio

        edl = _edl_with_metadata()
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        voiceovers = tl.metadata["autopilot"]["voiceovers"]
        assert len(voiceovers) == 1
        assert voiceovers[0]["text"] == "Welcome to the show"

    def test_timeline_level_broll_requests(self, tmp_path):
        """Timeline metadata includes broll_requests list."""
        from autopilot.plan.otio_export import export_otio

        edl = _edl_with_metadata()
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        broll = tl.metadata["autopilot"]["broll_requests"]
        assert len(broll) == 1
        assert broll[0]["description"] == "aerial city shot"

    def test_timeline_level_target_duration(self, tmp_path):
        """Timeline metadata includes target_duration_seconds."""
        from autopilot.plan.otio_export import export_otio

        edl = _edl_with_metadata()
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        assert tl.metadata["autopilot"]["target_duration_seconds"] == 120


# -- Step 13: Round-trip change detection tests --------------------------------


class TestDetectOtioChanges:
    """Verify detect_otio_changes detects modifications to .otio files."""

    def test_unmodified_returns_not_modified(self, tmp_path):
        """Unmodified .otio file returns modified=False."""
        from autopilot.plan.otio_export import detect_otio_changes, export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        result = detect_otio_changes(output, edl)
        assert result["modified"] is False
        assert result["changes"] == []

    def test_modified_clip_detected(self, tmp_path):
        """Modifying clip source_range in .otio is detected."""
        from autopilot.plan.otio_export import detect_otio_changes, export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        # Manually alter the .otio file (change clip source_range)
        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        # Change the clip's source range duration
        clips[0].source_range = otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30),
            duration=otio.opentime.RationalTime(600, 30),  # 20s instead of 10s
        )
        otio.adapters.write_to_file(tl, str(output))

        result = detect_otio_changes(output, edl)
        assert result["modified"] is True
        assert len(result["changes"]) > 0

    def test_edl_hash_stored_in_metadata(self, tmp_path):
        """export_otio stores edl_hash in timeline metadata."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()
        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        assert "edl_hash" in tl.metadata.get("autopilot", {})


# -- Step 15: Error handling tests ---------------------------------------------


class TestErrorHandling:
    """Verify proper error handling for edge cases."""

    def test_empty_clips_raises_otio_export_error(self, tmp_path):
        """export_otio raises OtioExportError for empty clips list."""
        from autopilot.plan.otio_export import OtioExportError, export_otio

        edl = _minimal_edl(clips=[])
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()

        with pytest.raises(OtioExportError, match="[Nn]o clips"):
            export_otio(edl, output, db)

    def test_missing_media_uses_fallback(self, tmp_path):
        """Clip not in catalog uses clip_id as fallback target_url."""
        from autopilot.plan.otio_export import export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "unknown_clip",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:05.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = MagicMock()
        db.get_media.return_value = None  # not found

        export_otio(edl, output, db)

        tl = otio.adapters.read_from_file(str(output))
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        assert clips[0].media_reference.target_url == "unknown_clip"

    def test_detect_otio_changes_nonexistent_file(self, tmp_path):
        """detect_otio_changes raises OtioExportError for nonexistent file."""
        from autopilot.plan.otio_export import OtioExportError, detect_otio_changes

        with pytest.raises(OtioExportError, match="not found"):
            detect_otio_changes(tmp_path / "nonexistent.otio", {})

    def test_unwritable_output_raises_otio_export_error(self, tmp_path):
        """export_otio raises OtioExportError for unwritable directory."""
        from autopilot.plan.otio_export import OtioExportError, export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                }
            ]
        )
        # Use a path that should not be writable
        output = Path("/nonexistent_dir/test.otio")
        db = _mock_db_for_clips()

        with pytest.raises(OtioExportError):
            export_otio(edl, output, db)

    def test_db_get_media_exception_wrapped(self, tmp_path):
        """db.get_media raising an exception is wrapped as OtioExportError."""
        from autopilot.plan.otio_export import OtioExportError, export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = MagicMock()
        db.get_media.side_effect = RuntimeError("DB locked")

        with pytest.raises(OtioExportError, match="v1"):
            export_otio(edl, output, db)


# -- Step 24: KeyError wrapping tests -----------------------------------------


class TestKeyErrorWrapping:
    """Verify bare bracket access on EDL dicts is wrapped as OtioExportError."""

    def test_crop_modes_missing_clip_id(self, tmp_path):
        """crop_modes entry missing 'clip_id' raises OtioExportError, not KeyError."""
        from autopilot.plan.otio_export import OtioExportError, export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                }
            ]
        )
        edl["crop_modes"] = [{"mode": "16:9"}]  # missing clip_id
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()

        with pytest.raises(OtioExportError, match="crop_modes|clip_id"):
            export_otio(edl, output, db)

    def test_clip_missing_clip_id(self, tmp_path):
        """clip dict missing 'clip_id' raises OtioExportError, not KeyError."""
        from autopilot.plan.otio_export import OtioExportError, export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()

        with pytest.raises(OtioExportError, match="clip_id"):
            export_otio(edl, output, db)

    def test_clip_missing_in_timecode(self, tmp_path):
        """clip dict missing 'in_timecode' raises OtioExportError, not KeyError."""
        from autopilot.plan.otio_export import OtioExportError, export_otio

        edl = _minimal_edl(
            clips=[
                {
                    "clip_id": "v1",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                }
            ]
        )
        output = tmp_path / "test.otio"
        db = _mock_db_for_clips()

        with pytest.raises(OtioExportError, match="in_timecode"):
            export_otio(edl, output, db)


# -- Step 17: Integration test ------------------------------------------------


class TestIntegration:
    """Full integration test with real DB, complete EDL, and round-trip."""

    def test_full_round_trip(self, catalog_db, tmp_path):
        """End-to-end: seed DB, export OTIO, round-trip, detect no changes, store in DB."""
        import json

        from autopilot.plan.otio_export import detect_otio_changes, export_otio

        # Seed DB with 2 media files
        catalog_db.insert_media(
            "v1",
            "/media/interview.mp4",
            fps=30.0,
            duration_seconds=120.0,
        )
        catalog_db.insert_media(
            "v2",
            "/media/broll_city.mp4",
            fps=24.0,
            duration_seconds=60.0,
        )

        # Create a full EDL
        edl = {
            "clips": [
                {
                    "clip_id": "v1",
                    "in_timecode": "00:00:05.000",
                    "out_timecode": "00:00:35.000",
                    "track": 1,
                },
                {
                    "clip_id": "v2",
                    "in_timecode": "00:00:00.000",
                    "out_timecode": "00:00:10.000",
                    "track": 1,
                },
            ],
            "transitions": [
                {
                    "type": "crossfade",
                    "duration": 0.5,
                    "position": 0,
                },
            ],
            "crop_modes": [
                {"clip_id": "v1", "mode": "16:9"},
                {"clip_id": "v2", "mode": "letterbox"},
            ],
            "titles": [
                {"text": "Welcome", "position": "center", "start": 0, "duration": 3},
            ],
            "audio_settings": [
                {"clip_id": "v1", "level_db": -3.0},
                {"clip_id": "v2", "level_db": -6.0},
            ],
            "music": [
                {"track": "ambient.mp3", "level_db": -18.0},
            ],
            "voiceovers": [
                {"text": "In this video...", "start": 0.0, "duration": 5.0},
            ],
            "broll_requests": [
                {"description": "sunset timelapse", "duration": 4.0},
            ],
            "target_duration_seconds": 45,
        }

        # Export
        output = tmp_path / "edit.otio"
        result = export_otio(edl, output, catalog_db)
        assert result == output
        assert output.exists()

        # Read back and verify structure
        tl = otio.adapters.read_from_file(str(output))
        assert tl.name == "autopilot_edit"

        # Check video tracks
        video_tracks = [t for t in tl.tracks if t.kind == otio.schema.TrackKind.Video]
        assert len(video_tracks) == 1  # both clips on track 1

        # Check clips (2 clips)
        all_clips = [c for c in video_tracks[0] if isinstance(c, otio.schema.Clip)]
        assert len(all_clips) == 2

        # v2 should come first (sorted by in_timecode: 00:00:00 < 00:00:05)
        assert all_clips[0].name == "v2"
        assert all_clips[0].media_reference.target_url == "/media/broll_city.mp4"
        assert all_clips[1].name == "v1"
        assert all_clips[1].media_reference.target_url == "/media/interview.mp4"

        # Check transitions
        transitions = [item for item in video_tracks[0] if isinstance(item, otio.schema.Transition)]
        assert len(transitions) == 1
        assert transitions[0].transition_type == otio.schema.Transition.Type.SMPTE_Dissolve

        # Check per-clip metadata
        assert all_clips[0].metadata["autopilot"]["crop_mode"] == "letterbox"
        assert all_clips[0].metadata["autopilot"]["level_db"] == -6.0
        assert all_clips[1].metadata["autopilot"]["crop_mode"] == "16:9"
        assert all_clips[1].metadata["autopilot"]["level_db"] == -3.0

        # Check timeline metadata
        tl_meta = tl.metadata["autopilot"]
        assert tl_meta["target_duration_seconds"] == 45
        assert len(tl_meta["titles"]) == 1
        assert len(tl_meta["music"]) == 1
        assert len(tl_meta["voiceovers"]) == 1
        assert len(tl_meta["broll_requests"]) == 1
        assert "edl_hash" in tl_meta

        # Detect no changes
        change_result = detect_otio_changes(output, edl)
        assert change_result["modified"] is False
        assert change_result["changes"] == []

        # Store otio_path in DB via upsert_edit_plan
        catalog_db.insert_narrative("test_narrative", title="Test")
        catalog_db.upsert_edit_plan(
            narrative_id="test_narrative",
            edl_json=json.dumps(edl),
            otio_path=str(output),
        )

        # Verify DB has correct otio_path
        cur = catalog_db.conn.execute(
            "SELECT otio_path FROM edit_plans WHERE narrative_id = ?",
            ("test_narrative",),
        )
        row = cur.fetchone()
        assert row is not None
        assert row["otio_path"] == str(output)
