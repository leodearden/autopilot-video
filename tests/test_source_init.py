"""Tests for autopilot.source request dataclasses and exports."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path


class TestMusicRequest:
    """MusicRequest dataclass has correct fields."""

    def test_importable(self):
        from autopilot.source import MusicRequest

        assert MusicRequest is not None

    def test_is_dataclass(self):
        from autopilot.source import MusicRequest

        assert hasattr(MusicRequest, "__dataclass_fields__")

    def test_fields(self):
        from autopilot.source import MusicRequest

        field_names = {f.name for f in fields(MusicRequest)}
        assert "mood" in field_names
        assert "duration" in field_names
        assert "start_time" in field_names
        assert "resolved_path" in field_names

    def test_resolved_path_defaults_none(self):
        from autopilot.source import MusicRequest

        req = MusicRequest(mood="upbeat", duration=30.0, start_time="00:01:00.000")
        assert req.resolved_path is None

    def test_resolved_path_accepts_path(self):
        from autopilot.source import MusicRequest

        p = Path("/tmp/music.wav")
        req = MusicRequest(mood="upbeat", duration=30.0, start_time="00:01:00.000", resolved_path=p)
        assert req.resolved_path == p


class TestBrollRequest:
    """BrollRequest dataclass has correct fields."""

    def test_importable(self):
        from autopilot.source import BrollRequest

        assert BrollRequest is not None

    def test_is_dataclass(self):
        from autopilot.source import BrollRequest

        assert hasattr(BrollRequest, "__dataclass_fields__")

    def test_fields(self):
        from autopilot.source import BrollRequest

        field_names = {f.name for f in fields(BrollRequest)}
        assert "description" in field_names
        assert "duration" in field_names
        assert "start_time" in field_names
        assert "resolved_path" in field_names

    def test_resolved_path_defaults_none(self):
        from autopilot.source import BrollRequest

        req = BrollRequest(
            description="aerial view of mountains",
            duration=5.0,
            start_time="00:02:00.000",
        )
        assert req.resolved_path is None


class TestVoiceoverRequest:
    """VoiceoverRequest dataclass has correct fields."""

    def test_importable(self):
        from autopilot.source import VoiceoverRequest

        assert VoiceoverRequest is not None

    def test_is_dataclass(self):
        from autopilot.source import VoiceoverRequest

        assert hasattr(VoiceoverRequest, "__dataclass_fields__")

    def test_fields(self):
        from autopilot.source import VoiceoverRequest

        field_names = {f.name for f in fields(VoiceoverRequest)}
        assert "text" in field_names
        assert "start_time" in field_names
        assert "duration" in field_names
        assert "resolved_path" in field_names

    def test_resolved_path_defaults_none(self):
        from autopilot.source import VoiceoverRequest

        req = VoiceoverRequest(text="Hello world", start_time="00:00:05.000", duration=3.0)
        assert req.resolved_path is None


class TestAssetRequest:
    """AssetRequest is a Union type alias covering all request types."""

    def test_importable(self):
        from autopilot.source import AssetRequest

        assert AssetRequest is not None

    def test_music_request_is_asset_request(self):
        from autopilot.source import AssetRequest, MusicRequest

        req = MusicRequest(mood="chill", duration=10.0, start_time="00:00:00.000")
        assert isinstance(req, MusicRequest)
        # Union type checking via get_args
        import typing

        args = typing.get_args(AssetRequest)
        assert MusicRequest in args

    def test_broll_request_is_asset_request(self):
        import typing

        from autopilot.source import AssetRequest, BrollRequest

        args = typing.get_args(AssetRequest)
        assert BrollRequest in args

    def test_voiceover_request_is_asset_request(self):
        import typing

        from autopilot.source import AssetRequest, VoiceoverRequest

        args = typing.get_args(AssetRequest)
        assert VoiceoverRequest in args


class TestSourceExports:
    """__all__ exports include all request types."""

    def test_all_exports(self):
        from autopilot import source

        assert "MusicRequest" in source.__all__
        assert "BrollRequest" in source.__all__
        assert "VoiceoverRequest" in source.__all__
        assert "AssetRequest" in source.__all__
