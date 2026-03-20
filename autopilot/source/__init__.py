"""Shared request dataclasses for the asset sourcing stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

__all__ = [
    "AssetRequest",
    "BrollRequest",
    "MusicRequest",
    "VoiceoverRequest",
]


@dataclass
class MusicRequest:
    """A request for a music track matching a mood, derived from EDL add_music entries."""

    mood: str
    duration: float
    start_time: str
    resolved_path: Path | None = None


@dataclass
class BrollRequest:
    """A request for B-roll footage, derived from EDL request_broll entries."""

    description: str
    duration: float
    start_time: str
    resolved_path: Path | None = None


@dataclass
class VoiceoverRequest:
    """A request for TTS voiceover narration, derived from EDL add_voiceover entries."""

    text: str
    start_time: str
    duration: float
    resolved_path: Path | None = None


AssetRequest = Union[MusicRequest, BrollRequest, VoiceoverRequest]
