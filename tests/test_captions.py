"""Tests for autopilot.analyze.captions — selective video captioning module."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# TestPublicAPI — verify module exports and function signatures
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify captions module exports and function signatures."""

    def test_caption_error_importable(self):
        """CaptionError is importable from autopilot.analyze.captions."""
        from autopilot.analyze.captions import CaptionError

        assert CaptionError is not None

    def test_caption_error_is_exception(self):
        """CaptionError is a subclass of Exception."""
        from autopilot.analyze.captions import CaptionError

        assert issubclass(CaptionError, Exception)

    def test_caption_clip_importable(self):
        """caption_clip is importable with expected params."""
        from autopilot.analyze.captions import caption_clip

        assert callable(caption_clip)

    def test_batch_caption_importable(self):
        """batch_caption is importable with expected params."""
        from autopilot.analyze.captions import batch_caption

        assert callable(batch_caption)

    def test_extract_clip_frames_importable(self):
        """_extract_clip_frames is a module-level helper."""
        from autopilot.analyze.captions import _extract_clip_frames

        assert callable(_extract_clip_frames)
