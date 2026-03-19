"""Tests for narrative organization (autopilot.organize.narratives)."""

from __future__ import annotations

import inspect
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

# -- Step 1: Public API surface tests ------------------------------------------


class TestPublicAPI:
    """Verify public API surface and type signatures."""

    def test_narrative_error_importable(self):
        """NarrativeError is importable from narratives module."""
        from autopilot.organize.narratives import NarrativeError

        assert NarrativeError is not None

    def test_narrative_error_is_exception(self):
        """NarrativeError is a subclass of Exception with message."""
        from autopilot.organize.narratives import NarrativeError

        assert issubclass(NarrativeError, Exception)
        err = NarrativeError("test message")
        assert str(err) == "test message"

    def test_narrative_dataclass_fields(self):
        """Narrative has all required fields."""
        from autopilot.organize.narratives import Narrative

        n = Narrative()
        assert hasattr(n, "narrative_id")
        assert hasattr(n, "title")
        assert hasattr(n, "description")
        assert hasattr(n, "proposed_duration_seconds")
        assert hasattr(n, "activity_cluster_ids")
        assert hasattr(n, "arc")
        assert hasattr(n, "emotional_journey")
        assert hasattr(n, "reasoning")
        assert hasattr(n, "status")

    def test_narrative_default_status(self):
        """Narrative defaults to status='proposed'."""
        from autopilot.organize.narratives import Narrative

        n = Narrative()
        assert n.status == "proposed"

    def test_narrative_activity_cluster_ids_is_list(self):
        """Narrative.activity_cluster_ids defaults to empty list."""
        from autopilot.organize.narratives import Narrative

        n = Narrative()
        assert isinstance(n.activity_cluster_ids, list)

    def test_narrative_arc_is_dict(self):
        """Narrative.arc defaults to empty dict."""
        from autopilot.organize.narratives import Narrative

        n = Narrative()
        assert isinstance(n.arc, dict)

    def test_build_master_storyboard_signature(self):
        """build_master_storyboard has db parameter and returns str."""
        from autopilot.organize.narratives import build_master_storyboard

        sig = inspect.signature(build_master_storyboard)
        params = list(sig.parameters.keys())
        assert "db" in params
        assert sig.return_annotation in (str, "str")

    def test_propose_narratives_signature(self):
        """propose_narratives has storyboard, db, config parameters."""
        from autopilot.organize.narratives import propose_narratives

        sig = inspect.signature(propose_narratives)
        params = list(sig.parameters.keys())
        assert "storyboard" in params
        assert "db" in params
        assert "config" in params

    def test_format_for_review_exists(self):
        """format_for_review function exists and is callable."""
        from autopilot.organize.narratives import format_for_review

        assert callable(format_for_review)

    def test_format_for_review_signature(self):
        """format_for_review has narratives parameter and returns str."""
        from autopilot.organize.narratives import format_for_review

        sig = inspect.signature(format_for_review)
        params = list(sig.parameters.keys())
        assert "narratives" in params
        assert sig.return_annotation in (str, "str")
