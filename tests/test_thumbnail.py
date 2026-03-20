"""Tests for thumbnail extraction (autopilot.upload.thumbnail)."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Public API surface tests
# ---------------------------------------------------------------------------


class TestThumbnailPublicAPI:
    """Verify ThumbnailError, extract_best_thumbnail surface."""

    def test_thumbnail_error_importable(self):
        """ThumbnailError is importable from thumbnail module."""
        from autopilot.upload.thumbnail import ThumbnailError

        assert ThumbnailError is not None

    def test_thumbnail_error_is_exception(self):
        """ThumbnailError is a subclass of Exception."""
        from autopilot.upload.thumbnail import ThumbnailError

        assert issubclass(ThumbnailError, Exception)
        err = ThumbnailError("test error")
        assert str(err) == "test error"

    def test_extract_best_thumbnail_signature(self):
        """extract_best_thumbnail has narrative_id, video_path, db params."""
        from autopilot.upload.thumbnail import extract_best_thumbnail

        sig = inspect.signature(extract_best_thumbnail)
        params = list(sig.parameters.keys())
        assert "narrative_id" in params
        assert "video_path" in params
        assert "db" in params

    def test_all_exports(self):
        """__all__ includes ThumbnailError and extract_best_thumbnail."""
        from autopilot.upload import thumbnail

        assert "ThumbnailError" in thumbnail.__all__
        assert "extract_best_thumbnail" in thumbnail.__all__


# ---------------------------------------------------------------------------
# Mock helpers for cv2
# ---------------------------------------------------------------------------


def _setup_cv2_mock():
    """Create a mock cv2 module for testing."""
    mock_cv2 = MagicMock()
    mock_cv2.CV_64F = 6
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    return mock_cv2


# ---------------------------------------------------------------------------
# Frame scoring tests
# ---------------------------------------------------------------------------


class TestScoringFunctions:
    """Verify individual frame scoring functions."""

    def test_sharpness_score_uses_laplacian_variance(self):
        """Sharpness is computed from Laplacian variance."""
        mock_cv2 = _setup_cv2_mock()
        # Simulate a Laplacian result with known variance
        laplacian_result = np.array([[10.0, 20.0], [30.0, 40.0]])
        mock_cv2.Laplacian.return_value = laplacian_result

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            from autopilot.upload.thumbnail import _sharpness_score

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            score = _sharpness_score(frame)

        mock_cv2.Laplacian.assert_called_once()
        assert isinstance(score, float)
        assert score > 0

    def test_rule_of_thirds_score_high_at_intersections(self):
        """Subject at rule-of-thirds intersection scores higher than center."""
        from autopilot.upload.thumbnail import _rule_of_thirds_score

        frame_shape = (1080, 1920, 3)
        # Detection bbox centered at a thirds intersection point (~640, 360)
        thirds_dets = [
            {"bbox": [590, 310, 690, 410], "confidence": 0.9}
        ]
        # Detection bbox centered at dead center (960, 540)
        center_dets = [
            {"bbox": [910, 490, 1010, 590], "confidence": 0.9}
        ]

        thirds_score = _rule_of_thirds_score(frame_shape, thirds_dets)
        center_score = _rule_of_thirds_score(frame_shape, center_dets)

        assert thirds_score > center_score

    def test_detection_confidence_score_aggregates(self):
        """Uses max confidence from list of detections."""
        from autopilot.upload.thumbnail import _detection_confidence_score

        detections = [
            {"class_name": "person", "confidence": 0.7},
            {"class_name": "car", "confidence": 0.95},
            {"class_name": "dog", "confidence": 0.8},
        ]
        score = _detection_confidence_score(detections)
        assert score == pytest.approx(0.95)

    def test_detection_confidence_empty(self):
        """Returns 0 for empty detections list."""
        from autopilot.upload.thumbnail import _detection_confidence_score

        assert _detection_confidence_score([]) == 0.0

    def test_combined_score_weighted(self):
        """Combined score uses weights summing to 1.0."""
        from autopilot.upload.thumbnail import _combined_score

        score = _combined_score(
            sharpness=1.0, thirds=1.0, confidence=1.0
        )
        assert score == pytest.approx(1.0)

        # Weights: 0.3 sharpness, 0.3 thirds, 0.4 confidence
        score2 = _combined_score(
            sharpness=0.5, thirds=0.5, confidence=0.5
        )
        assert score2 == pytest.approx(0.5)
