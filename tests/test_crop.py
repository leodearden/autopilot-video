"""Tests for auto-crop viewport computation (autopilot.render.crop)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest


class TestPublicAPI:
    """Verify public API surface: imports, signatures, and __all__ exports."""

    def test_crop_error_importable_and_is_exception(self) -> None:
        """CropError should be importable and be an Exception subclass."""
        from autopilot.render.crop import CropError

        assert issubclass(CropError, Exception)

    def test_compute_crop_path_importable(self) -> None:
        """compute_crop_path should be importable."""
        from autopilot.render.crop import compute_crop_path

        assert callable(compute_crop_path)

    def test_compute_crop_path_signature(self) -> None:
        """compute_crop_path should accept (media_id, target_aspect, db, config, edl_entry)."""
        from autopilot.render.crop import compute_crop_path

        sig = inspect.signature(compute_crop_path)
        param_names = list(sig.parameters.keys())
        assert param_names == [
            "media_id",
            "target_aspect",
            "db",
            "config",
            "edl_entry",
        ]

    def test_compute_crop_path_return_annotation(self) -> None:
        """compute_crop_path should be annotated to return np.ndarray."""
        from autopilot.render.crop import compute_crop_path

        sig = inspect.signature(compute_crop_path)
        # With PEP 563 (from __future__ import annotations), annotations are strings
        assert sig.return_annotation in (np.ndarray, "np.ndarray")

    def test_all_exports(self) -> None:
        """__all__ should export CropError and compute_crop_path."""
        from autopilot.render import crop

        assert hasattr(crop, "__all__")
        assert set(crop.__all__) == {"CropError", "compute_crop_path"}


class TestComputeCropDimensions:
    """Tests for _compute_crop_dimensions helper."""

    def test_16_9_from_square_source(self) -> None:
        """4096x4096 source with '16:9' -> maximize width, compute height."""
        from autopilot.render.crop import _compute_crop_dimensions

        crop_w, crop_h = _compute_crop_dimensions(4096, 4096, "16:9")
        assert crop_w == 4096
        assert crop_h == 4096 * 9 // 16  # 2304

    def test_9_16_from_square_source(self) -> None:
        """4096x4096 source with '9:16' -> maximize height, compute width."""
        from autopilot.render.crop import _compute_crop_dimensions

        crop_w, crop_h = _compute_crop_dimensions(4096, 4096, "9:16")
        assert crop_h == 4096
        assert crop_w == 4096 * 9 // 16  # 2304

    def test_16_9_from_landscape_source(self) -> None:
        """1920x1080 source with '16:9' -> full frame (already 16:9)."""
        from autopilot.render.crop import _compute_crop_dimensions

        crop_w, crop_h = _compute_crop_dimensions(1920, 1080, "16:9")
        assert crop_w == 1920
        assert crop_h == 1080

    def test_unknown_aspect_raises_crop_error(self) -> None:
        """Unknown aspect string raises CropError."""
        from autopilot.render.crop import CropError, _compute_crop_dimensions

        with pytest.raises(CropError, match="aspect"):
            _compute_crop_dimensions(4096, 4096, "potato")


class TestSelectSubjectTrack:
    """Tests for _select_subject_track helper."""

    def test_explicit_track_id_returned(self) -> None:
        """EDL with explicit integer subject_track_id returns that ID."""
        from autopilot.render.crop import _select_subject_track

        detections = [
            [{"track_id": 1, "bbox_xywh": [100, 100, 50, 50], "class": "person", "confidence": 0.9}],
            [{"track_id": 2, "bbox_xywh": [200, 200, 80, 80], "class": "car", "confidence": 0.8}],
        ]
        edl_entry = {"subject_track_id": 5}
        assert _select_subject_track(detections, edl_entry) == 5

    def test_auto_selects_largest_track(self) -> None:
        """Without explicit ID, selects track with highest cumulative bbox area * frame count."""
        from autopilot.render.crop import _select_subject_track

        # Track 1: present on 3 frames, bbox area = 50*50 = 2500 each -> total 7500
        # Track 2: present on 2 frames, bbox area = 100*100 = 10000 each -> total 20000
        det_frame1 = [
            {"track_id": 1, "bbox_xywh": [100, 100, 50, 50], "class": "person", "confidence": 0.9},
            {"track_id": 2, "bbox_xywh": [200, 200, 100, 100], "class": "car", "confidence": 0.8},
        ]
        det_frame2 = [
            {"track_id": 1, "bbox_xywh": [110, 110, 50, 50], "class": "person", "confidence": 0.9},
            {"track_id": 2, "bbox_xywh": [210, 210, 100, 100], "class": "car", "confidence": 0.8},
        ]
        det_frame3 = [
            {"track_id": 1, "bbox_xywh": [120, 120, 50, 50], "class": "person", "confidence": 0.9},
        ]
        detections = [det_frame1, det_frame2, det_frame3]
        edl_entry: dict = {}
        assert _select_subject_track(detections, edl_entry) == 2

    def test_none_track_id_triggers_auto(self) -> None:
        """subject_track_id=None triggers auto-selection."""
        from autopilot.render.crop import _select_subject_track

        detections = [
            [{"track_id": 7, "bbox_xywh": [100, 100, 60, 60], "class": "person", "confidence": 0.9}],
        ]
        edl_entry = {"subject_track_id": None}
        assert _select_subject_track(detections, edl_entry) == 7

    def test_empty_detections_raises_crop_error(self) -> None:
        """Empty detections list raises CropError."""
        from autopilot.render.crop import CropError, _select_subject_track

        with pytest.raises(CropError, match="[Nn]o.*detect"):
            _select_subject_track([], {})


class TestComputeRawCenter:
    """Tests for _compute_raw_center single-subject rule-of-thirds framing."""

    def test_subject_at_center_right_third(self) -> None:
        """Subject at frame center with right-third -> crop shifts left so subject is at 2/3."""
        from autopilot.render.crop import _compute_raw_center

        # Subject centered at (2048, 2048), bbox 200x400
        bbox = [2048.0, 2048.0, 200.0, 400.0]
        crop_w, crop_h = 4096, 2304
        cx, cy = _compute_raw_center(bbox, crop_w, crop_h, "right")
        # Subject center should be at 2/3 of crop width from left
        # So crop center_x = subject_cx - (2/3 - 1/2) * crop_w
        #                   = 2048 - (1/6) * 4096 = 2048 - 682.67 = ~1365.3
        expected_cx = bbox[0] - (2 / 3 - 0.5) * crop_w
        assert cx == pytest.approx(expected_cx, abs=1.0)

    def test_subject_at_center_left_third(self) -> None:
        """Subject at center with left-third -> crop shifts right so subject is at 1/3."""
        from autopilot.render.crop import _compute_raw_center

        bbox = [2048.0, 2048.0, 200.0, 400.0]
        crop_w, crop_h = 4096, 2304
        cx, cy = _compute_raw_center(bbox, crop_w, crop_h, "left")
        expected_cx = bbox[0] - (1 / 3 - 0.5) * crop_w
        assert cx == pytest.approx(expected_cx, abs=1.0)

    def test_vertical_eyes_at_top_third(self) -> None:
        """Eyes (top 1/3 of bbox) should be positioned at 1/3 from crop top."""
        from autopilot.render.crop import _compute_raw_center

        # Subject at (2048, 2048), bbox 200x600
        # Eye line is at subject_cy - bbox_h/3 = 2048 - 200 = 1848
        bbox = [2048.0, 2048.0, 200.0, 600.0]
        crop_w, crop_h = 4096, 2304
        _, cy = _compute_raw_center(bbox, crop_w, crop_h, "right")
        eye_y = bbox[1] - bbox[3] / 3.0
        # Eye should be at 1/3 from top of crop: cy - crop_h/2 + crop_h/3 = eye_y
        # So cy = eye_y + crop_h/2 - crop_h/3 = eye_y + crop_h/6
        expected_cy = eye_y + crop_h / 6.0
        assert cy == pytest.approx(expected_cy, abs=1.0)

    def test_subject_near_edge(self) -> None:
        """Subject near left edge -> raw center may be outside frame (pre-clamping)."""
        from autopilot.render.crop import _compute_raw_center

        bbox = [100.0, 2048.0, 50.0, 100.0]
        crop_w, crop_h = 4096, 2304
        cx, _ = _compute_raw_center(bbox, crop_w, crop_h, "right")
        # cx can be negative (clamping happens later)
        assert isinstance(cx, float)


class TestComputeMultiSubjectCenter:
    """Tests for _compute_multi_subject_center helper."""

    def test_two_subjects_centered(self) -> None:
        """Two subjects at known positions -> center crop on bounding box."""
        from autopilot.render.crop import _compute_multi_subject_center

        # Subject A at (1000, 1000, 200, 200), Subject B at (3000, 3000, 200, 200)
        bboxes = [
            [1000.0, 1000.0, 200.0, 200.0],
            [3000.0, 3000.0, 200.0, 200.0],
        ]
        crop_w, crop_h = 4096, 2304
        cx, cy = _compute_multi_subject_center(bboxes, crop_w, crop_h)
        # Center of bounding box containing both subjects: (2000, 2000)
        assert cx == pytest.approx(2000.0, abs=1.0)
        assert cy == pytest.approx(2000.0, abs=1.0)

    def test_single_subject_centers_on_it(self) -> None:
        """Single subject in multi-subject path -> center on that subject."""
        from autopilot.render.crop import _compute_multi_subject_center

        bboxes = [[2048.0, 2048.0, 300.0, 400.0]]
        crop_w, crop_h = 4096, 2304
        cx, cy = _compute_multi_subject_center(bboxes, crop_w, crop_h)
        assert cx == pytest.approx(2048.0, abs=1.0)
        assert cy == pytest.approx(2048.0, abs=1.0)
