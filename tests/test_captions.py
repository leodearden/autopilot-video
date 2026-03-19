"""Tests for autopilot.analyze.captions — selective video captioning module."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# -- Mock helpers --------------------------------------------------------------


def _make_mock_capture(
    fps: float = 30.0,
    total_frames: int = 900,
    width: int = 1920,
    height: int = 1080,
) -> MagicMock:
    """Create a MagicMock mimicking cv2.VideoCapture."""
    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4

    cap = MagicMock()
    cap.isOpened.return_value = True

    def get_prop(prop_id):
        prop_map = {
            CAP_PROP_FPS: fps,
            CAP_PROP_FRAME_COUNT: total_frames,
            CAP_PROP_FRAME_WIDTH: width,
            CAP_PROP_FRAME_HEIGHT: height,
        }
        return prop_map.get(prop_id, 0.0)

    cap.get.side_effect = get_prop

    # BGR frame (cv2 native format)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = 100  # B channel
    frame[:, :, 2] = 200  # R channel
    cap.read.return_value = (True, frame)
    cap.set.return_value = True
    return cap


def _make_mock_cv2():
    """Create a MagicMock cv2 module with correct CAP_PROP constants."""
    mock_cv2 = MagicMock()
    mock_cv2.CAP_PROP_FPS = 5
    mock_cv2.CAP_PROP_FRAME_COUNT = 7
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CAP_PROP_POS_FRAMES = 1
    return mock_cv2


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


# ---------------------------------------------------------------------------
# TestExtractClipFrames — frame extraction from video clips
# ---------------------------------------------------------------------------


class TestExtractClipFrames:
    """Tests for _extract_clip_frames() helper."""

    def test_extracts_correct_number_of_frames(self):
        """8 frames requested from 30s clip returns 8 PIL Images."""
        from PIL import Image

        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=900)
        mock_cv2.VideoCapture.return_value = cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            # Force reimport to pick up mocked cv2
            import importlib

            from autopilot.analyze import captions

            importlib.reload(captions)
            frames = captions._extract_clip_frames(
                Path("/test/video.mp4"), 0.0, 30.0, fps=30.0, num_frames=8
            )

        assert len(frames) == 8
        assert all(isinstance(f, Image.Image) for f in frames)

    def test_frame_times_evenly_spaced(self):
        """For start=10.0, end=40.0, num_frames=4, verify evenly-spaced frame positions."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=1200)
        mock_cv2.VideoCapture.return_value = cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            import importlib

            from autopilot.analyze import captions

            importlib.reload(captions)
            captions._extract_clip_frames(
                Path("/test/video.mp4"), 10.0, 40.0, fps=30.0, num_frames=4
            )

        # Expected timestamps: evenly spaced in [10, 40]
        # 10.0, 20.0, 30.0, 40.0 -> frame indices 300, 600, 900, 1200
        set_calls = [c for c in cap.set.call_args_list if c[0][0] == 1]  # CAP_PROP_POS_FRAMES=1
        assert len(set_calls) == 4
        frame_indices = [c[0][1] for c in set_calls]
        # All should be in the range [300, 1200]
        assert all(300 <= idx <= 1200 for idx in frame_indices)

    def test_handles_read_failure(self):
        """When cap.read() returns (False, None) for some frames, returns only successful."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=900)
        mock_cv2.VideoCapture.return_value = cap

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Alternate: success, failure, success, failure, ...
        cap.read.side_effect = [
            (True, frame),
            (False, None),
            (True, frame),
            (False, None),
            (True, frame),
            (False, None),
            (True, frame),
            (False, None),
        ]

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            import importlib

            from autopilot.analyze import captions

            importlib.reload(captions)
            frames = captions._extract_clip_frames(
                Path("/test/video.mp4"), 0.0, 30.0, fps=30.0, num_frames=8
            )

        assert len(frames) == 4  # Only successful reads

    def test_empty_range_returns_empty(self):
        """start_time == end_time returns empty list."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture()
        mock_cv2.VideoCapture.return_value = cap

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            import importlib

            from autopilot.analyze import captions

            importlib.reload(captions)
            frames = captions._extract_clip_frames(
                Path("/test/video.mp4"), 5.0, 5.0, fps=30.0, num_frames=8
            )

        assert frames == []

    def test_converts_bgr_to_rgb(self):
        """Verify cv2 BGR frame is converted to RGB PIL Image."""
        mock_cv2 = _make_mock_cv2()
        cap = _make_mock_capture(fps=30.0, total_frames=900)
        mock_cv2.VideoCapture.return_value = cap

        # BGR frame with B=100, G=0, R=200
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr_frame[:, :, 0] = 100  # Blue channel
        bgr_frame[:, :, 2] = 200  # Red channel
        cap.read.return_value = (True, bgr_frame)

        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            import importlib

            from autopilot.analyze import captions

            importlib.reload(captions)
            frames = captions._extract_clip_frames(
                Path("/test/video.mp4"), 0.0, 10.0, fps=30.0, num_frames=1
            )

        assert len(frames) == 1
        img = frames[0]
        pixel = img.getpixel((0, 0))
        # After BGR->RGB conversion: R=200, G=0, B=100
        assert pixel == (200, 0, 100)


# ---------------------------------------------------------------------------
# TestCaptionClip — caption_clip() core generation path
# ---------------------------------------------------------------------------


def _make_mock_model_and_processor():
    """Create mock transformers model and processor."""
    model = MagicMock()
    processor = MagicMock()

    # model.generate returns token ids
    model.generate.return_value = MagicMock()

    # processor.batch_decode returns caption strings
    processor.batch_decode.return_value = ["A person walking on a beach"]

    return model, processor


def _make_mock_scheduler(model_obj=None):
    """Create mock GPUScheduler that yields model_obj from context manager."""
    scheduler = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=model_obj)
    ctx.__exit__ = MagicMock(return_value=False)
    scheduler.model.return_value = ctx
    return scheduler


def _make_mock_config():
    """Create mock ModelConfig with caption_model."""
    config = MagicMock()
    config.caption_model = "Qwen/Qwen2.5-VL-7B-Instruct"
    return config


class TestCaptionClip:
    """Tests for caption_clip() core path."""

    def test_returns_caption_string(self, catalog_db):
        """caption_clip returns generated caption string."""
        from PIL import Image as PILImage

        from autopilot.analyze.captions import caption_clip

        # Insert media with fps/duration
        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        # Mock _extract_clip_frames to return dummy PIL images
        dummy_frames = [PILImage.new("RGB", (100, 100)) for _ in range(8)]

        with (
            patch("autopilot.analyze.captions._extract_clip_frames", return_value=dummy_frames),
            patch.object(Path, "exists", return_value=True),
        ):
            result = caption_clip(
                "m1", Path("/test/video.mp4"), 0.0, 30.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_stores_caption_in_db(self, catalog_db):
        """After caption_clip, db.get_caption returns the caption."""
        from PIL import Image as PILImage

        from autopilot.analyze.captions import caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        dummy_frames = [PILImage.new("RGB", (100, 100)) for _ in range(8)]

        with (
            patch("autopilot.analyze.captions._extract_clip_frames", return_value=dummy_frames),
            patch.object(Path, "exists", return_value=True),
        ):
            caption_clip(
                "m1", Path("/test/video.mp4"), 0.0, 30.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

        stored = catalog_db.get_caption("m1", 0.0, 30.0)
        assert stored is not None
        assert len(stored["caption"]) > 0

    def test_uses_scheduler(self, catalog_db):
        """Verify scheduler.model() context manager is called."""
        from PIL import Image as PILImage

        from autopilot.analyze.captions import caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        dummy_frames = [PILImage.new("RGB", (100, 100)) for _ in range(8)]

        with (
            patch("autopilot.analyze.captions._extract_clip_frames", return_value=dummy_frames),
            patch.object(Path, "exists", return_value=True),
        ):
            caption_clip(
                "m1", Path("/test/video.mp4"), 0.0, 30.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

        scheduler.model.assert_called_once_with("Qwen/Qwen2.5-VL-7B-Instruct")

    def test_extracts_frames_from_clip_segment(self, catalog_db):
        """Verify _extract_clip_frames is called with correct time range."""
        from PIL import Image as PILImage

        from autopilot.analyze.captions import caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        dummy_frames = [PILImage.new("RGB", (100, 100)) for _ in range(8)]

        with (
            patch(
                "autopilot.analyze.captions._extract_clip_frames", return_value=dummy_frames
            ) as mock_extract,
            patch.object(Path, "exists", return_value=True),
        ):
            caption_clip(
                "m1", Path("/test/video.mp4"), 5.0, 25.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

        mock_extract.assert_called_once()
        args = mock_extract.call_args
        assert args[0][1] == 5.0  # start_time
        assert args[0][2] == 25.0  # end_time


# ---------------------------------------------------------------------------
# TestIdempotency — caption_clip() skips when caption exists
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Tests for caption_clip() idempotency behavior."""

    def test_existing_caption_skips(self, catalog_db):
        """When caption already exists, return it without touching scheduler."""
        from autopilot.analyze.captions import caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )
        catalog_db.upsert_caption("m1", 0.0, 30.0, "Existing caption", "old-model")

        scheduler = _make_mock_scheduler()
        config = _make_mock_config()

        result = caption_clip(
            "m1", Path("/test/video.mp4"), 0.0, 30.0,
            db=catalog_db, scheduler=scheduler, config=config,
        )

        assert result == "Existing caption"
        scheduler.model.assert_not_called()

    def test_no_existing_caption_proceeds(self, catalog_db):
        """When no caption exists, scheduler.model() is called."""
        from PIL import Image as PILImage

        from autopilot.analyze.captions import caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        dummy_frames = [PILImage.new("RGB", (100, 100)) for _ in range(8)]

        with (
            patch("autopilot.analyze.captions._extract_clip_frames", return_value=dummy_frames),
            patch.object(Path, "exists", return_value=True),
        ):
            caption_clip(
                "m1", Path("/test/video.mp4"), 0.0, 30.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

        scheduler.model.assert_called_once()


# ---------------------------------------------------------------------------
# TestErrorHandling — caption_clip() error cases
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for caption_clip() error handling."""

    def test_video_not_found_raises(self, catalog_db):
        """Nonexistent video_path raises CaptionError matching 'not found'."""
        from autopilot.analyze.captions import CaptionError, caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/nonexistent/video.mp4", fps=30.0, duration_seconds=60.0
        )
        scheduler = _make_mock_scheduler()
        config = _make_mock_config()

        with pytest.raises(CaptionError, match="not found"):
            caption_clip(
                "m1", Path("/nonexistent/video.mp4"), 0.0, 30.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

    def test_no_frames_extracted_raises(self, catalog_db):
        """When _extract_clip_frames returns empty list, raises CaptionError."""
        from autopilot.analyze.captions import CaptionError, caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )
        scheduler = _make_mock_scheduler()
        config = _make_mock_config()

        with (
            patch("autopilot.analyze.captions._extract_clip_frames", return_value=[]),
            patch.object(Path, "exists", return_value=True),
        ):
            with pytest.raises(CaptionError, match="No frames"):
                caption_clip(
                    "m1", Path("/test/video.mp4"), 0.0, 30.0,
                    db=catalog_db, scheduler=scheduler, config=config,
                )

    def test_invalid_time_range_raises(self, catalog_db):
        """start_time > end_time raises CaptionError matching 'Invalid time range'."""
        from autopilot.analyze.captions import CaptionError, caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )
        scheduler = _make_mock_scheduler()
        config = _make_mock_config()

        with pytest.raises(CaptionError, match="Invalid time range"):
            caption_clip(
                "m1", Path("/test/video.mp4"), 30.0, 10.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

    def test_start_time_negative_raises(self, catalog_db):
        """start_time < 0 raises CaptionError."""
        from autopilot.analyze.captions import CaptionError, caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )
        scheduler = _make_mock_scheduler()
        config = _make_mock_config()

        with pytest.raises(CaptionError):
            caption_clip(
                "m1", Path("/test/video.mp4"), -5.0, 10.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )


# ---------------------------------------------------------------------------
# TestBatchCaption — batch_caption() sampling and iteration
# ---------------------------------------------------------------------------


class TestBatchCaption:
    """Tests for batch_caption() sampling and iteration."""

    def test_specific_media_ids(self, catalog_db):
        """sample_rate=1.0 captions all provided media_ids."""
        from autopilot.analyze.captions import batch_caption

        # Insert 3 media files
        for i in range(3):
            catalog_db.insert_media(
                id=f"m{i}", file_path=f"/test/video{i}.mp4",
                fps=30.0, duration_seconds=60.0,
            )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        with (
            patch("autopilot.analyze.captions._extract_clip_frames") as mock_extract,
            patch.object(Path, "exists", return_value=True),
        ):
            from PIL import Image as PILImage

            mock_extract.return_value = [PILImage.new("RGB", (100, 100))]
            batch_caption(
                ["m0", "m1", "m2"],
                db=catalog_db, scheduler=scheduler, config=config,
                sample_rate=1.0,
            )

        # All 3 should be captioned
        for i in range(3):
            captions = catalog_db.get_captions_for_media(f"m{i}")
            assert len(captions) >= 1

    def test_skips_already_captioned(self, catalog_db):
        """media_ids that already have captions in DB are skipped."""
        from autopilot.analyze.captions import batch_caption

        catalog_db.insert_media(
            id="m0", file_path="/test/video0.mp4", fps=30.0, duration_seconds=60.0
        )
        catalog_db.insert_media(
            id="m1", file_path="/test/video1.mp4", fps=30.0, duration_seconds=60.0
        )
        # Pre-populate caption for m0
        catalog_db.upsert_caption("m0", 0.0, 60.0, "Existing caption", "old-model")

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        with (
            patch("autopilot.analyze.captions._extract_clip_frames") as mock_extract,
            patch.object(Path, "exists", return_value=True),
        ):
            from PIL import Image as PILImage

            mock_extract.return_value = [PILImage.new("RGB", (100, 100))]
            batch_caption(
                ["m0", "m1"],
                db=catalog_db, scheduler=scheduler, config=config,
                sample_rate=1.0,
            )

        # m0 should still have original caption (idempotency)
        cap0 = catalog_db.get_caption("m0", 0.0, 60.0)
        assert cap0 is not None
        assert cap0["caption"] == "Existing caption"

    def test_empty_media_ids_noop(self, catalog_db):
        """Empty list returns immediately without error."""
        from autopilot.analyze.captions import batch_caption

        scheduler = _make_mock_scheduler()
        config = _make_mock_config()

        # Should not raise
        batch_caption(
            [], db=catalog_db, scheduler=scheduler, config=config, sample_rate=1.0
        )

    def test_samples_subset(self, catalog_db):
        """With sample_rate=0.5, approximately half of clips are captioned."""
        import random

        from autopilot.analyze.captions import batch_caption

        # Insert 10 media files
        for i in range(10):
            catalog_db.insert_media(
                id=f"m{i}", file_path=f"/test/video{i}.mp4",
                fps=30.0, duration_seconds=60.0,
            )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        random.seed(42)  # Deterministic for test

        with (
            patch("autopilot.analyze.captions._extract_clip_frames") as mock_extract,
            patch.object(Path, "exists", return_value=True),
        ):
            from PIL import Image as PILImage

            mock_extract.return_value = [PILImage.new("RGB", (100, 100))]
            batch_caption(
                [f"m{i}" for i in range(10)],
                db=catalog_db, scheduler=scheduler, config=config,
                sample_rate=0.5,
            )

        # Count captioned media
        captioned_count = sum(
            1 for i in range(10) if catalog_db.get_captions_for_media(f"m{i}")
        )
        # With sample_rate=0.5, expect roughly 5 (allow tolerance)
        assert 2 <= captioned_count <= 8


# ---------------------------------------------------------------------------
# TestBackendSelection — vLLM / transformers backend selection
# ---------------------------------------------------------------------------


class TestBackendSelection:
    """Tests for model backend selection and registration."""

    def test_model_registration_vram(self):
        """ModelSpec registered with ~14GB vram_bytes for FP16."""
        from autopilot.analyze.captions import _make_caption_model_spec

        config = _make_mock_config()
        spec = _make_caption_model_spec(config)
        # 14 GB = 14 * 1024^3
        assert spec.vram_bytes == 14 * 1024**3

    def test_register_caption_model(self):
        """register_caption_model registers spec with scheduler."""
        from autopilot.analyze.captions import register_caption_model

        scheduler = MagicMock()
        config = _make_mock_config()

        register_caption_model(scheduler, config)

        scheduler.register.assert_called_once()
        call_args = scheduler.register.call_args
        assert call_args[0][0] == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_transformers_fallback(self):
        """When vllm import fails, transformers inference path is taken."""
        from autopilot.analyze.captions import _make_caption_model_spec

        config = _make_mock_config()

        # Ensure vllm is not available
        with patch.dict(sys.modules, {"vllm": None}):
            spec = _make_caption_model_spec(config)

        # spec should be created successfully (transformers fallback)
        assert spec is not None
        assert callable(spec.load_fn)


# ---------------------------------------------------------------------------
# TestLogging — structured logging output
# ---------------------------------------------------------------------------


class TestLogging:
    """Tests for caption_clip() and batch_caption() logging."""

    def test_log_contains_media_id(self, catalog_db, caplog):
        """INFO log includes media_id on caption start."""
        from PIL import Image as PILImage

        from autopilot.analyze.captions import caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        dummy_frames = [PILImage.new("RGB", (100, 100)) for _ in range(8)]

        with (
            caplog.at_level(logging.INFO, logger="autopilot.analyze.captions"),
            patch("autopilot.analyze.captions._extract_clip_frames", return_value=dummy_frames),
            patch.object(Path, "exists", return_value=True),
        ):
            caption_clip(
                "m1", Path("/test/video.mp4"), 0.0, 30.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

        assert any("m1" in msg for msg in caplog.messages)

    def test_log_caption_complete(self, catalog_db, caplog):
        """INFO log on completion with media_id and caption length."""
        from PIL import Image as PILImage

        from autopilot.analyze.captions import caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        dummy_frames = [PILImage.new("RGB", (100, 100)) for _ in range(8)]

        with (
            caplog.at_level(logging.INFO, logger="autopilot.analyze.captions"),
            patch("autopilot.analyze.captions._extract_clip_frames", return_value=dummy_frames),
            patch.object(Path, "exists", return_value=True),
        ):
            caption_clip(
                "m1", Path("/test/video.mp4"), 0.0, 30.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

        # Should log completion
        assert any(
            "m1" in msg and ("complete" in msg.lower() or "caption" in msg.lower())
            for msg in caplog.messages
        )

    def test_log_skipping_on_idempotent(self, catalog_db, caplog):
        """INFO log with 'already exists' when caption exists."""
        from autopilot.analyze.captions import caption_clip

        catalog_db.insert_media(
            id="m1", file_path="/test/video.mp4", fps=30.0, duration_seconds=60.0
        )
        catalog_db.upsert_caption("m1", 0.0, 30.0, "Existing", "old-model")

        scheduler = _make_mock_scheduler()
        config = _make_mock_config()

        with caplog.at_level(logging.INFO, logger="autopilot.analyze.captions"):
            caption_clip(
                "m1", Path("/test/video.mp4"), 0.0, 30.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

        assert any("already exists" in msg for msg in caplog.messages)

    def test_log_batch_progress(self, catalog_db, caplog):
        """batch_caption logs progress count."""
        from autopilot.analyze.captions import batch_caption

        catalog_db.insert_media(
            id="m0", file_path="/test/v0.mp4", fps=30.0, duration_seconds=60.0
        )

        model, processor = _make_mock_model_and_processor()
        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        with (
            caplog.at_level(logging.INFO, logger="autopilot.analyze.captions"),
            patch("autopilot.analyze.captions._extract_clip_frames") as mock_extract,
            patch.object(Path, "exists", return_value=True),
        ):
            from PIL import Image as PILImage

            mock_extract.return_value = [PILImage.new("RGB", (100, 100))]
            batch_caption(
                ["m0"], db=catalog_db, scheduler=scheduler, config=config,
                sample_rate=1.0,
            )

        assert any("1/1" in msg and "captioned" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# TestModelConfig — caption_model field in config
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for ModelConfig.caption_model field."""

    def test_model_config_has_caption_model(self):
        """ModelConfig().caption_model has correct default."""
        from autopilot.config import ModelConfig

        assert ModelConfig().caption_model == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_config_yaml_override(self, tmp_path):
        """Loading config YAML with models.caption_model overrides default."""
        from autopilot.config import load_config

        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            f"input_dir: {tmp_path}\n"
            f"output_dir: {tmp_path}\n"
            "models:\n"
            "  caption_model: custom/model-name\n"
        )
        config = load_config(config_file)
        assert config.models.caption_model == "custom/model-name"


# ---------------------------------------------------------------------------
# TestIntegration — end-to-end integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration test with mocked model and real DB."""

    def test_caption_clip_end_to_end(self, catalog_db):
        """Mock cv2 + mock model, run caption_clip with real DB."""
        from PIL import Image as PILImage

        from autopilot.analyze.captions import caption_clip

        # Insert media with all fields needed
        catalog_db.insert_media(
            id="vid1", file_path="/test/vacation.mp4",
            fps=24.0, duration_seconds=120.0,
        )

        # Mock model to return a specific caption
        model = MagicMock()
        processor = MagicMock()
        expected_caption = "A family enjoying a sunset on the beach"
        processor.batch_decode.return_value = [expected_caption]

        scheduler = _make_mock_scheduler(model_obj={"model": model, "processor": processor})
        config = _make_mock_config()

        dummy_frames = [PILImage.new("RGB", (640, 480)) for _ in range(8)]

        with (
            patch("autopilot.analyze.captions._extract_clip_frames", return_value=dummy_frames),
            patch.object(Path, "exists", return_value=True),
        ):
            result = caption_clip(
                "vid1", Path("/test/vacation.mp4"), 10.0, 40.0,
                db=catalog_db, scheduler=scheduler, config=config,
            )

        # Verify return value
        assert result == expected_caption

        # Verify stored in DB and retrievable
        stored = catalog_db.get_caption("vid1", 10.0, 40.0)
        assert stored is not None
        assert stored["caption"] == expected_caption
        assert stored["model_name"] == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert stored["media_id"] == "vid1"
        assert stored["start_time"] == 10.0
        assert stored["end_time"] == 40.0

        # Verify idempotency on second call (should skip model)
        scheduler2 = _make_mock_scheduler()
        result2 = caption_clip(
            "vid1", Path("/test/vacation.mp4"), 10.0, 40.0,
            db=catalog_db, scheduler=scheduler2, config=config,
        )
        assert result2 == expected_caption
        scheduler2.model.assert_not_called()
