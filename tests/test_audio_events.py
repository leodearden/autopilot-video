"""Tests for audio event classification (autopilot.analyze.audio_events)."""

from __future__ import annotations

import inspect
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest

# -- Mock helpers --------------------------------------------------------------


def _make_mock_panns(
    labels: list[str],
) -> tuple[MagicMock, MagicMock]:
    """Create mock panns_inference module and config sub-module.

    Args:
        labels: List of AudioSet class labels.

    Returns:
        Tuple of (mock_panns_module, mock_panns_config_module).
        mock_panns_module has AudioTagging whose instances have .inference
        returning (np.random probabilities shape (1, len(labels)), mock_embedding).
        mock_panns_config_module has .labels attribute.
    """
    mock_panns = MagicMock()
    mock_config = MagicMock()
    mock_config.labels = labels

    # AudioTagging() returns a model object with .inference method
    probs = np.random.default_rng(42).random((1, len(labels)), dtype=np.float32)
    mock_embedding = np.zeros((1, 2048), dtype=np.float32)
    mock_model = MagicMock()
    mock_model.inference.return_value = (probs, mock_embedding)
    mock_panns.AudioTagging.return_value = mock_model

    # Wire config as sub-attribute
    mock_panns.config = mock_config

    return mock_panns, mock_config


def _make_mock_librosa(
    duration_seconds: float,
    sr: int = 32000,
) -> MagicMock:
    """Create mock librosa module returning audio of given duration.

    Args:
        duration_seconds: Length of the returned waveform in seconds.
        sr: Sample rate.

    Returns:
        MagicMock with .load returning (np.zeros waveform, sr).
    """
    mock_librosa = MagicMock()
    num_samples = int(duration_seconds * sr)
    waveform = np.zeros(num_samples, dtype=np.float32)
    mock_librosa.load.return_value = (waveform, sr)
    return mock_librosa


def _make_full_pipeline_mocks(
    catalog_db,
    media_id: str,
    duration_seconds: float,
    *,
    labels: list[str] | None = None,
    sr: int = 32000,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
    """Set up all mocks for a complete audio event classification run.

    Inserts a media file into catalog_db, configures mock panns_inference,
    mock librosa, and mock scheduler.

    Args:
        catalog_db: CatalogDB fixture instance.
        media_id: Media ID to use.
        duration_seconds: Audio duration in seconds.
        labels: AudioSet labels (defaults to 527 synthetic labels).
        sr: Sample rate.

    Returns:
        Tuple of (mock_panns, mock_panns_config, mock_librosa, scheduler).
    """
    if labels is None:
        labels = [f"class_{i}" for i in range(527)]

    catalog_db.insert_media(media_id, f"/tmp/{media_id}.wav")

    mock_panns, mock_panns_config = _make_mock_panns(labels)
    mock_librosa = _make_mock_librosa(duration_seconds, sr)

    # Configure scheduler with context manager protocol
    mock_model = mock_panns.AudioTagging.return_value
    scheduler = MagicMock()
    scheduler.device = 0
    scheduler.model.return_value.__enter__ = MagicMock(return_value=mock_model)
    scheduler.model.return_value.__exit__ = MagicMock(return_value=False)

    return mock_panns, mock_panns_config, mock_librosa, scheduler


# -- Test classes --------------------------------------------------------------


class TestPublicAPI:
    """Verify public API surface and type signatures."""

    def test_exports_importable(self):
        """AudioEventError and classify_audio_events are importable."""
        from autopilot.analyze.audio_events import (
            AudioEventError,
            classify_audio_events,
        )

        assert AudioEventError is not None
        assert classify_audio_events is not None

    def test_audio_event_error_is_exception(self):
        """AudioEventError is a subclass of Exception with message."""
        from autopilot.analyze.audio_events import AudioEventError

        assert issubclass(AudioEventError, Exception)
        err = AudioEventError("test message")
        assert str(err) == "test message"

    def test_classify_audio_events_signature(self):
        """classify_audio_events has correct parameter signature."""
        from autopilot.analyze.audio_events import classify_audio_events

        sig = inspect.signature(classify_audio_events)
        params = list(sig.parameters.keys())

        # Positional parameters
        assert "media_id" in params
        assert "audio_path" in params
        assert "db" in params
        assert "scheduler" in params

        # Keyword-only parameter with default
        top_k_param = sig.parameters["top_k"]
        assert top_k_param.kind == inspect.Parameter.KEYWORD_ONLY
        assert top_k_param.default == 5

        # Return annotation (string 'None' due to __future__ annotations)
        assert sig.return_annotation in (None, "None")


class TestWindowAudio:
    """Tests for _window_audio() private helper."""

    def test_exact_multiple(self):
        """3s at 32kHz -> 3 windows of 32000 samples each."""
        from autopilot.analyze.audio_events import _window_audio

        audio = np.zeros(3 * 32000, dtype=np.float32)
        windows = _window_audio(audio, 32000)
        assert len(windows) == 3
        for w in windows:
            assert w.shape == (32000,)

    def test_short_audio_padded(self):
        """0.5s -> 1 window zero-padded to 32000."""
        from autopilot.analyze.audio_events import _window_audio

        audio = np.ones(16000, dtype=np.float32)
        windows = _window_audio(audio, 32000)
        assert len(windows) == 1
        assert windows[0].shape == (32000,)
        # First half is ones, second half is zeros (padding)
        assert np.all(windows[0][:16000] == 1.0)
        assert np.all(windows[0][16000:] == 0.0)

    def test_remainder_padded(self):
        """2.5s -> 3 windows, last zero-padded."""
        from autopilot.analyze.audio_events import _window_audio

        audio = np.ones(int(2.5 * 32000), dtype=np.float32)
        windows = _window_audio(audio, 32000)
        assert len(windows) == 3
        # First two windows are full
        assert np.all(windows[0] == 1.0)
        assert np.all(windows[1] == 1.0)
        # Third window: first half ones, second half zeros
        assert np.all(windows[2][:16000] == 1.0)
        assert np.all(windows[2][16000:] == 0.0)

    def test_empty_audio(self):
        """0 samples -> empty list."""
        from autopilot.analyze.audio_events import _window_audio

        audio = np.array([], dtype=np.float32)
        windows = _window_audio(audio, 32000)
        assert windows == []

    def test_exactly_one_second(self):
        """32000 samples -> 1 window."""
        from autopilot.analyze.audio_events import _window_audio

        audio = np.ones(32000, dtype=np.float32)
        windows = _window_audio(audio, 32000)
        assert len(windows) == 1
        assert np.all(windows[0] == 1.0)

    def test_output_dtype_preserved(self):
        """float32 in -> float32 out."""
        from autopilot.analyze.audio_events import _window_audio

        audio = np.zeros(32000, dtype=np.float32)
        windows = _window_audio(audio, 32000)
        assert windows[0].dtype == np.float32


class TestExtractTopK:
    """Tests for _extract_top_k() private helper."""

    def test_basic_top5(self):
        """527-element array -> correct top-5 sorted descending."""
        from autopilot.analyze.audio_events import _extract_top_k

        labels = [f"class_{i}" for i in range(527)]
        probs = np.zeros(527, dtype=np.float32)
        probs[10] = 0.9
        probs[20] = 0.7
        probs[30] = 0.5
        probs[40] = 0.3
        probs[50] = 0.1
        result = _extract_top_k(probs, labels)
        assert len(result) == 5
        assert result[0]["class"] == "class_10"
        assert result[0]["probability"] == pytest.approx(0.9)
        assert result[1]["class"] == "class_20"
        # Verify descending order
        probabilities = [r["probability"] for r in result]
        assert probabilities == sorted(probabilities, reverse=True)

    def test_k_parameter(self):
        """k=3 -> 3 items."""
        from autopilot.analyze.audio_events import _extract_top_k

        labels = [f"class_{i}" for i in range(527)]
        probs = np.random.default_rng(0).random(527).astype(np.float32)
        result = _extract_top_k(probs, labels, k=3)
        assert len(result) == 3

    def test_output_keys(self):
        """Each dict has exactly {'class', 'probability'}."""
        from autopilot.analyze.audio_events import _extract_top_k

        labels = ["a", "b", "c"]
        probs = np.array([0.1, 0.5, 0.3], dtype=np.float32)
        result = _extract_top_k(probs, labels, k=3)
        for item in result:
            assert set(item.keys()) == {"class", "probability"}

    def test_numpy_float_coercion(self):
        """np.float32 probabilities -> isinstance(p, float) for all."""
        from autopilot.analyze.audio_events import _extract_top_k

        labels = [f"class_{i}" for i in range(10)]
        probs = np.random.default_rng(1).random(10).astype(np.float32)
        result = _extract_top_k(probs, labels, k=5)
        for item in result:
            assert isinstance(item["probability"], float)
            assert isinstance(item["class"], str)

    def test_json_serializable(self):
        """json.dumps succeeds on output."""
        from autopilot.analyze.audio_events import _extract_top_k

        labels = [f"class_{i}" for i in range(527)]
        probs = np.random.default_rng(2).random(527).astype(np.float32)
        result = _extract_top_k(probs, labels)
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_all_zeros(self):
        """Returns k items with prob 0.0."""
        from autopilot.analyze.audio_events import _extract_top_k

        labels = [f"class_{i}" for i in range(527)]
        probs = np.zeros(527, dtype=np.float32)
        result = _extract_top_k(probs, labels)
        assert len(result) == 5
        for item in result:
            assert item["probability"] == 0.0

    def test_k_larger_than_labels(self):
        """k=600 with 527 labels -> 527 items."""
        from autopilot.analyze.audio_events import _extract_top_k

        labels = [f"class_{i}" for i in range(527)]
        probs = np.random.default_rng(3).random(527).astype(np.float32)
        result = _extract_top_k(probs, labels, k=600)
        assert len(result) == 527


class TestIdempotency:
    """Tests for audio event idempotency check."""

    def test_scheduler_direct_call_is_vacuous(self, catalog_db):
        """Prove scheduler() is never called by production code.

        Temporary test: scheduler.call_count == 0 holds trivially in the
        skip path, demonstrating that scheduler.assert_not_called() adds
        no value over scheduler.model.assert_not_called().
        """
        from autopilot.analyze.audio_events import classify_audio_events

        catalog_db.insert_media("vid1", "/tmp/vid1.wav")
        with catalog_db:
            catalog_db.batch_insert_audio_events(
                [
                    ("vid1", 0.0, json.dumps([{"class": "Speech", "probability": 0.9}])),
                ]
            )

        scheduler = MagicMock()
        classify_audio_events(
            "vid1",
            Path("/tmp/vid1.wav"),
            catalog_db,
            scheduler,
        )

        # Production code calls scheduler.model(), never scheduler() directly.
        # Therefore scheduler.call_count is always 0 — the assertion is vacuous.
        assert scheduler.call_count == 0, (
            f"Expected scheduler() never called directly, got {scheduler.call_count} calls"
        )
        # The meaningful assertion is on the .model attribute:
        scheduler.model.assert_not_called()

    def test_skip_when_events_exist(self, catalog_db):
        """Skip classification when audio events already exist."""
        from autopilot.analyze.audio_events import classify_audio_events

        catalog_db.insert_media("vid1", "/tmp/vid1.wav")
        with catalog_db:
            catalog_db.batch_insert_audio_events(
                [
                    ("vid1", 0.0, json.dumps([{"class": "Speech", "probability": 0.9}])),
                ]
            )

        scheduler = MagicMock()
        classify_audio_events(
            "vid1",
            Path("/tmp/vid1.wav"),
            catalog_db,
            scheduler,
        )

        # Scheduler should NOT be called: classification skipped because events already exist
        scheduler.model.assert_not_called()

        # DB postcondition: the single pre-existing event is still present,
        # no extra rows written
        events = catalog_db.get_audio_events_for_media("vid1")
        assert len(events) == 1

    def test_skip_when_events_exist_at_nonzero_timestamp(self, catalog_db):
        """Skip classification when audio events exist only at non-zero timestamps.

        Regression test: the old idempotency check used
        get_audio_events_for_range(media_id, 0.0, 0.0) which only found events
        at exactly t=0.0, missing events that start later.
        """
        from autopilot.analyze.audio_events import classify_audio_events

        catalog_db.insert_media("vid1", "/tmp/vid1.wav")
        with catalog_db:
            catalog_db.batch_insert_audio_events(
                [
                    ("vid1", 1.5, json.dumps([{"class": "Speech", "probability": 0.9}])),
                    ("vid1", 3.0, json.dumps([{"class": "Music", "probability": 0.8}])),
                ]
            )

        scheduler = MagicMock()
        classify_audio_events(
            "vid1",
            Path("/tmp/vid1.wav"),
            catalog_db,
            scheduler,
        )

        # Scheduler should NOT be called: classification skipped because events already exist
        scheduler.model.assert_not_called()
        scheduler.assert_not_called()

        # DB postcondition: both pre-existing events remain, no extra rows written
        events = catalog_db.get_audio_events_for_media("vid1")
        assert len(events) == 2

    def test_skip_when_events_exist_beyond_range(self, catalog_db):
        """Skip classification when audio events exist beyond the old proxy range.

        Regression test: the old proxy pattern used
        get_audio_events_for_range(media_id, 0.0, 10.0) to check for existing
        events, which would miss events at t>=10.0. The unbounded
        get_audio_events_for_media (and has_audio_events) correctly detects
        events at any timestamp.
        """
        from autopilot.analyze.audio_events import classify_audio_events

        catalog_db.insert_media("vid1", "/tmp/vid1.wav")
        with catalog_db:
            catalog_db.batch_insert_audio_events(
                [
                    ("vid1", 15.0, json.dumps([{"class": "Speech", "probability": 0.9}])),
                ]
            )

        scheduler = MagicMock()
        classify_audio_events(
            "vid1",
            Path("/tmp/vid1.wav"),
            catalog_db,
            scheduler,
        )

        # Scheduler should NOT be called: classification skipped because events already exist
        scheduler.model.assert_not_called()
        scheduler.assert_not_called()

        # DB postcondition: the single pre-existing event at t=15.0 is still present
        events = catalog_db.get_audio_events_for_media("vid1")
        assert len(events) == 1
        assert float(events[0]["timestamp_seconds"]) == 15.0

    def test_processes_when_no_events(self, catalog_db):
        """Proceed with classification when no events exist."""
        from autopilot.analyze.audio_events import classify_audio_events

        mock_panns, mock_config, mock_librosa, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", 3.0
        )

        with patch.dict(
            sys.modules,
            {
                "panns_inference": mock_panns,
                "panns_inference.config": mock_config,
                "librosa": mock_librosa,
            },
        ):
            with patch.object(Path, "exists", return_value=True):
                classify_audio_events(
                    "vid1",
                    Path("/tmp/vid1.wav"),
                    catalog_db,
                    scheduler,
                )

        # Scheduler SHOULD be called for model loading
        scheduler.model.assert_called_once_with("panns_cnn14")


class TestInputValidation:
    """Tests for audio path validation."""

    def test_file_not_found_raises(self, catalog_db):
        """AudioEventError raised for non-existent audio file."""
        from autopilot.analyze.audio_events import (
            AudioEventError,
            classify_audio_events,
        )

        catalog_db.insert_media("vid1", "/nonexistent/audio.wav")
        scheduler = MagicMock()

        with pytest.raises(AudioEventError, match="not found"):
            classify_audio_events(
                "vid1",
                Path("/nonexistent/audio.wav"),
                catalog_db,
                scheduler,
            )

    def test_raises_before_lazy_imports(self, catalog_db):
        """Path validation happens before scheduler is touched."""
        from autopilot.analyze.audio_events import (
            AudioEventError,
            classify_audio_events,
        )

        catalog_db.insert_media("vid1", "/nonexistent/audio.wav")
        scheduler = MagicMock()

        with patch.object(Path, "exists", return_value=False):
            with pytest.raises(AudioEventError):
                classify_audio_events(
                    "vid1",
                    Path("/tmp/audio.wav"),
                    catalog_db,
                    scheduler,
                )

        # Scheduler should NOT be called (failed before imports)
        scheduler.model.assert_not_called()


class TestStatusUpdate:
    """Tests for media status update."""

    def test_sets_status_analyzing(self, catalog_db):
        """Media status updated to 'analyzing' during classification."""
        from autopilot.analyze.audio_events import classify_audio_events

        mock_panns, mock_config, mock_librosa, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", 3.0
        )

        with patch.dict(
            sys.modules,
            {
                "panns_inference": mock_panns,
                "panns_inference.config": mock_config,
                "librosa": mock_librosa,
            },
        ):
            with patch.object(Path, "exists", return_value=True):
                classify_audio_events(
                    "vid1",
                    Path("/tmp/vid1.wav"),
                    catalog_db,
                    scheduler,
                )

        media = catalog_db.get_media("vid1")
        assert media is not None
        assert media["status"] == "analyzing"


class TestClassification:
    """Tests for core classification pipeline."""

    def _run_classification(
        self,
        catalog_db,
        duration_seconds,
        *,
        top_k=5,
        labels=None,
    ):
        """Helper to run classify_audio_events with full mocks."""
        from autopilot.analyze.audio_events import classify_audio_events

        mock_panns, mock_config, mock_librosa, scheduler = _make_full_pipeline_mocks(
            catalog_db,
            "vid1",
            duration_seconds,
            labels=labels,
        )

        with patch.dict(
            sys.modules,
            {
                "panns_inference": mock_panns,
                "panns_inference.config": mock_config,
                "librosa": mock_librosa,
            },
        ):
            with patch.object(Path, "exists", return_value=True):
                classify_audio_events(
                    "vid1",
                    Path("/tmp/vid1.wav"),
                    catalog_db,
                    scheduler,
                    top_k=top_k,
                )

        return mock_panns, mock_config, mock_librosa, scheduler

    def test_loads_audio_at_32khz(self, catalog_db):
        """librosa.load called with str(audio_path), sr=32000, mono=True."""
        _, _, mock_librosa, _ = self._run_classification(catalog_db, 3.0)

        mock_librosa.load.assert_called_once_with(
            str(Path("/tmp/vid1.wav")),
            sr=32000,
            mono=True,
        )

    def test_loads_model_via_scheduler(self, catalog_db):
        """scheduler.model called with 'panns_cnn14'."""
        _, _, _, scheduler = self._run_classification(catalog_db, 3.0)

        scheduler.model.assert_called_once_with("panns_cnn14")

    def test_correct_window_count(self, catalog_db):
        """3s audio -> model.inference called 3 times."""
        mock_panns, _, _, _ = self._run_classification(catalog_db, 3.0)

        mock_model = mock_panns.AudioTagging.return_value
        assert mock_model.inference.call_count == 3

    def test_stores_top_k_per_second(self, catalog_db):
        """events_json is list of top_k dicts per window."""
        self._run_classification(catalog_db, 3.0)

        events = catalog_db.get_audio_events_for_media("vid1")
        assert len(events) == 3
        for ev in events:
            parsed = json.loads(str(ev["events_json"]))
            assert len(parsed) == 5

    def test_custom_top_k(self, catalog_db):
        """top_k=3 -> 3 events per window in DB."""
        self._run_classification(catalog_db, 2.0, top_k=3)

        events = catalog_db.get_audio_events_for_media("vid1")
        for ev in events:
            parsed = json.loads(str(ev["events_json"]))
            assert len(parsed) == 3


class TestDBStorage:
    """Tests for audio event DB storage."""

    def _run_and_get_events(self, catalog_db, duration_seconds, **kwargs):
        """Run classification and return stored events."""
        from autopilot.analyze.audio_events import classify_audio_events

        mock_panns, mock_config, mock_librosa, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", duration_seconds
        )

        with patch.dict(
            sys.modules,
            {
                "panns_inference": mock_panns,
                "panns_inference.config": mock_config,
                "librosa": mock_librosa,
            },
        ):
            with patch.object(Path, "exists", return_value=True):
                classify_audio_events(
                    "vid1",
                    Path("/tmp/vid1.wav"),
                    catalog_db,
                    scheduler,
                    **kwargs,
                )

        return catalog_db.get_audio_events_for_media("vid1")

    def test_correct_timestamps(self, catalog_db):
        """5s audio -> timestamps 0.0, 1.0, 2.0, 3.0, 4.0 in DB."""
        events = self._run_and_get_events(catalog_db, 5.0)
        timestamps = sorted(float(ev["timestamp_seconds"]) for ev in events)
        assert timestamps == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_events_json_schema(self, catalog_db):
        """Each row's events_json deserializes to list of dicts with correct keys."""
        events = self._run_and_get_events(catalog_db, 3.0)
        assert len(events) == 3
        for ev in events:
            parsed = json.loads(str(ev["events_json"]))
            assert isinstance(parsed, list)
            for item in parsed:
                assert isinstance(item["class"], str)
                assert isinstance(item["probability"], float)

    def test_retrievable_by_range(self, catalog_db):
        """get_audio_events_for_range returns stored rows."""
        self._run_and_get_events(catalog_db, 5.0)
        # Query subset
        subset = catalog_db.get_audio_events_for_range("vid1", 1.0, 3.0)
        timestamps = sorted(float(ev["timestamp_seconds"]) for ev in subset)
        assert timestamps == [1.0, 2.0, 3.0]

    def test_no_numpy_in_json(self, catalog_db):
        """All probability values are plain Python float, not numpy."""
        events = self._run_and_get_events(catalog_db, 2.0)
        for ev in events:
            parsed = json.loads(str(ev["events_json"]))
            for item in parsed:
                assert type(item["probability"]) is float


class TestErrorHandling:
    """Tests for error handling in classification pipeline."""

    def test_runtime_error_wrapped(self, catalog_db):
        """RuntimeError in librosa.load -> AudioEventError."""
        from autopilot.analyze.audio_events import (
            AudioEventError,
            classify_audio_events,
        )

        catalog_db.insert_media("vid1", "/tmp/vid1.wav")

        mock_librosa = MagicMock()
        mock_librosa.load.side_effect = RuntimeError("decode failed")

        with patch.dict(sys.modules, {"librosa": mock_librosa}):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(AudioEventError, match="decode failed"):
                    classify_audio_events(
                        "vid1",
                        Path("/tmp/vid1.wav"),
                        catalog_db,
                        MagicMock(),
                    )

    def test_audio_event_error_passthrough(self, catalog_db):
        """AudioEventError raised inside pipeline propagates unchanged."""
        from autopilot.analyze.audio_events import (
            AudioEventError,
            classify_audio_events,
        )

        catalog_db.insert_media("vid1", "/tmp/vid1.wav")

        mock_librosa = MagicMock()
        mock_librosa.load.side_effect = AudioEventError("custom error")

        with patch.dict(sys.modules, {"librosa": mock_librosa}):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(AudioEventError, match="^custom error$"):
                    classify_audio_events(
                        "vid1",
                        Path("/tmp/vid1.wav"),
                        catalog_db,
                        MagicMock(),
                    )

    def test_error_contains_media_id(self, catalog_db):
        """Wrapped error message includes media_id string."""
        from autopilot.analyze.audio_events import (
            AudioEventError,
            classify_audio_events,
        )

        catalog_db.insert_media("vid1", "/tmp/vid1.wav")

        mock_librosa = MagicMock()
        mock_librosa.load.side_effect = RuntimeError("oops")

        with patch.dict(sys.modules, {"librosa": mock_librosa}):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(AudioEventError, match="vid1"):
                    classify_audio_events(
                        "vid1",
                        Path("/tmp/vid1.wav"),
                        catalog_db,
                        MagicMock(),
                    )

    def test_no_partial_writes(self, catalog_db):
        """model.inference raises mid-way -> 0 audio_events rows in DB."""
        from autopilot.analyze.audio_events import (
            AudioEventError,
            classify_audio_events,
        )

        mock_panns, mock_config, mock_librosa, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", 3.0
        )
        # Make inference fail on the 2nd call
        mock_model = mock_panns.AudioTagging.return_value
        mock_model.inference.side_effect = [
            mock_model.inference.return_value,
            RuntimeError("CUDA OOM"),
        ]

        with patch.dict(
            sys.modules,
            {
                "panns_inference": mock_panns,
                "panns_inference.config": mock_config,
                "librosa": mock_librosa,
            },
        ):
            with patch.object(Path, "exists", return_value=True):
                with pytest.raises(AudioEventError):
                    classify_audio_events(
                        "vid1",
                        Path("/tmp/vid1.wav"),
                        catalog_db,
                        scheduler,
                    )

        # No partial data should be stored
        events = catalog_db.get_audio_events_for_media("vid1")
        assert len(events) == 0


class TestLogging:
    """Tests for structured logging output."""

    def test_start_log(self, catalog_db, caplog):
        """INFO log containing media_id and 'Starting'."""
        from autopilot.analyze.audio_events import classify_audio_events

        mock_panns, mock_config, mock_librosa, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", 3.0
        )

        with patch.dict(
            sys.modules,
            {
                "panns_inference": mock_panns,
                "panns_inference.config": mock_config,
                "librosa": mock_librosa,
            },
        ):
            with patch.object(Path, "exists", return_value=True):
                with caplog.at_level(logging.INFO):
                    classify_audio_events(
                        "vid1",
                        Path("/tmp/vid1.wav"),
                        catalog_db,
                        scheduler,
                    )

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("vid1" in m and "Starting" in m for m in info_messages)

    def test_completion_log(self, catalog_db, caplog):
        """INFO log containing media_id and seconds count."""
        from autopilot.analyze.audio_events import classify_audio_events

        mock_panns, mock_config, mock_librosa, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", 3.0
        )

        with patch.dict(
            sys.modules,
            {
                "panns_inference": mock_panns,
                "panns_inference.config": mock_config,
                "librosa": mock_librosa,
            },
        ):
            with patch.object(Path, "exists", return_value=True):
                with caplog.at_level(logging.INFO):
                    classify_audio_events(
                        "vid1",
                        Path("/tmp/vid1.wav"),
                        catalog_db,
                        scheduler,
                    )

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("3 seconds" in m for m in info_messages)

    def test_skip_log(self, catalog_db, caplog):
        """INFO log containing 'skipping' when idempotency triggers."""
        from autopilot.analyze.audio_events import classify_audio_events

        catalog_db.insert_media("vid1", "/tmp/vid1.wav")
        with catalog_db:
            catalog_db.batch_insert_audio_events(
                [
                    ("vid1", 0.0, json.dumps([{"class": "Speech", "probability": 0.9}])),
                ]
            )

        with caplog.at_level(logging.INFO):
            classify_audio_events(
                "vid1",
                Path("/tmp/vid1.wav"),
                catalog_db,
                MagicMock(),
            )

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("skipping" in m.lower() for m in info_messages)


class TestIntegration:
    """End-to-end integration tests."""

    def _run_pipeline(
        self,
        catalog_db,
        duration_seconds,
        *,
        top_k=5,
    ):
        """Run full pipeline and return stored events."""
        from autopilot.analyze.audio_events import classify_audio_events

        mock_panns, mock_config, mock_librosa, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", duration_seconds
        )

        with patch.dict(
            sys.modules,
            {
                "panns_inference": mock_panns,
                "panns_inference.config": mock_config,
                "librosa": mock_librosa,
            },
        ):
            with patch.object(Path, "exists", return_value=True):
                classify_audio_events(
                    "vid1",
                    Path("/tmp/vid1.wav"),
                    catalog_db,
                    scheduler,
                    top_k=top_k,
                )

        return scheduler

    def test_full_pipeline(self, catalog_db):
        """5s audio -> 5 rows, 5 events each, timestamps 0.0-4.0, sorted desc."""
        self._run_pipeline(catalog_db, 5.0)

        events = catalog_db.get_audio_events_for_media("vid1")
        assert len(events) == 5

        timestamps = sorted(float(ev["timestamp_seconds"]) for ev in events)
        assert timestamps == [0.0, 1.0, 2.0, 3.0, 4.0]

        for ev in events:
            parsed = json.loads(str(ev["events_json"]))
            assert len(parsed) == 5
            for item in parsed:
                assert isinstance(item["class"], str)
                assert isinstance(item["probability"], float)
            # Verify sorted descending
            probs = [item["probability"] for item in parsed]
            assert probs == sorted(probs, reverse=True)

    def test_idempotent_second_call(self, catalog_db):
        """Call twice, scheduler.model called only once."""
        from autopilot.analyze.audio_events import classify_audio_events

        mock_panns, mock_config, mock_librosa, scheduler = _make_full_pipeline_mocks(
            catalog_db, "vid1", 3.0
        )

        with patch.dict(
            sys.modules,
            {
                "panns_inference": mock_panns,
                "panns_inference.config": mock_config,
                "librosa": mock_librosa,
            },
        ):
            with patch.object(Path, "exists", return_value=True):
                classify_audio_events(
                    "vid1",
                    Path("/tmp/vid1.wav"),
                    catalog_db,
                    scheduler,
                )
                classify_audio_events(
                    "vid1",
                    Path("/tmp/vid1.wav"),
                    catalog_db,
                    scheduler,
                )

        scheduler.model.assert_called_once()

    def test_short_audio(self, catalog_db):
        """0.5s -> 1 row at t=0.0 with zero-padded window."""
        self._run_pipeline(catalog_db, 0.5)

        events = catalog_db.get_audio_events_for_media("vid1")
        assert len(events) == 1
        assert float(events[0]["timestamp_seconds"]) == 0.0

    def test_custom_top_k_integration(self, catalog_db):
        """top_k=3 -> 3 events per window."""
        self._run_pipeline(catalog_db, 2.0, top_k=3)

        events = catalog_db.get_audio_events_for_media("vid1")
        assert len(events) == 2
        for ev in events:
            parsed = json.loads(str(ev["events_json"]))
            assert len(parsed) == 3

    def test_media_status_updated(self, catalog_db):
        """Status='analyzing' after run."""
        self._run_pipeline(catalog_db, 2.0)

        media = catalog_db.get_media("vid1")
        assert media is not None
        assert media["status"] == "analyzing"
