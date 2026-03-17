"""Tests for autopilot.config — typed configuration dataclasses and loader."""

import dataclasses
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Step 1: Public API — all names importable and correct types
# ---------------------------------------------------------------------------


def test_all_dataclasses_importable() -> None:
    """All config dataclasses can be imported from autopilot.config."""
    from autopilot.config import (
        AutopilotConfig,
        CameraConfig,
        CreatorConfig,
        LLMConfig,
        ModelConfig,
        OutputConfig,
        ProcessingConfig,
        YouTubeConfig,
    )

    for cls in (
        CreatorConfig,
        CameraConfig,
        OutputConfig,
        ModelConfig,
        LLMConfig,
        YouTubeConfig,
        ProcessingConfig,
        AutopilotConfig,
    ):
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} must be a dataclass"


def test_config_error_is_exception() -> None:
    """ConfigError is an Exception subclass."""
    from autopilot.config import ConfigError

    assert issubclass(ConfigError, Exception)


def test_load_config_importable() -> None:
    """load_config is a callable."""
    from autopilot.config import load_config

    assert callable(load_config)


# ---------------------------------------------------------------------------
# Step 3: load_config with valid fixture
# ---------------------------------------------------------------------------

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def test_load_config_valid() -> None:
    """load_config parses valid_config.yaml into correct dataclass tree."""
    from autopilot.config import (
        AutopilotConfig,
        CameraConfig,
        load_config,
    )

    cfg = load_config(FIXTURES / "valid_config.yaml")

    # Top-level type
    assert isinstance(cfg, AutopilotConfig)

    # Paths are Path objects
    assert isinstance(cfg.input_dir, pathlib.Path)
    assert isinstance(cfg.output_dir, pathlib.Path)
    assert cfg.input_dir == pathlib.Path("/tmp/test-footage")
    assert cfg.output_dir == pathlib.Path("/tmp/test-output")

    # Creator
    assert cfg.creator.name == "Test Creator"
    assert cfg.creator.channel_style == "cinematic, dramatic, fast-paced"
    assert cfg.creator.target_audience == "filmmakers, 18-30"
    assert cfg.creator.default_video_duration_minutes == "5-10"
    assert cfg.creator.narration_style == "heavy voiceover, storytelling"
    assert cfg.creator.music_preference == "electronic, synth, atmospheric"

    # Cameras — dict[str, CameraConfig]
    assert "gopro_hero12" in cfg.cameras
    cam = cfg.cameras["gopro_hero12"]
    assert isinstance(cam, CameraConfig)
    assert cam.source_resolution == (3840, 2160)
    assert isinstance(cam.source_resolution, tuple)
    assert cam.aspect_mode == "landscape"
    assert cam.has_gyro_data is False
    assert isinstance(cam.has_gyro_data, bool)
    assert cam.default_crop_target == "9:16"
    assert cam.crop_smoothing_tau == 0.8

    # Output
    assert cfg.output.primary_aspect == "9:16"
    assert cfg.output.resolution == (1080, 1920)
    assert cfg.output.codec == "h264"
    assert cfg.output.quality_crf == 23
    assert cfg.output.audio_bitrate == "192k"
    assert cfg.output.target_loudness_lufs == -14

    # Models
    assert cfg.models.whisper_size == "large-v3-turbo"
    assert cfg.models.yolo_variant == "yolo11l"
    assert cfg.models.yolo_sample_every_n_frames == 5
    assert cfg.models.clip_model == "openai/clip-vit-large"
    assert cfg.models.face_model == "buffalo_s"
    assert cfg.models.tts_engine == "elevenlabs"
    assert cfg.models.music_engine == "fetch_list_only"

    # LLM
    assert cfg.llm.provider == "openai"
    assert cfg.llm.planning_model == "gpt-4o"
    assert cfg.llm.utility_model == "gpt-4o-mini"

    # YouTube
    assert cfg.youtube.privacy_status == "private"
    assert cfg.youtube.default_category == "22"
    assert isinstance(cfg.youtube.credentials_path, pathlib.Path)
    assert cfg.youtube.credentials_path == pathlib.Path("/tmp/test-creds.json")

    # Processing
    assert cfg.processing.max_wall_clock_hours == 12
    assert cfg.processing.gpu_device == 1
    assert cfg.processing.num_cpu_workers == 4
    assert cfg.processing.batch_size_yolo == 8
    assert cfg.processing.batch_size_whisper == 12


# ---------------------------------------------------------------------------
# Step 5: Defaults — all omitted fields get PRD values
# ---------------------------------------------------------------------------


def test_load_config_defaults() -> None:
    """Loading minimal_config.yaml applies all PRD default values."""
    from autopilot.config import load_config

    cfg = load_config(FIXTURES / "minimal_config.yaml")

    # Required fields present
    assert cfg.input_dir == pathlib.Path("/tmp/input")
    assert cfg.output_dir == pathlib.Path("/tmp/output")

    # Creator defaults
    assert cfg.creator.default_video_duration_minutes == "8-15"

    # Output defaults
    assert cfg.output.primary_aspect == "16:9"
    assert cfg.output.resolution == (1920, 1080)
    assert cfg.output.codec == "h264"
    assert cfg.output.quality_crf == 18
    assert cfg.output.audio_bitrate == "256k"
    assert cfg.output.target_loudness_lufs == -16

    # Model defaults
    assert cfg.models.whisper_size == "large-v3"
    assert cfg.models.yolo_variant == "yolo11x"
    assert cfg.models.yolo_sample_every_n_frames == 3
    assert cfg.models.clip_model == "google/siglip2-so400m-patch14"
    assert cfg.models.face_model == "buffalo_l"
    assert cfg.models.tts_engine == "kokoro"
    assert cfg.models.music_engine == "musicgen"

    # LLM defaults
    assert cfg.llm.provider == "anthropic"
    assert cfg.llm.planning_model == "claude-opus-4-20250514"
    assert cfg.llm.utility_model == "claude-sonnet-4-20250514"

    # YouTube defaults
    assert cfg.youtube.privacy_status == "unlisted"
    assert cfg.youtube.default_category == "19"

    # Processing defaults
    assert cfg.processing.max_wall_clock_hours == 36
    assert cfg.processing.gpu_device == 0
    assert cfg.processing.num_cpu_workers == 12
    assert cfg.processing.batch_size_yolo == 16
    assert cfg.processing.batch_size_whisper == 24

    # Cameras defaults to empty dict
    assert cfg.cameras == {}


def test_load_config_partial_section(tmp_path: pathlib.Path) -> None:
    """Partial model section: specified fields override, rest get defaults."""
    from autopilot.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "input_dir: /tmp/in\n"
        "output_dir: /tmp/out\n"
        "models:\n"
        '  whisper_size: "large-v3-turbo"\n'
    )
    cfg = load_config(config_file)
    assert cfg.models.whisper_size == "large-v3-turbo"
    # Remaining fields get defaults
    assert cfg.models.yolo_variant == "yolo11x"
    assert cfg.models.tts_engine == "kokoro"
    assert cfg.models.music_engine == "musicgen"


# ---------------------------------------------------------------------------
# Step 7: Path expansion — ~ is expanded in path fields
# ---------------------------------------------------------------------------


def test_load_config_path_expansion(tmp_path: pathlib.Path) -> None:
    """Tilde in path fields is expanded to absolute paths."""
    from autopilot.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "input_dir: ~/footage\n"
        "output_dir: ~/output\n"
        "youtube:\n"
        "  credentials_path: ~/.config/creds.json\n"
    )
    cfg = load_config(config_file)

    # All path fields must be absolute (no ~ remaining)
    assert cfg.input_dir.is_absolute(), f"input_dir not absolute: {cfg.input_dir}"
    assert "~" not in str(cfg.input_dir)
    assert cfg.output_dir.is_absolute(), f"output_dir not absolute: {cfg.output_dir}"
    assert "~" not in str(cfg.output_dir)
    assert cfg.youtube.credentials_path.is_absolute(), (
        f"credentials_path not absolute: {cfg.youtube.credentials_path}"
    )
    assert "~" not in str(cfg.youtube.credentials_path)


def test_load_config_default_credentials_path_expanded(tmp_path: pathlib.Path) -> None:
    """Default credentials_path (youtube section omitted) is also expanded."""
    from autopilot.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text("input_dir: /tmp/in\noutput_dir: /tmp/out\n")
    cfg = load_config(config_file)

    assert cfg.youtube.credentials_path.is_absolute(), (
        f"default credentials_path not absolute: {cfg.youtube.credentials_path}"
    )
    assert "~" not in str(cfg.youtube.credentials_path)


# ---------------------------------------------------------------------------
# Step 9: Missing required fields
# ---------------------------------------------------------------------------


def test_load_config_missing_input_dir(tmp_path: pathlib.Path) -> None:
    """ConfigError raised when input_dir is missing."""
    from autopilot.config import ConfigError, load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text("output_dir: /tmp/out\n")
    with pytest.raises(ConfigError, match="input_dir"):
        load_config(config_file)


def test_load_config_missing_output_dir(tmp_path: pathlib.Path) -> None:
    """ConfigError raised when output_dir is missing."""
    from autopilot.config import ConfigError, load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text("input_dir: /tmp/in\n")
    with pytest.raises(ConfigError, match="output_dir"):
        load_config(config_file)


def test_load_config_empty_yaml(tmp_path: pathlib.Path) -> None:
    """ConfigError raised when YAML document is empty (None from safe_load)."""
    from autopilot.config import ConfigError, load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text("")
    with pytest.raises(ConfigError):
        load_config(config_file)
