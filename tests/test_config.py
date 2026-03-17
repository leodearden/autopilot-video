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


# ---------------------------------------------------------------------------
# Step 11: Invalid enum values
# ---------------------------------------------------------------------------

_BASE_REQUIRED = "input_dir: /tmp/in\noutput_dir: /tmp/out\n"


@pytest.mark.parametrize(
    "section,field,bad_value",
    [
        ("models", "whisper_size", "tiny"),
        ("models", "yolo_variant", "yolo5"),
        ("models", "tts_engine", "openai"),
        ("models", "music_engine", "suno"),
        ("youtube", "privacy_status", "public"),
    ],
)
def test_load_config_invalid_enums(
    tmp_path: pathlib.Path, section: str, field: str, bad_value: str
) -> None:
    """ConfigError raised for invalid enum-like values, mentioning field and value."""
    from autopilot.config import ConfigError, load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"{_BASE_REQUIRED}{section}:\n  {field}: {bad_value}\n")
    with pytest.raises(ConfigError, match=field):
        load_config(config_file)


# ---------------------------------------------------------------------------
# Step 13: Numeric bounds validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section,field,bad_value",
    [
        ("output", "quality_crf", -1),
        ("output", "quality_crf", 52),
        ("models", "yolo_sample_every_n_frames", 0),
        ("processing", "max_wall_clock_hours", 0),
        ("processing", "num_cpu_workers", 0),
        ("processing", "batch_size_yolo", 0),
        ("processing", "batch_size_whisper", 0),
        ("processing", "gpu_device", -1),
    ],
)
def test_load_config_numeric_bounds_invalid(
    tmp_path: pathlib.Path, section: str, field: str, bad_value: int
) -> None:
    """ConfigError raised for out-of-bounds numeric values."""
    from autopilot.config import ConfigError, load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"{_BASE_REQUIRED}{section}:\n  {field}: {bad_value}\n")
    with pytest.raises(ConfigError, match=field):
        load_config(config_file)


@pytest.mark.parametrize(
    "section,field,good_value",
    [
        ("output", "quality_crf", 0),
        ("output", "quality_crf", 51),
        ("processing", "gpu_device", 0),
        ("models", "yolo_sample_every_n_frames", 1),
    ],
)
def test_load_config_numeric_bounds_valid(
    tmp_path: pathlib.Path, section: str, field: str, good_value: int
) -> None:
    """Valid boundary values do not raise."""
    from autopilot.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"{_BASE_REQUIRED}{section}:\n  {field}: {good_value}\n")
    cfg = load_config(config_file)
    # Reach into the right sub-config
    sub = getattr(cfg, section if section != "models" else "models")
    assert getattr(sub, field) == good_value


# ---------------------------------------------------------------------------
# Step 15: File-level errors (missing file, invalid YAML)
# ---------------------------------------------------------------------------


def test_load_config_nonexistent_file(tmp_path: pathlib.Path) -> None:
    """ConfigError (not FileNotFoundError) raised for missing file."""
    from autopilot.config import ConfigError, load_config

    with pytest.raises(ConfigError):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_invalid_yaml_syntax(tmp_path: pathlib.Path) -> None:
    """ConfigError (not yaml.YAMLError) raised for malformed YAML."""
    from autopilot.config import ConfigError, load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: [invalid\n  yaml: {{{\n")
    with pytest.raises(ConfigError):
        load_config(config_file)


def test_load_config_yaml_null_document(tmp_path: pathlib.Path) -> None:
    """YAML null document raises ConfigError about missing required fields."""
    from autopilot.config import ConfigError, load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text("---\n...\n")
    with pytest.raises(ConfigError):
        load_config(config_file)


# ---------------------------------------------------------------------------
# Step 17: Camera parsing edge cases
# ---------------------------------------------------------------------------


def test_load_config_two_cameras(tmp_path: pathlib.Path) -> None:
    """Multiple camera profiles parsed into dict[str, CameraConfig]."""
    from autopilot.config import CameraConfig, load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"{_BASE_REQUIRED}"
        "cameras:\n"
        "  cam_a:\n"
        "    source_resolution: [3840, 2160]\n"
        "    aspect_mode: landscape\n"
        "  cam_b:\n"
        "    source_resolution: [1920, 1080]\n"
        "    has_gyro_data: true\n"
    )
    cfg = load_config(config_file)
    assert len(cfg.cameras) == 2
    assert "cam_a" in cfg.cameras
    assert "cam_b" in cfg.cameras
    assert isinstance(cfg.cameras["cam_a"], CameraConfig)
    assert cfg.cameras["cam_a"].source_resolution == (3840, 2160)
    assert cfg.cameras["cam_a"].aspect_mode == "landscape"
    assert cfg.cameras["cam_b"].source_resolution == (1920, 1080)
    assert cfg.cameras["cam_b"].has_gyro_data is True


def test_load_config_cameras_empty_dict(tmp_path: pathlib.Path) -> None:
    """cameras: {} -> empty dict."""
    from autopilot.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(f"{_BASE_REQUIRED}cameras: {{}}\n")
    cfg = load_config(config_file)
    assert cfg.cameras == {}


def test_load_config_camera_types(tmp_path: pathlib.Path) -> None:
    """Camera fields have correct types: tuple, float, bool."""
    from autopilot.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"{_BASE_REQUIRED}"
        "cameras:\n"
        "  test_cam:\n"
        "    source_resolution: [4096, 2160]\n"
        "    crop_smoothing_tau: 0.3\n"
        "    has_gyro_data: false\n"
    )
    cfg = load_config(config_file)
    cam = cfg.cameras["test_cam"]
    assert isinstance(cam.source_resolution, tuple)
    assert isinstance(cam.crop_smoothing_tau, float)
    assert isinstance(cam.has_gyro_data, bool)


def test_load_config_camera_partial_defaults(tmp_path: pathlib.Path) -> None:
    """Partial camera profile fills in CameraConfig defaults."""
    from autopilot.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"{_BASE_REQUIRED}"
        "cameras:\n"
        "  minimal_cam:\n"
        "    aspect_mode: portrait\n"
    )
    cfg = load_config(config_file)
    cam = cfg.cameras["minimal_cam"]
    assert cam.aspect_mode == "portrait"
    # Defaults
    assert cam.source_resolution == (1920, 1080)
    assert cam.has_gyro_data is False
    assert cam.default_crop_target == "16:9"
    assert cam.crop_smoothing_tau == 0.5


# ---------------------------------------------------------------------------
# Step 19: Integration test with project config.yaml
# ---------------------------------------------------------------------------


def test_load_config_integration(project_root: pathlib.Path) -> None:
    """Load the actual project config.yaml and verify key values."""
    from autopilot.config import AutopilotConfig, load_config

    cfg = load_config(project_root / "config.yaml")

    assert isinstance(cfg, AutopilotConfig)

    # Camera
    assert "dji_osmo_action_6" in cfg.cameras
    assert cfg.cameras["dji_osmo_action_6"].source_resolution == (4096, 4096)

    # Models
    assert cfg.models.whisper_size == "large-v3"

    # Output
    assert cfg.output.quality_crf == 18

    # LLM
    assert cfg.llm.provider == "anthropic"

    # Paths are expanded
    assert cfg.input_dir.is_absolute()
    assert cfg.output_dir.is_absolute()
    assert cfg.youtube.credentials_path.is_absolute()
