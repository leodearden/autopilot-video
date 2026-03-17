"""Typed configuration dataclasses and YAML loader for autopilot-video."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "AutopilotConfig",
    "CameraConfig",
    "ConfigError",
    "CreatorConfig",
    "LLMConfig",
    "ModelConfig",
    "OutputConfig",
    "ProcessingConfig",
    "YouTubeConfig",
    "load_config",
]


class ConfigError(Exception):
    """Raised for any configuration-related error."""


@dataclass
class CreatorConfig:
    """Creator profile used in LLM prompts."""

    name: str = ""
    channel_style: str = ""
    target_audience: str = ""
    default_video_duration_minutes: str = "8-15"
    narration_style: str = ""
    music_preference: str = ""


@dataclass
class CameraConfig:
    """Per-camera profile for source footage."""

    source_resolution: tuple[int, int] = (1920, 1080)
    aspect_mode: str = "landscape"
    has_gyro_data: bool = False
    default_crop_target: str = "16:9"
    crop_smoothing_tau: float = 0.5


@dataclass
class OutputConfig:
    """Output encoding and format settings."""

    primary_aspect: str = "16:9"
    resolution: tuple[int, int] = (1920, 1080)
    codec: str = "h264"
    quality_crf: int = 18
    audio_bitrate: str = "256k"
    target_loudness_lufs: int = -16


@dataclass
class ModelConfig:
    """ML model selection and parameters."""

    whisper_size: str = "large-v3"
    yolo_variant: str = "yolo11x"
    yolo_sample_every_n_frames: int = 3
    clip_model: str = "google/siglip2-so400m-patch14"
    face_model: str = "buffalo_l"
    tts_engine: str = "kokoro"
    music_engine: str = "musicgen"


@dataclass
class LLMConfig:
    """LLM provider and model settings."""

    provider: str = "anthropic"
    planning_model: str = "claude-opus-4-20250514"
    utility_model: str = "claude-sonnet-4-20250514"


@dataclass
class YouTubeConfig:
    """YouTube upload settings."""

    privacy_status: str = "unlisted"
    default_category: str = "19"
    credentials_path: Path = Path("~/.config/autopilot/youtube_oauth.json")


@dataclass
class ProcessingConfig:
    """Resource allocation and processing limits."""

    max_wall_clock_hours: int = 36
    gpu_device: int = 0
    num_cpu_workers: int = 12
    batch_size_yolo: int = 16
    batch_size_whisper: int = 24


@dataclass
class AutopilotConfig:
    """Top-level configuration container."""

    input_dir: Path = Path(".")
    output_dir: Path = Path(".")
    creator: CreatorConfig = field(default_factory=CreatorConfig)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    output: OutputConfig = field(default_factory=OutputConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    youtube: YouTubeConfig = field(default_factory=YouTubeConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)


def _to_tuple(value: Any) -> tuple[int, int]:
    """Convert a list/sequence to a 2-tuple of ints."""
    return (int(value[0]), int(value[1]))


def _build_creator(data: dict[str, Any]) -> CreatorConfig:
    """Build CreatorConfig from parsed YAML dict."""
    return CreatorConfig(
        name=str(data.get("name", "")),
        channel_style=str(data.get("channel_style", "")),
        target_audience=str(data.get("target_audience", "")),
        default_video_duration_minutes=str(data.get("default_video_duration_minutes", "8-15")),
        narration_style=str(data.get("narration_style", "")),
        music_preference=str(data.get("music_preference", "")),
    )


def _build_camera(data: dict[str, Any]) -> CameraConfig:
    """Build CameraConfig from parsed YAML dict."""
    result = CameraConfig()
    if "source_resolution" in data:
        result.source_resolution = _to_tuple(data["source_resolution"])
    if "aspect_mode" in data:
        result.aspect_mode = str(data["aspect_mode"])
    if "has_gyro_data" in data:
        result.has_gyro_data = bool(data["has_gyro_data"])
    if "default_crop_target" in data:
        result.default_crop_target = str(data["default_crop_target"])
    if "crop_smoothing_tau" in data:
        result.crop_smoothing_tau = float(data["crop_smoothing_tau"])
    return result


def _build_output(data: dict[str, Any]) -> OutputConfig:
    """Build OutputConfig from parsed YAML dict."""
    result = OutputConfig()
    if "primary_aspect" in data:
        result.primary_aspect = str(data["primary_aspect"])
    if "resolution" in data:
        result.resolution = _to_tuple(data["resolution"])
    if "codec" in data:
        result.codec = str(data["codec"])
    if "quality_crf" in data:
        result.quality_crf = int(data["quality_crf"])
    if "audio_bitrate" in data:
        result.audio_bitrate = str(data["audio_bitrate"])
    if "target_loudness_lufs" in data:
        result.target_loudness_lufs = int(data["target_loudness_lufs"])
    return result


def _build_models(data: dict[str, Any]) -> ModelConfig:
    """Build ModelConfig from parsed YAML dict."""
    result = ModelConfig()
    if "whisper_size" in data:
        result.whisper_size = str(data["whisper_size"])
    if "yolo_variant" in data:
        result.yolo_variant = str(data["yolo_variant"])
    if "yolo_sample_every_n_frames" in data:
        result.yolo_sample_every_n_frames = int(data["yolo_sample_every_n_frames"])
    if "clip_model" in data:
        result.clip_model = str(data["clip_model"])
    if "face_model" in data:
        result.face_model = str(data["face_model"])
    if "tts_engine" in data:
        result.tts_engine = str(data["tts_engine"])
    if "music_engine" in data:
        result.music_engine = str(data["music_engine"])
    return result


def _build_llm(data: dict[str, Any]) -> LLMConfig:
    """Build LLMConfig from parsed YAML dict."""
    result = LLMConfig()
    if "provider" in data:
        result.provider = str(data["provider"])
    if "planning_model" in data:
        result.planning_model = str(data["planning_model"])
    if "utility_model" in data:
        result.utility_model = str(data["utility_model"])
    return result


def _build_youtube(data: dict[str, Any]) -> YouTubeConfig:
    """Build YouTubeConfig from parsed YAML dict."""
    result = YouTubeConfig()
    if "privacy_status" in data:
        result.privacy_status = str(data["privacy_status"])
    if "default_category" in data:
        result.default_category = str(data["default_category"])
    if "credentials_path" in data:
        result.credentials_path = Path(str(data["credentials_path"]))
    return result


def _build_processing(data: dict[str, Any]) -> ProcessingConfig:
    """Build ProcessingConfig from parsed YAML dict."""
    result = ProcessingConfig()
    if "max_wall_clock_hours" in data:
        result.max_wall_clock_hours = int(data["max_wall_clock_hours"])
    if "gpu_device" in data:
        result.gpu_device = int(data["gpu_device"])
    if "num_cpu_workers" in data:
        result.num_cpu_workers = int(data["num_cpu_workers"])
    if "batch_size_yolo" in data:
        result.batch_size_yolo = int(data["batch_size_yolo"])
    if "batch_size_whisper" in data:
        result.batch_size_whisper = int(data["batch_size_whisper"])
    return result


def load_config(path: str | Path) -> AutopilotConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Fully validated AutopilotConfig instance.

    Raises:
        ConfigError: For any configuration error (missing file, parse error,
            missing required fields, invalid values).
    """
    path = Path(path)

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    # Validate required fields
    missing = [f for f in ("input_dir", "output_dir") if f not in raw]
    if missing:
        raise ConfigError(f"Missing required config fields: {', '.join(missing)}")

    # Build sub-configs from sections (defaulting to empty dicts)
    creator = _build_creator(raw.get("creator", {}))

    cameras_raw = raw.get("cameras", {}) or {}
    cameras = {name: _build_camera(cam) for name, cam in cameras_raw.items()}

    output = _build_output(raw.get("output", {}))
    models = _build_models(raw.get("models", {}))
    llm = _build_llm(raw.get("llm", {}))
    youtube = _build_youtube(raw.get("youtube", {}))
    processing = _build_processing(raw.get("processing", {}))

    # Expand ~ in path fields
    youtube.credentials_path = youtube.credentials_path.expanduser()

    return AutopilotConfig(
        input_dir=Path(str(raw.get("input_dir", "."))).expanduser(),
        output_dir=Path(str(raw.get("output_dir", "."))).expanduser(),
        creator=creator,
        cameras=cameras,
        output=output,
        models=models,
        llm=llm,
        youtube=youtube,
        processing=processing,
    )
