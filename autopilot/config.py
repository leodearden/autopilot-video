"""Typed configuration dataclasses and YAML loader for autopilot-video."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    raise NotImplementedError
