"""Tests for project scaffolding: directory structure, pyproject.toml, config.yaml."""

import pathlib
import tomllib

import yaml


def test_directory_structure_exists(project_root: pathlib.Path) -> None:
    """All required directories exist."""
    required_dirs = [
        "autopilot",
        "autopilot/ingest",
        "autopilot/analyze",
        "autopilot/organize",
        "autopilot/plan",
        "autopilot/source",
        "autopilot/render",
        "autopilot/upload",
        "autopilot/prompts",
        "tests/fixtures",
        "scripts",
        "output",
    ]
    for d in required_dirs:
        assert (project_root / d).is_dir(), f"Directory {d}/ must exist"


def test_python_packages_have_init(project_root: pathlib.Path) -> None:
    """Every Python package has an __init__.py file."""
    python_packages = [
        "autopilot",
        "autopilot/ingest",
        "autopilot/analyze",
        "autopilot/organize",
        "autopilot/plan",
        "autopilot/source",
        "autopilot/render",
        "autopilot/upload",
    ]
    for pkg in python_packages:
        init = project_root / pkg / "__init__.py"
        assert init.is_file(), f"{pkg}/__init__.py must exist"


def test_prompts_dir_is_not_python_package(project_root: pathlib.Path) -> None:
    """The prompts directory is for .md templates, not a Python package."""
    assert not (project_root / "autopilot" / "prompts" / "__init__.py").exists(), (
        "autopilot/prompts/ should NOT have __init__.py"
    )


# ---------------------------------------------------------------------------
# pyproject.toml tests
# ---------------------------------------------------------------------------


def test_pyproject_toml_valid(project_root: pathlib.Path) -> None:
    """pyproject.toml is valid TOML with correct build system and metadata."""
    toml_path = project_root / "pyproject.toml"
    assert toml_path.is_file(), "pyproject.toml must exist"

    with open(toml_path, "rb") as f:
        data = tomllib.load(f)

    # Build system
    assert data["build-system"]["requires"] == ["hatchling"]
    assert data["build-system"]["build-backend"] == "hatchling.build"

    # Project metadata
    assert data["project"]["name"] == "autopilot-video"
    assert data["project"]["requires-python"] == ">=3.11"

    # Optional dependency groups exist
    opt_deps = data["project"].get("optional-dependencies", {})
    assert "dev" in opt_deps, "dev optional-dependencies group must exist"


def test_pyproject_toml_has_all_dependencies(project_root: pathlib.Path) -> None:
    """All PRD-required runtime dependencies are listed."""
    with open(project_root / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    deps = data["project"].get("dependencies", [])
    # Normalize: lowercase, strip extras/version specifiers for matching
    dep_names = {d.split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip().lower() for d in deps}

    required = [
        "click",
        "pyyaml",
        "whisperx",
        "faster-whisper",
        "ultralytics",
        "insightface",
        "transformers",
        "faiss-gpu",
        "panns-inference",
        "audiocraft",
        "scikit-learn",
        "moviepy",
        "pyav",
        "opentimelineio",
        "scenedetect",
        "kokoro",
        "anthropic",
        "google-api-python-client",
        "requests",
    ]
    for pkg in required:
        assert pkg in dep_names, f"Missing dependency: {pkg}"


def test_pyproject_toml_has_dev_dependencies(project_root: pathlib.Path) -> None:
    """Dev optional-dependencies include pytest, ruff, and pyright."""
    with open(project_root / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    dev_deps = data["project"]["optional-dependencies"]["dev"]
    dev_names = {d.split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip().lower() for d in dev_deps}

    for tool in ["pytest", "ruff", "pyright"]:
        assert tool in dev_names, f"Missing dev dependency: {tool}"


# ---------------------------------------------------------------------------
# config.yaml tests
# ---------------------------------------------------------------------------


def _load_config(project_root: pathlib.Path) -> dict:
    """Helper to load and parse config.yaml."""
    config_path = project_root / "config.yaml"
    assert config_path.is_file(), "config.yaml must exist"
    with open(config_path) as f:
        return yaml.safe_load(f)


def test_config_yaml_valid(project_root: pathlib.Path) -> None:
    """config.yaml is valid YAML with all required top-level keys."""
    data = _load_config(project_root)
    required_keys = [
        "input_dir",
        "output_dir",
        "creator",
        "cameras",
        "output",
        "models",
        "llm",
        "youtube",
        "processing",
    ]
    for key in required_keys:
        assert key in data, f"Missing top-level key: {key}"


def test_config_yaml_creator_section(project_root: pathlib.Path) -> None:
    """Creator section has all required keys."""
    data = _load_config(project_root)
    creator = data["creator"]
    required = [
        "name",
        "channel_style",
        "target_audience",
        "default_video_duration_minutes",
        "narration_style",
        "music_preference",
    ]
    for key in required:
        assert key in creator, f"Missing creator key: {key}"


def test_config_yaml_models_section(project_root: pathlib.Path) -> None:
    """Models section has all required keys."""
    data = _load_config(project_root)
    models = data["models"]
    required = [
        "whisper_size",
        "yolo_variant",
        "yolo_sample_every_n_frames",
        "clip_model",
        "face_model",
        "tts_engine",
        "music_engine",
    ]
    for key in required:
        assert key in models, f"Missing models key: {key}"


def test_config_yaml_processing_section(project_root: pathlib.Path) -> None:
    """Processing section has all required keys."""
    data = _load_config(project_root)
    processing = data["processing"]
    required = [
        "max_wall_clock_hours",
        "gpu_device",
        "num_cpu_workers",
        "batch_size_yolo",
        "batch_size_whisper",
    ]
    for key in required:
        assert key in processing, f"Missing processing key: {key}"


def test_config_yaml_output_section(project_root: pathlib.Path) -> None:
    """Output section has all required keys."""
    data = _load_config(project_root)
    output = data["output"]
    required = [
        "primary_aspect",
        "resolution",
        "codec",
        "quality_crf",
        "audio_bitrate",
        "target_loudness_lufs",
    ]
    for key in required:
        assert key in output, f"Missing output key: {key}"


def test_config_yaml_llm_section(project_root: pathlib.Path) -> None:
    """LLM section has all required keys."""
    data = _load_config(project_root)
    llm = data["llm"]
    required = ["provider", "planning_model", "utility_model"]
    for key in required:
        assert key in llm, f"Missing llm key: {key}"


def test_config_yaml_youtube_section(project_root: pathlib.Path) -> None:
    """YouTube section has all required keys."""
    data = _load_config(project_root)
    youtube = data["youtube"]
    required = ["privacy_status", "default_category", "credentials_path"]
    for key in required:
        assert key in youtube, f"Missing youtube key: {key}"
