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
    dep_names = {
        d.split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip().lower()
        for d in deps
    }

    required = [
        "click",
        "pyyaml",
        "whisperx",
        "faster-whisper",
        "ultralytics",
        "insightface",
        "transformers",
        "faiss-cpu",
        "panns-inference",
        "audiocraft",
        "scikit-learn",
        "moviepy",
        "av",
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
    dev_names = {
        d.split("[")[0].split(">")[0].split("<")[0].split("=")[0].split("!")[0].strip().lower()
        for d in dev_deps
    }

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


# ---------------------------------------------------------------------------
# Dependency version constraint tests
# ---------------------------------------------------------------------------

# ML/volatile dependencies that MUST have lower-bound (>=) and upper-bound (<) constraints
ML_DEPS = {
    "whisperx",
    "faster-whisper",
    "ultralytics",
    "insightface",
    "transformers",
    "audiocraft",
    "moviepy",
    "opentimelineio",
    "scenedetect",
    "panns-inference",
    "kokoro",
    "av",
    "qwen-vl-utils",
    "faiss-cpu",
    "scikit-learn",
}

# Stable deps that should NOT be upper-bounded
STABLE_DEPS = {
    "click",
    "pyyaml",
    "requests",
    "numpy",
    "pillow",
    "anthropic",
    "google-api-python-client",
}


def _parse_dep_name(dep: str) -> str:
    """Extract the bare package name (lowercase) from a dependency specifier."""
    # Strip extras like [cuda], then strip version operators
    name = dep.split("[")[0]
    for op in (">=", "<=", "!=", "==", ">", "<", "~="):
        name = name.split(op)[0]
    return name.strip().lower()


def test_ml_deps_have_version_constraints(project_root: pathlib.Path) -> None:
    """ML/volatile dependencies must have both lower-bound (>=) and upper-bound (<) constraints."""
    with open(project_root / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    deps = data["project"].get("dependencies", [])
    dep_map = {_parse_dep_name(d): d for d in deps}

    for pkg in ML_DEPS:
        assert pkg in dep_map, f"ML dependency {pkg} not found in pyproject.toml"
        spec = dep_map[pkg]
        assert ">=" in spec, f"{pkg} must have a lower-bound (>=) constraint, got: {spec}"
        assert "<" in spec and "<=" not in spec.replace("<=", ""), (
            f"{pkg} must have an upper-bound (<) constraint, got: {spec}"
        )


def test_stable_deps_not_overconstrained(project_root: pathlib.Path) -> None:
    """Stable ecosystem deps must NOT have upper-bound (<) constraints."""
    with open(project_root / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)

    deps = data["project"].get("dependencies", [])
    dep_map = {_parse_dep_name(d): d for d in deps}

    for pkg in STABLE_DEPS:
        if pkg not in dep_map:
            continue  # not listed, that's fine for this test
        spec = dep_map[pkg]
        # Remove the package name to get just the version specifier part
        name_end = spec.split(">")[0].split("<")[0]
        name_end = name_end.split("=")[0].split("!")[0].split("[")[0]
        version_part = spec[len(name_end) :]
        # Check there's no standalone < (not <=, not part of >=)
        assert "<" not in version_part.replace("<=", "").replace("<<", ""), (
            f"Stable dep {pkg} should NOT have upper-bound constraint, got: {spec}"
        )


# ---------------------------------------------------------------------------
# .gitignore and package version tests
# ---------------------------------------------------------------------------


def test_uv_lock_file_exists(project_root: pathlib.Path) -> None:
    """uv.lock must exist at the project root for reproducible installs."""
    lock_path = project_root / "uv.lock"
    assert lock_path.is_file(), "uv.lock must exist at the project root"
    content = lock_path.read_text()
    assert len(content) > 0, "uv.lock must not be empty"


def test_uv_lock_not_gitignored(project_root: pathlib.Path) -> None:
    """uv.lock must NOT be in .gitignore — it should be tracked in git."""
    gitignore_path = project_root / ".gitignore"
    if not gitignore_path.is_file():
        return  # no .gitignore, so uv.lock can't be ignored
    content = gitignore_path.read_text()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert stripped != "uv.lock", ".gitignore must NOT contain 'uv.lock'"


def test_gitignore_has_output_dir(project_root: pathlib.Path) -> None:
    """'.gitignore' contains 'output/' to prevent committing rendered videos."""
    gitignore_path = project_root / ".gitignore"
    assert gitignore_path.is_file(), ".gitignore must exist"
    content = gitignore_path.read_text()
    assert "output/" in content, ".gitignore must contain 'output/'"


def test_package_version(project_root: pathlib.Path) -> None:
    """autopilot package exposes a valid semver __version__."""
    import re
    import sys

    # Ensure the project root is on sys.path so we can import autopilot
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    import importlib

    # Force reimport in case it was cached
    if "autopilot" in sys.modules:
        importlib.reload(sys.modules["autopilot"])
    else:
        importlib.import_module("autopilot")

    import autopilot

    assert hasattr(autopilot, "__version__"), "autopilot must have __version__"
    # Basic semver: MAJOR.MINOR.PATCH
    assert re.match(r"^\d+\.\d+\.\d+", autopilot.__version__), (
        f"__version__ must be valid semver, got: {autopilot.__version__}"
    )
