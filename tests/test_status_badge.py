"""Tests for macros/status_badge.html — generic status badge macro."""

from __future__ import annotations

import re
from pathlib import Path

import jinja2

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "autopilot" / "web" / "templates"


def _extract_classes(html: str) -> list[str]:
    """Parse the class attribute of the first span in the rendered HTML into a list."""
    m = re.search(r'<span[^>]+class="([^"]*)"', html)
    if not m:
        return []
    return m.group(1).split()


def _render_badge(
    status: str,
    color_map: dict[str, str],
    extra_classes: str = "",
    label: str | None = None,
    default: str = "bg-gray-700 text-gray-300",
) -> str:
    """Render the generic status_badge macro with the given arguments."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("macros/status_badge.html")
    module = template.module
    kwargs: dict = {"color_map": color_map, "extra_classes": extra_classes, "default": default}
    if label is not None:
        kwargs["label"] = label
    return module.status_badge(status, **kwargs)  # type: ignore[reportAttributeAccessIssue]


def _render_badge_ext(
    status: str,
    color_map: dict[str, str],
    extra_classes: str = "",
    label: str | None = None,
    default: str = "bg-gray-700 text-gray-300",
    px: str = "px-2",
    py: str = "py-1",
    rounded: str = "rounded",
    text_size: str = "text-xs",
    font_weight: str = "font-medium",
) -> str:
    """Render the generic status_badge macro with named override parameters."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("macros/status_badge.html")
    module = template.module
    kwargs: dict = {
        "color_map": color_map,
        "extra_classes": extra_classes,
        "default": default,
        "px": px,
        "py": py,
        "rounded": rounded,
        "text_size": text_size,
        "font_weight": font_weight,
    }
    if label is not None:
        kwargs["label"] = label
    return module.status_badge(status, **kwargs)  # type: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# Unit tests for the generic macro
# ---------------------------------------------------------------------------


class TestGenericStatusBadge:
    """Verify color_map lookup and fallback behavior."""

    SAMPLE_MAP = {
        "running": "bg-blue-900 text-blue-300",
        "completed": "bg-green-900 text-green-300",
        "failed": "bg-red-900 text-red-300",
    }

    def test_known_status_returns_correct_classes(self) -> None:
        html = _render_badge("running", self.SAMPLE_MAP)
        assert "bg-blue-900" in html
        assert "text-blue-300" in html

    def test_unknown_status_falls_back_to_default_gray(self) -> None:
        html = _render_badge("cancelled", self.SAMPLE_MAP)
        assert "bg-gray-700" in html
        assert "text-gray-300" in html

    def test_none_status_shows_unknown_text_and_default_color(self) -> None:
        html = _render_badge(None, self.SAMPLE_MAP)  # type: ignore[arg-type]
        assert "unknown" in html
        assert "bg-gray-700" in html
        assert "text-gray-300" in html

    def test_empty_string_status_shows_unknown(self) -> None:
        html = _render_badge("", self.SAMPLE_MAP)
        assert "unknown" in html

    def test_status_text_displayed(self) -> None:
        html = _render_badge("completed", self.SAMPLE_MAP)
        assert "completed" in html

    def test_base_classes_present(self) -> None:
        html = _render_badge("running", self.SAMPLE_MAP)
        assert "px-2" in html
        assert "py-1" in html
        assert "rounded" in html
        assert "text-xs" in html
        assert "font-medium" in html


class TestExtraClasses:
    """Verify the extra_classes parameter adds classes to the badge span."""

    SAMPLE_MAP = {"ok": "bg-green-900 text-green-300"}

    def test_extra_classes_included(self) -> None:
        html = _render_badge("ok", self.SAMPLE_MAP, extra_classes="ml-2 whitespace-nowrap")
        assert "ml-2" in html
        assert "whitespace-nowrap" in html

    def test_no_extra_classes_by_default(self) -> None:
        html = _render_badge("ok", self.SAMPLE_MAP)
        assert "ml-2" not in html


class TestLabelParameter:
    """Verify the label parameter overrides the displayed text."""

    SAMPLE_MAP = {"done": "bg-green-900 text-green-300"}

    def test_custom_label_overrides_status_text(self) -> None:
        html = _render_badge("done", self.SAMPLE_MAP, label="done: 5")
        assert "done: 5" in html

    def test_label_none_defaults_to_status_text(self) -> None:
        html = _render_badge("done", self.SAMPLE_MAP)
        assert "done" in html


class TestDefaultColorOverride:
    """Verify passing a custom default uses that color for unknown statuses."""

    SAMPLE_MAP = {"known": "bg-green-900 text-green-300"}

    def test_purple_default_for_unknown(self) -> None:
        html = _render_badge(
            "mystery",
            self.SAMPLE_MAP,
            default="bg-purple-900 text-purple-300",
        )
        assert "bg-purple-900" in html
        assert "text-purple-300" in html

    def test_purple_default_not_gray(self) -> None:
        html = _render_badge(
            "mystery",
            self.SAMPLE_MAP,
            default="bg-purple-900 text-purple-300",
        )
        assert "bg-gray-700" not in html


class TestClassNormalization:
    """Verify whitespace is normalized in the rendered class attribute."""

    SAMPLE_MAP = {"ok": "bg-green-900 text-green-300"}

    def test_no_double_space_when_extra_classes_empty(self) -> None:
        html = _render_badge("ok", self.SAMPLE_MAP)
        m = re.search(r'class="([^"]*)"', html)
        assert m is not None
        assert "  " not in m.group(1)


class TestOverrideParameters:
    """Verify named override parameters for layout/typography classes."""

    SAMPLE_MAP = {"ok": "bg-green-900 text-green-300"}

    def test_px_override_replaces_default(self) -> None:
        html = _render_badge_ext("ok", self.SAMPLE_MAP, px="px-3")
        classes = _extract_classes(html)
        px_classes = [c for c in classes if c.startswith("px-")]
        assert len(px_classes) == 1
        assert px_classes[0] == "px-3"

    def test_py_override_replaces_default(self) -> None:
        html = _render_badge_ext("ok", self.SAMPLE_MAP, py="py-0.5")
        classes = _extract_classes(html)
        py_classes = [c for c in classes if c.startswith("py-")]
        assert len(py_classes) == 1
        assert py_classes[0] == "py-0.5"

    def test_rounded_override_replaces_default(self) -> None:
        html = _render_badge_ext("ok", self.SAMPLE_MAP, rounded="rounded-full")
        classes = _extract_classes(html)
        assert "rounded-full" in classes
        assert "rounded" not in classes

    def test_text_size_override_replaces_default(self) -> None:
        html = _render_badge_ext("ok", self.SAMPLE_MAP, text_size="text-sm")
        classes = _extract_classes(html)
        assert "text-sm" in classes
        assert "text-xs" not in classes

    def test_font_weight_empty_drops_class(self) -> None:
        html = _render_badge_ext("ok", self.SAMPLE_MAP, font_weight="")
        classes = _extract_classes(html)
        assert not any(c.startswith("font-") for c in classes)

    def test_override_does_not_duplicate_default(self) -> None:
        html = _render_badge_ext("ok", self.SAMPLE_MAP, px="px-3")
        classes = _extract_classes(html)
        px_classes = [c for c in classes if c.startswith("px-")]
        assert len(px_classes) == 1


# ---------------------------------------------------------------------------
# Template integration tests — source + render assertions
# ---------------------------------------------------------------------------

_IMPORT_LINE = "{% from 'macros/status_badge.html' import status_badge %}"


def _make_env() -> jinja2.Environment:
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )


class TestPipelineIndexBadges:
    """Verify pipeline/index.html uses the generic status_badge macro."""

    def test_source_contains_import(self) -> None:
        source = (TEMPLATES_DIR / "pipeline" / "index.html").read_text()
        assert _IMPORT_LINE in source

    def test_running_run_badge_blue(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/index.html").render(
            page_title="Pipeline",
            run={
                "run_id": "r-1", "status": "running",
                "current_stage": "ingest", "started_at": "now",
            },
            stages=[],
            runs=[],
        )
        assert "bg-blue-900" in html

    def test_completed_run_badge_green(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/index.html").render(
            page_title="Pipeline",
            run={
                "run_id": "r-1", "status": "completed",
                "current_stage": None, "started_at": "now",
            },
            stages=[],
            runs=[],
        )
        assert "bg-green-900" in html

    def test_failed_run_badge_red(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/index.html").render(
            page_title="Pipeline",
            run={"run_id": "r-1", "status": "failed", "current_stage": None, "started_at": "now"},
            stages=[],
            runs=[],
        )
        assert "bg-red-900" in html

    def test_stage_done_badge_green(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/index.html").render(
            page_title="Pipeline",
            run={
                "run_id": "r-1", "status": "running",
                "current_stage": "ingest", "started_at": "now",
            },
            stages=[{
                "name": "ingest", "status": "done",
                "done": 3, "total": 3, "gate_mode": "auto",
            }],
            runs=[],
        )
        assert "bg-green-900" in html

    def test_stage_waiting_badge_amber(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/index.html").render(
            page_title="Pipeline",
            run={
                "run_id": "r-1", "status": "running",
                "current_stage": "ingest", "started_at": "now",
            },
            stages=[{
                "name": "render", "status": "waiting",
                "done": 0, "total": 2, "gate_mode": "manual",
            }],
            runs=[],
        )
        assert "bg-amber-900" in html


class TestPipelineStagesBadges:
    """Verify pipeline/stages.html uses the generic status_badge macro."""

    def test_source_contains_import(self) -> None:
        source = (TEMPLATES_DIR / "pipeline" / "stages.html").read_text()
        assert _IMPORT_LINE in source

    def test_stage_status_badge(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/stages.html").render(
            page_title="Stages",
            run={"run_id": "r-1"},
            stages=[{"name": "ingest", "status": "done", "done": 3, "total": 3,
                      "gate_mode": "auto", "counts": {}}],
        )
        assert "bg-green-900" in html

    def test_summary_count_label(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/stages.html").render(
            page_title="Stages",
            run={"run_id": "r-1"},
            stages=[{"name": "ingest", "status": "running", "done": 2, "total": 5,
                      "gate_mode": "auto", "counts": {"done": 3}}],
        )
        assert "done: 3" in html


class TestPipelineJobsBadges:
    """Verify pipeline/jobs.html uses the generic status_badge macro."""

    def test_source_contains_import(self) -> None:
        source = (TEMPLATES_DIR / "pipeline" / "jobs.html").read_text()
        assert _IMPORT_LINE in source

    def test_done_badge_green(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/jobs.html").render(
            page_title="Jobs",
            jobs=[{"job_id": "j-1", "stage": "ingest", "job_type": "asr",
                   "target_label": "v1.mp4", "target_id": "m-1", "status": "done",
                   "progress_pct": 100, "duration_seconds": 5.0}],
            filter_stage=None, filter_status=None, pipeline_stages=["ingest"],
        )
        assert "bg-green-900" in html

    def test_error_badge_red(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/jobs.html").render(
            page_title="Jobs",
            jobs=[{"job_id": "j-1", "stage": "ingest", "job_type": "asr",
                   "target_label": "v1.mp4", "target_id": "m-1", "status": "error",
                   "progress_pct": None, "duration_seconds": None}],
            filter_stage=None, filter_status=None, pipeline_stages=["ingest"],
        )
        assert "bg-red-900" in html

    def test_pending_badge_gray(self) -> None:
        env = _make_env()
        html = env.get_template("pipeline/jobs.html").render(
            page_title="Jobs",
            jobs=[{"job_id": "j-1", "stage": "ingest", "job_type": "asr",
                   "target_label": "v1.mp4", "target_id": "m-1", "status": "pending",
                   "progress_pct": None, "duration_seconds": None}],
            filter_stage=None, filter_status=None, pipeline_stages=["ingest"],
        )
        assert "bg-gray-700" in html


class TestMediaDetailBadge:
    """Verify media/detail.html uses the generic status_badge macro."""

    def test_source_contains_import(self) -> None:
        source = (TEMPLATES_DIR / "media" / "detail.html").read_text()
        assert _IMPORT_LINE in source

    def test_analyzed_badge_green(self) -> None:
        env = _make_env()
        env.globals["format_duration"] = lambda s: f"{s}s"
        html = env.get_template("media/detail.html").render(
            media={"id": "m-1", "file_path": "/v/test.mp4", "status": "analyzed",
                   "duration_seconds": 60, "resolution_w": 1920, "resolution_h": 1080,
                   "fps": 30, "codec": "h264", "file_size_bytes": 1000,
                   "created_at": "2024-01-01"},
        )
        assert "bg-green-900" in html

    def test_ingested_badge_blue(self) -> None:
        env = _make_env()
        env.globals["format_duration"] = lambda s: f"{s}s"
        html = env.get_template("media/detail.html").render(
            media={"id": "m-1", "file_path": "/v/test.mp4", "status": "ingested",
                   "duration_seconds": 60, "resolution_w": 1920, "resolution_h": 1080,
                   "fps": 30, "codec": "h264", "file_size_bytes": 1000,
                   "created_at": "2024-01-01"},
        )
        assert "bg-blue-900" in html


class TestMediaRowBadge:
    """Verify partials/media_row.html uses the generic status_badge macro."""

    def test_source_contains_import(self) -> None:
        source = (TEMPLATES_DIR / "partials" / "media_row.html").read_text()
        assert _IMPORT_LINE in source

    def test_analyzed_badge_green(self) -> None:
        env = _make_env()
        env.globals["format_duration"] = lambda s: f"{s}s"
        html = env.get_template("partials/media_row.html").render(
            item={"id": "m-1", "file_path": "/v/test.mp4", "status": "analyzed",
                  "duration_seconds": 60, "resolution_w": 1920, "resolution_h": 1080,
                  "created_at": "2024-01-01", "has_transcript": False,
                  "has_detections": False, "has_faces": False, "has_embeddings": False,
                  "has_audio_events": False, "has_captions": False},
        )
        assert "bg-green-900" in html

    def test_unknown_status_gray(self) -> None:
        env = _make_env()
        env.globals["format_duration"] = lambda s: f"{s}s"
        html = env.get_template("partials/media_row.html").render(
            item={"id": "m-1", "file_path": "/v/test.mp4", "status": "pending",
                  "duration_seconds": 60, "resolution_w": None, "resolution_h": None,
                  "created_at": None, "has_transcript": False,
                  "has_detections": False, "has_faces": False, "has_embeddings": False,
                  "has_audio_events": False, "has_captions": False},
        )
        assert "bg-gray-700" in html


class TestReviewUploadsBadge:
    """Verify review/uploads.html uses the generic status_badge macro."""

    def test_source_contains_import(self) -> None:
        source = (TEMPLATES_DIR / "review" / "uploads.html").read_text()
        assert _IMPORT_LINE in source

    def test_public_badge_green(self) -> None:
        env = _make_env()
        html = env.get_template("review/uploads.html").render(
            page_title="Uploads",
            uploads=[{"narrative_id": "n-1", "narrative_title": "T",
                      "youtube_video_id": "abc", "youtube_url": "https://y.com/abc",
                      "privacy_status": "public", "uploaded_at": "2024-01-01"}],
        )
        assert "bg-green-900" in html

    def test_unlisted_badge_yellow(self) -> None:
        env = _make_env()
        html = env.get_template("review/uploads.html").render(
            page_title="Uploads",
            uploads=[{"narrative_id": "n-1", "narrative_title": "T",
                      "youtube_video_id": "abc", "youtube_url": "https://y.com/abc",
                      "privacy_status": "unlisted", "uploaded_at": "2024-01-01"}],
        )
        assert "bg-yellow-900" in html


class TestReviewRendersBadge:
    """Verify review/renders.html uses the generic status_badge macro."""

    def test_source_contains_import(self) -> None:
        source = (TEMPLATES_DIR / "review" / "renders.html").read_text()
        assert _IMPORT_LINE in source

    def test_passes_true_badge_green(self) -> None:
        env = _make_env()
        html = env.get_template("review/renders.html").render(
            page_title="Render Review",
            narrative={
                "narrative_id": "n-1", "title": "T",
                "description": "D", "status": "approved",
            },
            edit_plan={"validation": {"passes": True}},
            has_render=False, scenes=[],
        )
        assert "bg-green-900" in html

    def test_passes_false_badge_red(self) -> None:
        env = _make_env()
        html = env.get_template("review/renders.html").render(
            page_title="Render Review",
            narrative={
                "narrative_id": "n-1", "title": "T",
                "description": "D", "status": "approved",
            },
            edit_plan={"validation": {"passes": False}},
            has_render=False, scenes=[],
        )
        assert "bg-red-900" in html


class TestReviewHubBadge:
    """Verify review/hub.html uses the generic status_badge macro."""

    def test_source_contains_import(self) -> None:
        source = (TEMPLATES_DIR / "review" / "hub.html").read_text()
        assert _IMPORT_LINE in source

    def test_waiting_badge_amber(self) -> None:
        env = _make_env()
        html = env.get_template("review/hub.html").render(
            waiting_gates=[{
                "stage": "render", "link": "/review/render",
                "pending_count": 2, "pending_label": "items",
            }],
        )
        assert "bg-amber-900" in html
