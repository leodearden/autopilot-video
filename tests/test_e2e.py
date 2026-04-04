"""End-to-end integration test for the full pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Pre-import submodules of packages with empty __init__.py so that
# @patch("autopilot.<pkg>.<sub>") can find the attribute.
import autopilot.analyze.asr  # noqa: F401
import autopilot.analyze.audio_events  # noqa: F401
import autopilot.analyze.embeddings  # noqa: F401
import autopilot.analyze.faces  # noqa: F401
import autopilot.analyze.gpu_scheduler  # noqa: F401
import autopilot.analyze.objects  # noqa: F401
import autopilot.analyze.scenes  # noqa: F401
import autopilot.plan.edl  # noqa: F401
import autopilot.plan.otio_export  # noqa: F401
import autopilot.plan.script  # noqa: F401
import autopilot.plan.validator  # noqa: F401
import autopilot.source.resolve  # noqa: F401


class TestEndToEnd:
    """Comprehensive e2e test using in-memory DB and mocked external deps."""

    @pytest.fixture
    def fake_media_file(self):
        """Create a fake MediaFile-like object."""
        mf = MagicMock()
        mf.file_path = Path("/fake/input/video1.mp4")
        mf.sha256_prefix = "abc123"
        mf.codec = "h264"
        mf.resolution_w = 1920
        mf.resolution_h = 1080
        mf.fps = 30.0
        mf.duration_seconds = 120.0
        mf.created_at = "2026-01-01T00:00:00"
        mf.gps_lat = 37.7749
        mf.gps_lon = -122.4194
        mf.audio_channels = 2
        mf.metadata_json = "{}"
        return mf

    @pytest.fixture
    def fake_narrative(self):
        """Create a fake Narrative-like object."""
        narr = MagicMock()
        narr.narrative_id = "narr-1"
        narr.title = "Test Narrative"
        narr.description = "A test narrative"
        narr.proposed_duration_seconds = 300
        narr.activity_cluster_ids = ["c1"]
        narr.arc = "rising"
        narr.emotional_journey = "calm to exciting"
        narr.reasoning = "test"
        narr.status = "proposed"
        return narr

    @patch("autopilot.upload.thumbnail")
    @patch("autopilot.upload.youtube")
    @patch("autopilot.render.validate")
    @patch("autopilot.render.router")
    @patch("autopilot.source.resolve")
    @patch("autopilot.plan.otio_export")
    @patch("autopilot.plan.validator")
    @patch("autopilot.plan.edl")
    @patch("autopilot.plan.script")
    @patch("autopilot.organize.narratives")
    @patch("autopilot.organize.classify")
    @patch("autopilot.organize.cluster")
    @patch("autopilot.analyze.faces")
    @patch("autopilot.analyze.audio_events")
    @patch("autopilot.analyze.embeddings")
    @patch("autopilot.analyze.objects")
    @patch("autopilot.analyze.scenes")
    @patch("autopilot.analyze.asr")
    @patch("autopilot.analyze.gpu_scheduler.GPUScheduler")
    @patch("autopilot.ingest.dedup")
    @patch("autopilot.ingest.normalizer")
    @patch("autopilot.ingest.scanner")
    def test_full_pipeline_smoke(
        self,
        mock_scanner,
        mock_normalizer,
        mock_dedup,
        mock_gpu_cls,
        mock_asr,
        mock_scenes,
        mock_objects,
        mock_embeddings,
        mock_audio_events,
        mock_faces,
        mock_cluster,
        mock_classify,
        mock_narratives,
        mock_script,
        mock_edl,
        mock_validator,
        mock_otio,
        mock_resolve,
        mock_router,
        mock_render_validate,
        mock_youtube,
        mock_thumbnail,
        fake_media_file,
        fake_narrative,
        catalog_db,
        minimal_config,
    ):
        """Full pipeline runs e2e with mocked external deps, DB state correct."""
        from autopilot.orchestrator import PipelineOrchestrator, StageStatus

        # ---- INGEST mocks ----
        mock_scanner.scan_directory.return_value = [fake_media_file]
        mock_normalizer.normalize_audio.return_value = Path("/fake/norm.wav")
        mock_dedup.mark_duplicates.return_value = 0

        # ---- ANALYZE mocks ----
        mock_gpu_cls.return_value = MagicMock()

        # ---- CLASSIFY mocks ----
        mock_cluster.cluster_activities.return_value = None
        mock_classify.label_activities.return_value = None

        # ---- NARRATE mocks ----
        mock_narratives.build_master_storyboard.return_value = "storyboard"
        mock_narratives.propose_narratives.return_value = [fake_narrative]
        mock_narratives.format_for_review.return_value = "review text"

        # ---- SCRIPT mock ----
        mock_script.generate_script.return_value = {"scenes": [{"id": "s1"}]}

        # ---- EDL mocks ----
        edl_result = {
            "timeline": [{"clip": "c1"}],
            "target_duration_seconds": 300,
        }

        def _generate_edl_side_effect(nid, db, llm_config):
            # Simulate generate_edl persisting edl_json (satisfies assertion guard)
            db.upsert_edit_plan(nid, json.dumps(edl_result))
            return edl_result

        mock_edl.generate_edl.side_effect = _generate_edl_side_effect
        mock_otio.export_otio.return_value = Path("/fake/timeline.otio")

        # ---- SOURCE mocks ----
        mock_resolve.resolve_edl_assets.return_value = {
            "edl": {},
            "unresolved": [],
        }

        # ---- RENDER mocks ----
        # Create render output so UPLOAD stage can find it
        render_dir = minimal_config.output_dir / "renders" / "narr-1"
        render_dir.mkdir(parents=True)
        video_file = render_dir / "output.mp4"
        video_file.write_text("fake video")
        mock_router.route_and_render.return_value = video_file

        render_report = MagicMock()
        render_report.passed = True
        render_report.issues = []
        mock_render_validate.validate_render.return_value = render_report

        # ---- UPLOAD mocks ----
        mock_youtube.upload_video.return_value = "https://youtu.be/test123"
        mock_thumbnail.extract_best_thumbnail.return_value = Path("/thumb.jpg")

        # ---- Seed DB state that stages expect ----
        # INGEST will insert media into DB via _run_ingest.
        # ANALYZE reads from list_all_media, which will see the
        # media inserted by INGEST above.

        # NARRATE proposes narratives which get approved by auto-approve.
        # We need to seed approved narrative for post-NARRATE stages
        # (SCRIPT, EDL, SOURCE, RENDER, UPLOAD all query for approved narratives).
        catalog_db.insert_narrative(
            "narr-1",
            title="Test Narrative",
            description="A test narrative",
            proposed_duration_seconds=300,
            activity_cluster_ids_json='["c1"]',
            arc_notes="rising",
            emotional_journey="calm to exciting",
            status="approved",
        )

        # SCRIPT stage expects a script exists for EDL generation
        catalog_db.upsert_narrative_script("narr-1", json.dumps({"scenes": [{"id": "s1"}]}))

        # EDL stage writes edit plan which SOURCE and RENDER read
        catalog_db.upsert_edit_plan(
            "narr-1",
            json.dumps({"timeline": [{"clip": "c1"}]}),
            otio_path="/fake/timeline.otio",
            validation_json=json.dumps({"passed": True}),
        )

        # ---- Run the full pipeline ----
        orch = PipelineOrchestrator(budget_seconds=3600, force=True)
        results = orch.run(config=minimal_config, db=catalog_db)

        # ---- Verify all stages completed ----
        for stage_name, result in results.items():
            assert result.status == StageStatus.DONE, (
                f"{stage_name} status was {result.status}, error: {result.error_message}"
            )

        # ---- Verify key module calls ----
        mock_scanner.scan_directory.assert_called_once()
        mock_dedup.mark_duplicates.assert_called_once()
        mock_gpu_cls.assert_called_once()
        mock_asr.transcribe_media.assert_called_once()
        mock_scenes.detect_shots.assert_called_once()
        mock_objects.detect_objects.assert_called_once()
        mock_faces.detect_faces.assert_called_once()
        mock_faces.cluster_faces.assert_called_once()
        mock_embeddings.compute_embeddings.assert_called_once()
        mock_audio_events.classify_audio_events.assert_called_once()
        mock_cluster.cluster_activities.assert_called_once()
        mock_classify.label_activities.assert_called_once()
        mock_narratives.build_master_storyboard.assert_called_once()
        mock_narratives.propose_narratives.assert_called_once()
        mock_script.generate_script.assert_called_once()
        mock_edl.generate_edl.assert_called_once()
        mock_otio.export_otio.assert_called_once()
        mock_resolve.resolve_edl_assets.assert_called_once()
        mock_router.route_and_render.assert_called_once()
        mock_render_validate.validate_render.assert_called_once()
        mock_youtube.upload_video.assert_called_once()
        mock_thumbnail.extract_best_thumbnail.assert_called_once()

        # ---- Verify DB state ----
        media = catalog_db.get_media("abc123")
        assert media is not None

        narr = catalog_db.get_narrative("narr-1")
        assert narr is not None

        script_data = catalog_db.get_narrative_script("narr-1")
        assert script_data is not None

        edit_plan = catalog_db.get_edit_plan("narr-1")
        assert edit_plan is not None
