from __future__ import annotations

from pathlib import Path

import pytest

from monitor_me.db import MonitorMeDB
from monitor_me.local_capture import IterableFrameSource, LocalCameraCaptureRunner, LocalCaptureConfig
from monitor_me.short_clip_vlm import ShortClipVLMExperimentService
from monitor_me.smolvlm2_client import (
    SMOLVLM2_SCHEMA_VERSION,
    SmolVLM2Config,
    SmolVLM2OpenAIClient,
    build_smolvlm2_short_clip_schema,
    validate_smolvlm2_short_clip_json,
)
from monitor_me.yolo_onnx import ObjectDetection
from tests.helpers import motion_frames


class PersonOnlyDetector:
    model_id = "yolo11n-coco-onnx"

    def detect(self, frame):
        return [ObjectDetection(label="person", raw_label="person", class_id=0, confidence=0.88, bbox=[0.2, 0.2, 0.7, 0.9], model_id=self.model_id)]


class GoodSmolVLM2:
    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def analyze_clip(self, *, clip_manifest_path, event, related_events, clip_artifact, frame_artifacts):
        assert Path(clip_manifest_path).exists()
        assert frame_artifacts
        return {
            "schema_version": SMOLVLM2_SCHEMA_VERSION,
            "event_id": event["event_id"],
            "artifact_id": clip_artifact["artifact_id"],
            "visible_scene": "indoor",
            "person_like_presence": "visible",
            "vehicle_like_presence": "not_visible",
            "motion_claim": "single_frame_only_no_motion_claim",
            "safe_observation": "single frame reviewed",
            "unsupported_claims": [],
        }


class BadSmolVLM2:
    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

    def analyze_clip(self, *, clip_manifest_path, event, related_events, clip_artifact, frame_artifacts):
        return {
            "schema_version": SMOLVLM2_SCHEMA_VERSION,
            "event_id": event["event_id"],
            "artifact_id": clip_artifact["artifact_id"],
            "visible_scene": "indoor",
            "person_like_presence": "visible",
            "vehicle_like_presence": "not_visible",
            "motion_claim": "single_frame_only_no_motion_claim",
            "safe_observation": "single frame reviewed",
            "unsupported_claims": ["face recognition"],
        }


def _capture_with_smolvlm2(tmp_path: Path, vlm) -> tuple[MonitorMeDB, object]:
    db = MonitorMeDB(tmp_path / "monitorme.db")
    config = LocalCaptureConfig(
        camera_id="c922_node1_gate",
        device="/dev/video0",
        width=160,
        height=120,
        fps=30,
        duration_sec=5.0,
        max_frames=3,
        motion_threshold=1.0,
        min_event_gap_sec=0.0,
        data_root=str(tmp_path / "data"),
        detector_enabled=True,
        detector_model_id="yolo11n-coco-onnx",
        smolvlm2_enabled=True,
        smolvlm2_model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        smolvlm2_clip_frame_count=3,
    )
    result = LocalCameraCaptureRunner(
        db,
        config,
        frame_source=IterableFrameSource(motion_frames()),
        object_detector=PersonOnlyDetector(),
        short_clip_vlm_client=vlm,
    ).run()
    assert result.ok
    return db, result


def test_smolvlm2_short_clip_experiment_runs_after_trigger_only(tmp_path):
    db, result = _capture_with_smolvlm2(tmp_path, GoodSmolVLM2())

    assert result.motion_event_ids
    assert result.object_event_ids
    assert result.smolvlm2_experiment_ids

    rows = db.list_smolvlm2_clip_experiments(session_id=result.session_id, limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["event_id"] == result.motion_event_ids[0]
    assert row["clip_artifact_id"]
    assert row["experiment"]["schema_version"] == SMOLVLM2_SCHEMA_VERSION
    assert row["experiment"]["event_id"] == result.motion_event_ids[0]
    assert row["experiment"]["artifact_id"] == row["clip_artifact_id"]
    assert row["experiment"]["motion_claim"] == "single_frame_only_no_motion_claim"
    assert row["experiment"]["unsupported_claims"] == []
    assert "identity" not in str(row["experiment"]).lower()

    clip_artifacts = db.list_artifacts(session_id=result.session_id, artifact_type="short_clip_manifest")
    assert clip_artifacts
    assert Path(clip_artifacts[0]["path"]).exists()
    manifest = Path(result.manifest_path).read_text()
    assert "smolvlm2_experiment_ids" in manifest
    assert result.smolvlm2_experiment_ids[0] in manifest


def test_invalid_smolvlm2_experiment_is_stored_as_failed_and_not_promoted(tmp_path):
    db, result = _capture_with_smolvlm2(tmp_path, BadSmolVLM2())

    assert result.motion_event_ids
    assert result.smolvlm2_experiment_ids
    rows = db.list_smolvlm2_clip_experiments(session_id=result.session_id, status="failed", limit=10)
    assert len(rows) == 1
    assert rows[0]["experiment"] == {}
    assert rows[0]["error"]
    assert "unsupported_claims" in rows[0]["error"].lower() or "unsupported claim" in rows[0]["error"].lower()


def test_smolvlm2_disabled_by_default_does_not_create_clip_or_experiment(tmp_path):
    db = MonitorMeDB(tmp_path / "monitorme.db")
    config = LocalCaptureConfig(
        camera_id="c922_node1_gate",
        device="/dev/video0",
        width=160,
        height=120,
        fps=30,
        duration_sec=5.0,
        max_frames=3,
        motion_threshold=1.0,
        min_event_gap_sec=0.0,
        data_root=str(tmp_path / "data"),
        detector_enabled=True,
    )
    result = LocalCameraCaptureRunner(
        db,
        config,
        frame_source=IterableFrameSource(motion_frames()),
        object_detector=PersonOnlyDetector(),
        short_clip_vlm_client=GoodSmolVLM2(),
    ).run()

    assert result.ok
    assert result.smolvlm2_experiment_ids == []
    assert db.list_smolvlm2_clip_experiments(session_id=result.session_id) == []
    assert db.list_artifacts(session_id=result.session_id, artifact_type="short_clip_manifest") == []


def test_smolvlm2_validator_rejects_identity_intent_unknown_ids_and_bad_schema():
    event = {"event_id": "evt_1", "camera_id": "cam", "artifact_id": "art_1", "frame_id": 2}
    clip_artifact = {"artifact_id": "clip_1", "path": "clip_manifest.json"}
    manifest = {"frames": [{"frame_id": 2, "artifact_id": "frame_1", "path": "frame.jpg"}]}
    with pytest.raises(ValueError):
        validate_smolvlm2_short_clip_json(
            {
                "schema_version": SMOLVLM2_SCHEMA_VERSION,
                "event_id": "evt_fake",
                "artifact_id": "clip_1",
                "visible_scene": "indoor",
                "person_like_presence": "visible",
                "vehicle_like_presence": "not_visible",
                "motion_claim": "single_frame_only_no_motion_claim",
                "safe_observation": "single frame reviewed",
                "unsupported_claims": [],
            },
            event=event,
            related_events=[],
            clip_artifact=clip_artifact,
            frame_artifacts=[{"artifact_id": "frame_1"}],
            clip_manifest=manifest,
        )
    with pytest.raises(ValueError):
        validate_smolvlm2_short_clip_json(
            {
                "schema_version": SMOLVLM2_SCHEMA_VERSION,
                "event_id": "evt_1",
                "artifact_id": "clip_1",
                "visible_scene": "indoor",
                "person_like_presence": "visible",
                "vehicle_like_presence": "not_visible",
                "motion_claim": "single_frame_only_no_motion_claim",
                "safe_observation": "single frame reviewed",
                "unsupported_claims": ["intent"],
            },
            event=event,
            related_events=[],
            clip_artifact=clip_artifact,
            frame_artifacts=[{"artifact_id": "frame_1"}],
            clip_manifest=manifest,
        )


def test_smolvlm2_structured_output_schema_const_binds_event_and_artifact():
    schema = build_smolvlm2_short_clip_schema(
        event={"event_id": "evt_1"},
        clip_artifact={"artifact_id": "art_1"},
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["event_id"]["const"] == "evt_1"
    assert schema["properties"]["artifact_id"]["const"] == "art_1"
    assert schema["properties"]["motion_claim"]["enum"] == ["single_frame_only_no_motion_claim"]
    assert schema["properties"]["unsupported_claims"]["maxItems"] == 0


def test_smolvlm2_config_defaults_are_live_smoke_test_friendly(monkeypatch):
    monkeypatch.delenv("MONITORME_SMOLVLM2_MAX_FRAMES", raising=False)
    monkeypatch.delenv("MONITORME_SMOLVLM2_MAX_TOKENS", raising=False)
    cfg = SmolVLM2Config.from_env()
    assert cfg.max_frames == 1
    assert cfg.max_tokens == 300
    assert cfg.temperature == 0.0


def test_smolvlm2_client_rejects_remote_endpoint_by_default():
    with pytest.raises(ValueError, match="loopback/local"):
        SmolVLM2OpenAIClient(SmolVLM2Config(enabled=True, base_url="https://example.com/v1", allow_remote=False))


def test_manual_smolvlm2_experiment_service_can_run_on_existing_event(tmp_path):
    db, result = _capture_with_smolvlm2(tmp_path, GoodSmolVLM2())
    manual = ShortClipVLMExperimentService(db, vlm=GoodSmolVLM2()).analyze_event(result.motion_event_ids[0])
    assert manual["status"] == "completed"
    assert manual["experiment_id"]
