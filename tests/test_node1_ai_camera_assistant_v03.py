from __future__ import annotations

from pathlib import Path

import pytest

from monitor_me.db import MonitorMeDB
from monitor_me.keyframe_vlm import KeyframeVLMAnalysisService
from monitor_me.local_capture import IterableFrameSource, LocalCameraCaptureRunner, LocalCaptureConfig
from monitor_me.vlm_client import QwenVLMConfig, QwenVLMOpenAIClient, validate_qwen_keyframe_json
from monitor_me.yolo_onnx import ObjectDetection
from tests.helpers import motion_frames


class PersonOnlyDetector:
    model_id = "yolo11n-coco-onnx"

    def detect(self, frame):
        return [ObjectDetection(label="person", raw_label="person", class_id=0, confidence=0.88, bbox=[0.2, 0.2, 0.7, 0.9], model_id=self.model_id)]


class GoodQwenVLM:
    model_id = "Qwen/Qwen3-VL-2B-Instruct"

    def analyze_keyframe(self, *, image_path, event, related_events, artifact):
        assert Path(image_path).exists()
        return {
            "schema_version": "monitorme.qwen_vlm_keyframe.v0.3",
            "scene_summary": "A local keyframe shows a person-shaped figure in the camera scene.",
            "visible_entities": [
                {
                    "label": "person",
                    "description": "A person-like visual region is visible in the stored keyframe.",
                    "confidence_hint": "medium",
                    "location_hint": "center area",
                }
            ],
            "text_visible": "unknown",
            "image_quality": "usable test keyframe",
            "safety_notes": "Companion VLM observation only; YOLO and policy remain source of truth.",
            "cited_event_ids": [event["event_id"]],
            "cited_artifact_ids": [artifact["artifact_id"]],
            "limitations": ["Single keyframe only; no personal recognition or behavior inference."],
        }


class BadQwenVLM:
    model_id = "Qwen/Qwen3-VL-2B-Instruct"

    def analyze_keyframe(self, *, image_path, event, related_events, artifact):
        return {
            "schema_version": "monitorme.qwen_vlm_keyframe.v0.3",
            "scene_summary": "A suspicious person was recognized by face.",
            "visible_entities": [],
            "text_visible": "unknown",
            "image_quality": "usable",
            "safety_notes": "threat detected",
            "cited_event_ids": ["evt_fake"],
            "cited_artifact_ids": [artifact["artifact_id"]],
            "limitations": [],
        }


def _capture_with_vlm(tmp_path: Path, vlm) -> tuple[MonitorMeDB, object]:
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
        vlm_enabled=True,
        vlm_model_id="Qwen/Qwen3-VL-2B-Instruct",
    )
    result = LocalCameraCaptureRunner(
        db,
        config,
        frame_source=IterableFrameSource(motion_frames()),
        object_detector=PersonOnlyDetector(),
        keyframe_vlm_client=vlm,
    ).run()
    assert result.ok
    return db, result


def test_qwen_vlm_keyframe_analysis_runs_after_trigger_only(tmp_path):
    db, result = _capture_with_vlm(tmp_path, GoodQwenVLM())

    assert result.motion_event_ids
    assert result.object_event_ids
    assert result.vlm_analysis_ids

    rows = db.list_vlm_analyses(session_id=result.session_id, limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "completed"
    assert row["event_id"] == result.motion_event_ids[0]
    assert row["artifact_id"]
    assert row["analysis"]["schema_version"] == "monitorme.qwen_vlm_keyframe.v0.3"
    assert row["analysis"]["cited_event_ids"] == [result.motion_event_ids[0]]
    assert "identity" not in row["analysis"]["scene_summary"].lower()

    manifest = Path(result.manifest_path).read_text()
    assert "vlm_analysis_ids" in manifest
    assert result.vlm_analysis_ids[0] in manifest


def test_invalid_qwen_vlm_analysis_is_stored_as_failed_and_not_promoted(tmp_path):
    db, result = _capture_with_vlm(tmp_path, BadQwenVLM())

    assert result.motion_event_ids
    assert result.vlm_analysis_ids
    rows = db.list_vlm_analyses(session_id=result.session_id, status="failed", limit=10)
    assert len(rows) == 1
    assert rows[0]["analysis"] == {}
    assert rows[0]["error"]
    assert "unknown event" in rows[0]["error"].lower() or "unsupported claim" in rows[0]["error"].lower()


def test_qwen_vlm_disabled_by_default_does_not_create_analysis(tmp_path):
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
        keyframe_vlm_client=GoodQwenVLM(),
    ).run()

    assert result.ok
    assert result.vlm_analysis_ids == []
    assert db.list_vlm_analyses(session_id=result.session_id) == []


def test_qwen_vlm_validator_rejects_identity_intent_and_unknown_ids():
    event = {"event_id": "evt_1", "camera_id": "cam", "artifact_id": "art_1"}
    artifact = {"artifact_id": "art_1", "path": "frame.jpg"}
    with pytest.raises(ValueError):
        validate_qwen_keyframe_json(
            {
                "schema_version": "monitorme.qwen_vlm_keyframe.v0.3",
                "scene_summary": "The person is suspicious and was recognized.",
                "visible_entities": [],
                "text_visible": "unknown",
                "image_quality": "usable",
                "safety_notes": "unknown",
                "cited_event_ids": ["evt_fake"],
                "cited_artifact_ids": ["art_1"],
                "limitations": [],
            },
            event=event,
            related_events=[],
            artifact=artifact,
        )


def test_qwen_vlm_client_rejects_remote_endpoint_by_default():
    with pytest.raises(ValueError, match="loopback/local"):
        QwenVLMOpenAIClient(QwenVLMConfig(enabled=True, base_url="https://example.com/v1", allow_remote=False))


def test_manual_vlm_analysis_endpoint_service_can_run_on_existing_event(tmp_path):
    db, result = _capture_with_vlm(tmp_path, GoodQwenVLM())
    manual = KeyframeVLMAnalysisService(db, vlm=GoodQwenVLM()).analyze_event(result.motion_event_ids[0])
    assert manual["status"] == "completed"
    assert manual["analysis_id"]
