from __future__ import annotations

import json
from pathlib import Path

from tests.helpers import motion_frames

from monitor_me.db import MonitorMeDB
from monitor_me.hash_utils import sha256_file
from monitor_me.local_capture import IterableFrameSource, LocalCameraCaptureRunner, LocalCaptureConfig
from monitor_me.yolo_onnx import ObjectDetection


class FakeOverlayDetector:
    model_id = "yolo11n-coco-onnx"

    def detect(self, frame):
        return [
            ObjectDetection(
                label="person",
                raw_label="person",
                class_id=0,
                confidence=0.91,
                bbox=[0.1, 0.2, 0.45, 0.85],
                model_id=self.model_id,
            ),
            ObjectDetection(
                label="vehicle",
                raw_label="car",
                class_id=2,
                confidence=0.77,
                bbox=[0.5, 0.35, 0.9, 0.75],
                model_id=self.model_id,
            ),
        ]


def test_step17e_writes_annotated_overlay_without_modifying_raw_keyframe(tmp_path):
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
        overlay_enabled=True,
    )

    result = LocalCameraCaptureRunner(
        db,
        config,
        frame_source=IterableFrameSource(motion_frames()),
        object_detector=FakeOverlayDetector(),
    ).run()

    assert result.ok
    assert len(result.object_event_ids) == 2
    assert len(result.overlay_artifact_ids) == 1
    assert len(result.overlay_paths) == 1

    raw_artifacts = db.list_artifacts(session_id=result.session_id)
    keyframes = [a for a in raw_artifacts if a["artifact_type"] == "keyframe"]
    overlays = [a for a in raw_artifacts if a["artifact_type"] == "annotated_keyframe"]
    assert len(keyframes) == 1
    assert len(overlays) == 1

    raw_path = Path(keyframes[0]["path"])
    overlay_path = Path(overlays[0]["path"])
    assert raw_path.exists()
    assert overlay_path.exists()
    assert raw_path != overlay_path
    assert sha256_file(raw_path) == keyframes[0]["sha256"]
    assert sha256_file(overlay_path) == overlays[0]["sha256"]
    assert overlay_path.stat().st_size > 0

    object_rows = db.list_events(event_type="object_detected", session_id=result.session_id, limit=10)
    for row in object_rows:
        # The object row remains linked to the raw keyframe artifact. The overlay
        # is a separate derived session artifact, not a replacement for evidence.
        assert row["artifact_id"] == keyframes[0]["artifact_id"]
        assert row["parent_event_id"] == result.motion_event_ids[0]
        assert row["model_id"] == "yolo11n-coco-onnx"

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    frame_record = manifest["frames"][0]
    assert frame_record["overlay_artifact_id"] == overlays[0]["artifact_id"]
    assert frame_record["overlay_path"] == str(overlay_path)
    assert len(frame_record["overlay_boxes"]) == 2
    first_box = frame_record["overlay_boxes"][0]
    assert first_box["event_id"] in result.object_event_ids
    assert first_box["parent_event_id"] == result.motion_event_ids[0]
    assert first_box["model_id"] == "yolo11n-coco-onnx"

    audits = db.recent_audit(session_id=result.session_id, limit=50)
    assert "overlay.create" in {row["action"] for row in audits}


def test_step17e_can_disable_overlays(tmp_path):
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
        overlay_enabled=False,
    )

    result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames()), object_detector=FakeOverlayDetector()).run()

    assert result.ok
    assert result.object_event_ids
    assert result.overlay_artifact_ids == []
    assert result.overlay_paths == []
    assert [a for a in db.list_artifacts(session_id=result.session_id) if a["artifact_type"] == "annotated_keyframe"] == []
