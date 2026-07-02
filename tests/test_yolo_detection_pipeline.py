from __future__ import annotations

from tests.helpers import motion_frames

from monitor_me.db import MonitorMeDB
from monitor_me.local_capture import IterableFrameSource, LocalCameraCaptureRunner, LocalCaptureConfig
from monitor_me.yolo_onnx import ObjectDetection, canonical_label


class FakeYoloDetector:
    model_id = "yolo11n-coco-onnx"

    def detect(self, frame):
        return [
            ObjectDetection(
                label="person",
                raw_label="person",
                class_id=0,
                confidence=0.91,
                bbox=[0.1, 0.2, 0.4, 0.8],
                model_id=self.model_id,
                attrs={"test_detector": True},
            ),
            ObjectDetection(
                label="vehicle",
                raw_label="car",
                class_id=2,
                confidence=0.77,
                bbox=[0.45, 0.35, 0.9, 0.75],
                model_id=self.model_id,
                attrs={"test_detector": True},
            ),
        ]


def test_yolo_child_object_rows_are_normalized_after_parent_motion_event(tmp_path):
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
        detector_model_path=str(tmp_path / "models" / "yolo11n.onnx"),
    )

    result = LocalCameraCaptureRunner(
        db,
        config,
        frame_source=IterableFrameSource(motion_frames()),
        object_detector=FakeYoloDetector(),
    ).run()

    assert result.ok
    assert len(result.motion_event_ids) == 1
    assert len(result.object_event_ids) == 2
    assert result.detector["enabled"] is True
    assert result.detector["loaded"] is True
    assert result.detector["object_events"] == 2

    parent = db.get_event(result.motion_event_ids[0])
    assert parent is not None
    assert parent["event_type"] == "motion_detected"
    assert parent["artifact_id"]

    object_rows = db.list_events(event_type="object_detected", session_id=result.session_id, limit=10)
    labels = {row["label"] for row in object_rows}
    assert labels == {"person", "vehicle"}
    for row in object_rows:
        assert row["parent_event_id"] == parent["event_id"]
        assert row["frame_id"] == parent["frame_id"]
        assert row["model_id"] == "yolo11n-coco-onnx"
        assert row["artifact_id"] == parent["artifact_id"]
        assert row["source_node"] == "node1"
        assert row["source_kind"] == "local_v4l2"
        assert row["confidence"] > 0
        assert row["bbox"]
        assert row["attrs"]["detector"] == "yolo_onnx"
        assert row["attrs"]["keyframe_artifact_id"] == parent["artifact_id"]

    audits = db.recent_audit(session_id=result.session_id, limit=80)
    actions = {row["action"] for row in audits}
    assert "detector.loaded" in actions
    assert "detector.run" in actions
    assert "event.insert" in actions


def test_detector_enabled_but_missing_model_does_not_fabricate_objects(tmp_path):
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
        detector_model_path=str(tmp_path / "missing.onnx"),
    )

    result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames())).run()

    assert result.ok
    assert result.motion_event_ids
    assert result.object_event_ids == []
    assert result.detector["enabled"] is True
    assert result.detector["loaded"] is False
    assert result.detector["error"]
    assert db.list_events(event_type="object_detected") == []
    actions = {row["action"] for row in db.recent_audit(session_id=result.session_id, limit=20)}
    assert "detector.unavailable" in actions


def test_coco_vehicle_classes_are_canonicalized_for_cctv_queries():
    assert canonical_label("car") == "vehicle"
    assert canonical_label("truck") == "vehicle"
    assert canonical_label("person") == "person"


def test_assistant_union_query_returns_person_when_vehicle_absent(tmp_path):
    from monitor_me.assistant import MonitorMeAssistant

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

    class PersonOnlyDetector:
        model_id = "yolo11n-coco-onnx"

        def detect(self, frame):
            return [
                ObjectDetection(
                    label="person",
                    raw_label="person",
                    class_id=0,
                    confidence=0.88,
                    bbox=[0.2, 0.2, 0.7, 0.9],
                    model_id=self.model_id,
                )
            ]

    LocalCameraCaptureRunner(
        db,
        config,
        frame_source=IterableFrameSource(motion_frames()),
        object_detector=PersonOnlyDetector(),
    ).run()

    answer = MonitorMeAssistant(db).ask("What person and vehicle events happened today?")

    assert answer.evidence
    assert any(item["label"] == "person" for item in answer.evidence)
    assert "vehicle" in answer.answer.lower()
    assert "no local evidence" in answer.answer.lower()


def test_plus_query_keeps_strict_person_vehicle_correlation(tmp_path):
    from monitor_me.assistant import MonitorMeAssistant

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

    class PersonOnlyDetector:
        model_id = "yolo11n-coco-onnx"

        def detect(self, frame):
            return [ObjectDetection(label="person", raw_label="person", class_id=0, confidence=0.88, bbox=[0.2, 0.2, 0.7, 0.9], model_id=self.model_id)]

    LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames()), object_detector=PersonOnlyDetector()).run()

    answer = MonitorMeAssistant(db).ask("Which clips had person + vehicle?")

    assert answer.evidence == []
    assert "do not have local evidence" in answer.answer.lower()
