from __future__ import annotations

from pathlib import Path

from monitor_me.assistant import MonitorMeAssistant
from monitor_me.assistant_summary import AssistantSummaryService
from monitor_me.capture_policy import evaluate_node1_event_policy
from monitor_me.db import MonitorMeDB
from monitor_me.event_contract import build_event_contract
from monitor_me.local_capture import IterableFrameSource, LocalCameraCaptureRunner, LocalCaptureConfig
from monitor_me.report_tools import IncidentReportBuilder
from monitor_me.yolo_onnx import ObjectDetection

from tests.helpers import motion_frames


class PersonGuitarDetector:
    model_id = "test-yolo-custom"

    def detect(self, frame):
        return [
            ObjectDetection(label="person", raw_label="person", class_id=0, confidence=0.91, bbox=[0.1, 0.1, 0.6, 0.9], model_id=self.model_id),
            ObjectDetection(label="guitar", raw_label="guitar", class_id=999, confidence=0.82, bbox=[0.35, 0.45, 0.75, 0.85], model_id=self.model_id),
        ]


class PersonOnlyDetector:
    model_id = "test-yolo-custom"

    def detect(self, frame):
        return [ObjectDetection(label="person", raw_label="person", class_id=0, confidence=0.88, bbox=[0.2, 0.2, 0.7, 0.9], model_id=self.model_id)]


def _capture_with_detector(tmp_path: Path, detector):
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
        detector_model_id=detector.model_id,
        overlay_enabled=True,
    )
    result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames()), object_detector=detector).run()
    assert result.ok
    return db, result


def test_capture_auto_creates_event_contracts_policy_and_summaries(tmp_path):
    db, result = _capture_with_detector(tmp_path, PersonGuitarDetector())

    assert result.motion_event_ids
    assert len(result.object_event_ids) == 2
    assert len(result.assistant_summary_ids) >= 3
    assert len(result.event_contract_ids) >= 3

    parent_event_id = result.motion_event_ids[0]
    contract_row = db.latest_event_contract(parent_event_id)
    assert contract_row is not None
    contract = contract_row["contract"]
    assert contract["schema_version"] == "1.0"
    assert contract["source_node"] == "node1"
    assert contract["camera_id"] == "c922_node1_gate"
    labels = {item["class_name"] for item in contract["detections"]}
    assert labels == {"person", "guitar"}

    policy = contract_row["policy_decision"]
    assert policy["action"] == "request_capture_review"
    assert policy["severity_label"] == "review"
    assert "person confidence" in policy["reason"]

    summaries = db.list_summaries(session_id=result.session_id, limit=10)
    assert len(summaries) >= 3
    assert any("person=1" in row["summary_text"] and "guitar=1" in row["summary_text"] for row in summaries)
    assert all("weapon" not in row["summary_text"].lower() for row in summaries)
    assert all("identity" not in row["summary_text"].lower() for row in summaries)


def test_event_contract_builder_and_policy_are_deterministic(tmp_path):
    db, result = _capture_with_detector(tmp_path, PersonGuitarDetector())
    contract = build_event_contract(db, result.motion_event_ids[0])
    policy = evaluate_node1_event_policy(contract).as_dict()

    assert contract["contract_type"] == "monitorme.node1_ai_camera_event"
    assert contract["detector"]["role"] == "fast_visual_facts_only"
    assert len(contract["detections"]) == 2
    assert policy["decision"] == "allow"
    assert policy["action"] == "request_capture_review"
    assert policy["duration_sec"] == 90


def test_assistant_can_answer_person_guitar_only_when_rows_exist(tmp_path):
    db, _ = _capture_with_detector(tmp_path, PersonGuitarDetector())
    answer = MonitorMeAssistant(db).ask("What person and guitar events happened today?")

    assert answer.evidence
    labels = {item["label"] for item in answer.evidence}
    assert {"person", "guitar"}.issubset(labels)
    assert "event_id=" in answer.answer
    assert "guitar" in answer.answer.lower()


def test_assistant_does_not_invent_guitar_when_absent(tmp_path):
    db, _ = _capture_with_detector(tmp_path, PersonOnlyDetector())
    answer = MonitorMeAssistant(db).ask("What person and guitar events happened today?")

    assert answer.evidence
    assert all(item["label"] != "guitar" for item in answer.evidence)
    assert "no local evidence found for requested label(s): guitar" in answer.answer.lower()


def test_incident_report_includes_assistant_summary_section(tmp_path):
    db, result = _capture_with_detector(tmp_path, PersonGuitarDetector())
    event_id = result.object_event_ids[0]
    report = IncidentReportBuilder(db, reports_root=tmp_path / "reports", evidence_root=tmp_path / "packs").build(
        event_ids=[event_id], title="Node1 person guitar event", severity="review"
    )
    text = Path(report["report_path"]).read_text()
    assert "## Assistant summaries" in text
    assert "summary_id=" in text
    assert event_id in text
