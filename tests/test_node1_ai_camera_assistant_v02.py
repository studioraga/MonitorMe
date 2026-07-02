from __future__ import annotations

import pytest

from monitor_me.assistant_summary import AssistantSummaryService
from monitor_me.llm_client import parse_json_object, validate_gemma_summary_json
from tests.test_node1_ai_camera_assistant_v01 import PersonGuitarDetector, _capture_with_detector


class GoodGemmaSummaryClient:
    model_id = "google/gemma-3-1b-it"

    def summarize_event_contract(self, *, event_contract, policy_decision, evidence):
        event_ids = [event_contract["event_id"]]
        return {
            "operator_summary": f"Node1 camera {event_contract['camera_id']} recorded local person and guitar evidence for event_id={event_contract['event_id']}.",
            "event_reason": f"The deterministic policy action is {policy_decision['action']} based on stored YOLO facts.",
            "dashboard_tag": "person_guitar_review",
            "recommended_next_step": f"Review the stored evidence using policy_action={policy_decision['action']}",
            "severity_label": "review",
            "cited_event_ids": event_ids,
            "_validated": True,
        }


class BadGemmaSummaryClient:
    model_id = "google/gemma-3-1b-it"

    def summarize_event_contract(self, *, event_contract, policy_decision, evidence):
        return {
            "operator_summary": "A suspicious armed person was recognized by face.",
            "event_reason": "I inferred intent from the video.",
            "dashboard_tag": "danger",
            "recommended_next_step": "Call police",
            "severity_label": "urgent",
            "cited_event_ids": [event_contract["event_id"]],
            "_validated": True,
        }


def test_parse_json_object_accepts_fenced_json():
    parsed = parse_json_object('```json\n{"operator_summary":"ok"}\n```')
    assert parsed["operator_summary"] == "ok"


def test_validate_gemma_summary_rejects_unknown_event_id():
    event_contract = {"event_id": "evt_known", "camera_id": "cam", "detections": [{"class_name": "person"}]}
    policy = {"action": "request_capture_review"}
    evidence = [{"event_id": "evt_known", "camera_id": "cam", "label": "person"}]
    with pytest.raises(ValueError, match="unknown event IDs"):
        validate_gemma_summary_json(
            {
                "operator_summary": "Person evidence for event_id=evt_unknown.",
                "event_reason": "Stored evidence only.",
                "dashboard_tag": "person_review",
                "recommended_next_step": "Review policy_action=request_capture_review",
                "severity_label": "review",
                "cited_event_ids": ["evt_unknown"],
            },
            event_contract=event_contract,
            policy_decision=policy,
            evidence=evidence,
        )


def test_gemma_summary_is_used_when_json_is_valid(tmp_path):
    db, result = _capture_with_detector(tmp_path, PersonGuitarDetector())
    service = AssistantSummaryService(db, llm=GoodGemmaSummaryClient())
    out = service.summarize_event(result.motion_event_ids[0])

    assert out["summary_source"] == "gemma_max"
    assert out["llm_model_id"] == "google/gemma-3-1b-it"
    assert out["gemma_summary_json"]["severity_label"] == "review"
    assert "policy_action=request_capture_review" in out["summary_text"]
    stored_rows = db.list_summaries(event_id=result.motion_event_ids[0], limit=10)
    stored = [row for row in stored_rows if row["summary_id"] == out["summary_id"]][0]
    assert stored["facts"]["summary_source"] == "gemma_max"
    assert stored["facts"]["gemma_summary_json"]["dashboard_tag"] == "person_guitar_review"


def test_invalid_gemma_summary_falls_back_to_deterministic(tmp_path):
    db, result = _capture_with_detector(tmp_path, PersonGuitarDetector())
    service = AssistantSummaryService(db, llm=BadGemmaSummaryClient())
    out = service.summarize_event(result.motion_event_ids[0])

    assert out["summary_source"] == "deterministic"
    assert out["fallback_reason"]
    assert "suspicious" not in out["summary_text"].lower()
    assert "weapon" not in out["summary_text"].lower()
    stored_rows = db.list_summaries(event_id=result.motion_event_ids[0], limit=10)
    stored = [row for row in stored_rows if row["summary_id"] == out["summary_id"]][0]
    assert stored["facts"]["summary_source"] == "deterministic"
    assert stored["facts"]["fallback_reason"]
