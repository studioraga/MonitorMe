from __future__ import annotations

from monitor_me.assistant import MonitorMeAssistant
from monitor_me.llm_client import FakeLLMClient

from tests.helpers import make_real_motion_capture


def test_assistant_answer_contains_required_real_motion_evidence_refs(tmp_path):
    db, result, event = make_real_motion_capture(tmp_path)
    assistant = MonitorMeAssistant(db)

    answer = assistant.ask("What motion events happened today?")

    assert "event_id=" in answer.answer
    assert "session_id=" in answer.answer
    assert "frame_id=" in answer.answer
    assert answer.evidence
    assert answer.evidence[0]["event_id"] == event["event_id"]
    assert answer.evidence[0]["session_id"] == result.session_id
    assert answer.evidence[0]["frame_id"] is not None
    assert answer.evidence[0]["label"] == "motion"
    assert answer.evidence[0]["policy_decision"]["decision"] == "allow"


def test_assistant_refuses_weapon_claim_without_normalized_evidence(tmp_path):
    db, _, _ = make_real_motion_capture(tmp_path)
    assistant = MonitorMeAssistant(db)

    result = assistant.ask("Was the person carrying a weapon?")

    assert "do not have local evidence" in result.answer.lower()
    assert "normalized local evidence" in result.answer.lower()


def test_fact_guard_rejects_unsafe_llm_summary(tmp_path):
    db, _, _ = make_real_motion_capture(tmp_path)
    assistant = MonitorMeAssistant(db, llm=FakeLLMClient("A person had a weapon and was suspicious."))

    result = assistant.ask("What motion events happened today?", use_llm=True)

    assert "weapon" not in result.answer.lower()
    assert "suspicious" not in result.answer.lower()
    assert "event_id=" in result.answer
