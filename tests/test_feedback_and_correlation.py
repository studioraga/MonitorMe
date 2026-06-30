from __future__ import annotations

from monitor_me.assistant import MonitorMeAssistant
from monitor_me.tracker_tools import TrackerTools

from tests.helpers import make_real_motion_capture


def test_false_positive_feedback_tracker_for_real_motion(tmp_path):
    db, _, event = make_real_motion_capture(tmp_path)
    event_id = str(event["event_id"])

    feedback_id = TrackerTools(db).mark_event(event_id, label="false_positive", reason="validation")
    rows = TrackerTools(db).false_positive_tracker()

    assert feedback_id.startswith("fb_")
    assert rows[0]["event_id"] == event_id
    assert rows[0]["label"] == "false_positive"


def test_person_vehicle_question_does_not_invent_labels_from_motion_only(tmp_path):
    db, _, _ = make_real_motion_capture(tmp_path)
    assistant = MonitorMeAssistant(db)

    result = assistant.ask("Which clips had person + vehicle?")

    assert "do not have local evidence" in result.answer.lower() or "no matching" in result.answer.lower()
    assert result.evidence == []
