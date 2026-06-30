from __future__ import annotations

from tests.helpers import make_real_motion_capture


def test_real_capture_creates_normalized_motion_rows_without_fake_object_labels(tmp_path):
    db, result, event = make_real_motion_capture(tmp_path)

    motion = db.get_event(str(event["event_id"]))
    assert motion is not None
    assert motion["event_type"] == "motion_detected"
    assert motion["label"] == "motion"
    assert motion["session_id"] == result.session_id
    assert motion["source_node"] == "node1"
    assert motion["source_kind"] == "local_v4l2"
    assert motion["artifact_id"]
    assert motion["confidence"] and motion["confidence"] > 0

    object_rows = db.list_events(event_type="object_detected")
    assert object_rows == []
