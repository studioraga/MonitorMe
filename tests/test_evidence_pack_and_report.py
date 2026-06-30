from __future__ import annotations

import json
from pathlib import Path

from monitor_me.evidence_pack import EvidencePackBuilder
from monitor_me.report_tools import IncidentReportBuilder

from tests.helpers import make_real_motion_capture


def test_evidence_pack_contains_real_event_session_policy_audit_artifacts(tmp_path):
    db, result, event = make_real_motion_capture(tmp_path)
    event_id = event["event_id"]

    manifest = EvidencePackBuilder(db, root=tmp_path / "packs").build_for_event(str(event_id))
    actual_pack_dir = tmp_path / "packs" / str(manifest["pack_id"])

    assert actual_pack_dir.is_dir()
    for name in [
        "event.json",
        "related_events.json",
        "capture_session.json",
        "artifacts.json",
        "model_metadata.json",
        "policy_decision.json",
        "audit.json",
        "assistant_summary.json",
        "manifest.json",
        "report.md",
    ]:
        assert (actual_pack_dir / name).is_file(), name

    event_json = json.loads((actual_pack_dir / "event.json").read_text())
    policy = json.loads((actual_pack_dir / "policy_decision.json").read_text())
    artifacts = json.loads((actual_pack_dir / "artifacts.json").read_text())
    audit = json.loads((actual_pack_dir / "audit.json").read_text())

    assert event_json["event_id"] == event_id
    assert event_json["session_id"] == result.session_id
    assert event_json["frame_id"] is not None
    assert event_json["model_id"] is None
    assert policy["decision"] == "allow"
    assert artifacts
    assert any(a["artifact_type"] == "keyframe" for a in artifacts)
    assert audit
    assert manifest["manifest_sha256"]


def test_incident_report_links_evidence_pack_ids_for_real_motion(tmp_path):
    db, _, event = make_real_motion_capture(tmp_path)
    event_id = str(event["event_id"])

    result = IncidentReportBuilder(db, reports_root=tmp_path / "reports", evidence_root=tmp_path / "packs").build(
        event_ids=[event_id], title="Gate motion event", severity="info"
    )

    assert result["report_id"].startswith("ir_")
    assert result["evidence_pack_ids"]
    report_text = Path(result["report_path"]).read_text()
    assert event_id in report_text
    assert "motion_detected" in report_text
