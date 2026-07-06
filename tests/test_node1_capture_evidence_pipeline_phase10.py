from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitor_me.db import MonitorMeDB
from monitor_me.local_capture import IterableFrameSource, LocalCameraCaptureRunner, LocalCaptureConfig
from tests.helpers import motion_frames


NATIVE_CPU = Path("native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab")


def _require_native_cpu():
    if not NATIVE_CPU.exists():
        pytest.skip("native CPU binary is not built")


def test_capture_run_evidence_pipeline_indexes_manifest_and_stores_facts(tmp_path: Path):
    _require_native_cpu()
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
        evidence_pipeline_enabled=True,
        evidence_pipeline_binary=str(NATIVE_CPU),
        evidence_pipeline_max_batch_bytes=1_600_000,
        evidence_pipeline_max_batch_clips=3,
        evidence_pipeline_key_moments=4,
        evidence_pipeline_min_key_gap_ms=1,
        evidence_pipeline_dedup_hamming_threshold=0,
        evidence_pipeline_fingerprint_cycle=6,
    )

    result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames())).run()

    assert result.ok is True
    assert result.motion_event_ids
    assert result.evidence_pipeline_event_ids
    assert len(result.evidence_pipeline_artifact_ids) == 2

    event = db.get_event(result.evidence_pipeline_event_ids[0])
    assert event is not None
    assert event["event_type"] == "evidence_pipeline_indexed"
    assert event["label"] == "facts_only_evidence_pipeline"
    attrs = event["attrs"]
    assert attrs["schema"] == "monitorme.node1_evidence_pipeline_profile.v0.1"
    assert attrs["ok"] is True
    assert attrs["capture_session_id"] == result.session_id
    assert attrs["capture_manifest_rows"] == len(result.motion_event_ids)
    assert attrs["fingerprint_count"] == len(result.motion_event_ids)
    assert attrs["key_moment_count"] >= 1
    assert attrs["safety"]["ok"] is True
    assert attrs["safety"]["violation_count"] == 0
    assert attrs["facts_only"] is True
    assert attrs["privacy"] == {"external_upload": False, "identity": False, "intent": False, "media_decode": False}

    csv_artifacts = db.list_artifacts(session_id=result.session_id, artifact_type="evidence_pipeline_manifest_csv")
    profile_artifacts = db.list_artifacts(session_id=result.session_id, artifact_type="evidence_pipeline_profile")
    assert len(csv_artifacts) == 1
    assert len(profile_artifacts) == 1
    assert Path(csv_artifacts[0]["path"]).exists()
    assert Path(profile_artifacts[0]["path"]).exists()

    profile = json.loads(Path(profile_artifacts[0]["path"]).read_text(encoding="utf-8"))
    assert profile["facts_only"] is True
    native = profile["result"]["evidence_pipeline"]
    assert native["schema"] == "node1_non_llm_evidence_pipeline.v0.1"
    assert native["facts_only"] is True
    assert native["safety"]["ok"] is True
    assert native["storage_batch"]["planned_read_bytes"] == native["storage_batch"]["total_manifest_bytes"]

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["evidence_pipeline"]["enabled"] is True
    assert manifest["evidence_pipeline"]["event_count"] == 1
    assert manifest["evidence_pipeline"]["artifact_count"] == 2
    assert manifest["evidence_pipeline"]["last_result"]["ok"] is True
    assert result.evidence_pipeline_event_ids[0] in manifest["evidence_pipeline_event_ids"]
    assert set(result.evidence_pipeline_artifact_ids).issubset(set(manifest["artifact_ids"]))


def test_capture_run_evidence_pipeline_disabled_by_default(tmp_path: Path):
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
    )
    result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames())).run()

    assert result.ok is True
    assert result.evidence_pipeline_event_ids == []
    assert result.evidence_pipeline_artifact_ids == []
    assert db.list_events(session_id=result.session_id, event_type="evidence_pipeline_indexed") == []
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["evidence_pipeline"]["enabled"] is False
    assert manifest["evidence_pipeline_event_ids"] == []
