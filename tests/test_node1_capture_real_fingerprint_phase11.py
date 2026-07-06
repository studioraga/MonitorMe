from __future__ import annotations

import csv
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


def test_capture_run_routes_decoded_keyframes_into_real_media_fingerprints(tmp_path: Path):
    _require_native_cpu()
    db = MonitorMeDB(tmp_path / "monitorme.db")
    config = LocalCaptureConfig(
        camera_id="c922_node1_gate",
        device="/dev/video0",
        width=160,
        height=120,
        fps=30,
        duration_sec=5.0,
        max_frames=4,
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
        evidence_pipeline_fingerprint_width=16,
        evidence_pipeline_fingerprint_height=16,
        evidence_pipeline_fingerprint_cycle=6,
        evidence_pipeline_real_fingerprint_enabled=True,
    )

    result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames())).run()

    assert result.ok is True
    assert result.motion_event_ids
    assert result.evidence_pipeline_event_ids
    assert len(result.evidence_pipeline_artifact_ids) == 2

    csv_artifacts = db.list_artifacts(session_id=result.session_id, artifact_type="evidence_pipeline_manifest_csv")
    profile_artifacts = db.list_artifacts(session_id=result.session_id, artifact_type="evidence_pipeline_profile")
    assert len(csv_artifacts) == 1
    assert len(profile_artifacts) == 1

    csv_path = Path(csv_artifacts[0]["path"])
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == len(result.motion_event_ids)
    assert rows
    assert all(row["fingerprint_source"] == "decoded_keyframe" for row in rows)
    assert all(int(row["decoded_width"]) == 160 for row in rows)
    assert all(int(row["decoded_height"]) == 120 for row in rows)
    assert all(row["ahash64"] for row in rows)
    assert all(row["dhash64"] for row in rows)
    assert all(row["fingerprint64"] for row in rows)
    assert all(len(row["histogram16"].split("|")) == 16 for row in rows)

    profile = json.loads(Path(profile_artifacts[0]["path"]).read_text(encoding="utf-8"))
    native = profile["result"]["evidence_pipeline"]
    assert native["ok"] is True
    assert native["real_media_ingestion"] is True
    assert native["media_fingerprint_count"] == len(rows)
    assert native["synthetic_fingerprint_count"] == 0
    assert native["fingerprint_count"] == len(rows)
    assert native["safety"]["ok"] is True
    assert native["safety"]["violation_count"] == 0
    assert all(item["from_media"] is True for item in native["fingerprints"])
    assert all(item["fingerprint_source"] == "decoded_keyframe" for item in native["fingerprints"])
    assert all(item["decoded_width"] == 160 and item["decoded_height"] == 120 for item in native["fingerprints"])

    event = db.get_event(result.evidence_pipeline_event_ids[0])
    assert event is not None
    attrs = event["attrs"]
    assert attrs["schema"] == "monitorme.node1_evidence_pipeline_profile.v0.1"
    assert attrs["ok"] is True
    assert attrs["fingerprint_count"] == len(rows)
    assert attrs["media_fingerprint_count"] == len(rows)
    assert attrs["synthetic_fingerprint_count"] == 0
    assert attrs["real_media_ingestion"] is True
    assert attrs["privacy"]["media_decode"] is True
    assert attrs["privacy"]["media_decode_scope"] == "stored_keyframes_only"
    assert attrs["facts_only"] is True

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["evidence_pipeline"]["enabled"] is True
    assert manifest["evidence_pipeline"]["real_fingerprint_enabled"] is True
    assert manifest["evidence_pipeline"]["fingerprint_source"] == "decoded_keyframe"
    assert manifest["evidence_pipeline"]["last_result"]["ok"] is True


def test_capture_run_can_disable_real_media_fingerprint_decode(tmp_path: Path):
    _require_native_cpu()
    db = MonitorMeDB(tmp_path / "monitorme.db")
    config = LocalCaptureConfig(
        camera_id="c922_node1_gate",
        device="/dev/video0",
        width=160,
        height=120,
        fps=30,
        duration_sec=5.0,
        max_frames=4,
        motion_threshold=1.0,
        min_event_gap_sec=0.0,
        data_root=str(tmp_path / "data"),
        evidence_pipeline_enabled=True,
        evidence_pipeline_binary=str(NATIVE_CPU),
        evidence_pipeline_min_key_gap_ms=1,
        evidence_pipeline_real_fingerprint_enabled=False,
    )

    result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames())).run()
    assert result.ok is True
    profile_artifacts = db.list_artifacts(session_id=result.session_id, artifact_type="evidence_pipeline_profile")
    native = json.loads(Path(profile_artifacts[0]["path"]).read_text(encoding="utf-8"))["result"]["evidence_pipeline"]
    assert native["real_media_ingestion"] is False
    assert native["media_fingerprint_count"] == 0
    assert native["synthetic_fingerprint_count"] == native["fingerprint_count"]
    assert all(item["from_media"] is False for item in native["fingerprints"])
