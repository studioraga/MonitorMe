from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from monitor_me.db import MonitorMeDB
from monitor_me.routes import create_app


def _seed_evidence_index(db_path):
    db = MonitorMeDB(db_path)
    camera_id = "c922_node1_gate"
    session_id = "sess_api_evidence"
    db.upsert_camera(camera_id, name="C922 Node1 Gate", device="/dev/video0")
    db.upsert_model(
        "node1-non-llm-evidence-pipeline-v0.1",
        role="evidence_pipeline",
        provider="local",
        version="v0.1",
        path="native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab",
        metadata={"facts_only": True},
        enabled=False,
    )
    db.create_session(
        session_id=session_id,
        camera_id=camera_id,
        manifest_path="data/captures/sess_api_evidence/manifest.json",
        dataset_path="data/captures/sess_api_evidence",
        frames_seen=2,
        frames_written=2,
    )
    manifest_artifact_id = db.add_artifact(
        session_id=session_id,
        camera_id=camera_id,
        artifact_type="evidence_pipeline_manifest_csv",
        path="data/captures/sess_api_evidence/evidence_pipeline/capture_evidence_manifest.csv",
        media_type="text/csv",
        size_bytes=256,
        sha256="0" * 64,
    )
    profile_artifact_id = db.add_artifact(
        session_id=session_id,
        camera_id=camera_id,
        artifact_type="evidence_pipeline_profile",
        path="data/captures/sess_api_evidence/evidence_pipeline/evidence_pipeline_profile.json",
        media_type="application/json",
        size_bytes=512,
        sha256="1" * 64,
    )
    event_id = db.insert_event(
        camera_id=camera_id,
        session_id=session_id,
        event_type="evidence_pipeline_indexed",
        label="facts_only_evidence_pipeline",
        confidence=1.0,
        artifact_id=profile_artifact_id,
        model_id="node1-non-llm-evidence-pipeline-v0.1",
        attrs={"facts_only": True},
    )
    evidence = {
        "schema": "node1_non_llm_evidence_pipeline.v0.1",
        "facts_only": True,
        "manifest_entries": 2,
        "fingerprint_count": 2,
        "media_fingerprint_count": 2,
        "synthetic_fingerprint_count": 0,
        "real_media_ingestion": True,
        "duplicate_group_count": 1,
        "duplicate_clip_count": 1,
        "unique_clip_count": 1,
        "key_moment_count": 1,
        "planned_read_bytes": 3000,
        "total_manifest_bytes": 3000,
        "timeline": {
            "clip_count": 2,
            "timeline_start_ms": 0,
            "timeline_end_ms": 66,
            "timeline_span_ms": 66,
            "covered_duration_ms": 66,
            "max_gap_ms": 0,
            "total_bytes": 3000,
        },
        "latency": {"total_ms": 0.25, "fingerprint_ms": 0.1, "dedup_ms": 0.05, "key_selection_ms": 0.02, "planned_read_mb_per_s": 12.0},
        "safety": {"ok": True, "violation_count": 0, "facts_only": True, "no_semantic_claims": True},
        "fingerprints": [
            {
                "clip_id": "evt_a",
                "clip_index": 0,
                "path": "keyframes/frame_000001.jpg",
                "start_ms": 0,
                "duration_ms": 33,
                "from_media": True,
                "fingerprint_source": "decoded_keyframe",
                "decoded_width": 160,
                "decoded_height": 120,
                "ahash64": "11",
                "dhash64": "22",
                "fingerprint64": "33",
                "fingerprint_hex": "0x21",
                "histogram": [1] * 16,
                "histogram_bins": 16,
                "duplicate_group": 0,
                "duplicate_of": -1,
                "nearest_hamming": 0,
                "fingerprint_score": 1.0,
            },
            {
                "clip_id": "evt_b",
                "clip_index": 1,
                "path": "keyframes/frame_000002.jpg",
                "start_ms": 33,
                "duration_ms": 33,
                "from_media": True,
                "fingerprint_source": "decoded_keyframe",
                "decoded_width": 160,
                "decoded_height": 120,
                "ahash64": "11",
                "dhash64": "22",
                "fingerprint64": "33",
                "fingerprint_hex": "0x21",
                "histogram": [1] * 16,
                "histogram_bins": 16,
                "duplicate_group": 0,
                "duplicate_of": 0,
                "nearest_hamming": 0,
                "fingerprint_score": 1.0,
            },
        ],
        "duplicate_groups": [
            {
                "group_id": 0,
                "representative_clip_id": "evt_a",
                "representative_clip_index": 0,
                "group_size": 2,
                "duplicate_count": 1,
                "min_hamming": 0,
                "max_hamming": 0,
                "clip_ids": ["evt_a", "evt_b"],
                "clip_indices": [0, 1],
            }
        ],
        "key_moments": [
            {
                "rank": 1,
                "clip_id": "evt_a",
                "clip_index": 0,
                "start_ms": 0,
                "duration_ms": 33,
                "reason": "priority_score",
                "priority_score": 0.9,
                "motion_score": 0.2,
                "audio_score": 0.0,
                "lighting_delta": 5.0,
                "changed_pixels": 100,
            }
        ],
    }
    profile_id = db.persist_evidence_pipeline_index(
        event_id=event_id,
        session_id=session_id,
        camera_id=camera_id,
        manifest_artifact_id=manifest_artifact_id,
        profile_artifact_id=profile_artifact_id,
        manifest_csv_path="data/captures/sess_api_evidence/evidence_pipeline/capture_evidence_manifest.csv",
        profile_path="data/captures/sess_api_evidence/evidence_pipeline/evidence_pipeline_profile.json",
        evidence=evidence,
        capture_manifest_rows=2,
    )
    db.close()
    return {"profile_id": profile_id, "event_id": event_id, "session_id": session_id, "camera_id": camera_id}


def test_evidence_pipeline_summary_api_routes_are_registered(tmp_path):
    app = create_app(str(tmp_path / "monitorme.db"))
    paths = {route.path for route in app.routes}

    assert "/evidence/pipeline/summaries" in paths
    assert "/evidence/pipeline/sessions/{session_id}/summary" in paths
    assert "/evidence/pipeline/profiles/{profile_id}/summary" in paths
    assert "/evidence/pipeline/profiles/{profile_id}/fingerprints" in paths
    assert "/evidence/pipeline/profiles/{profile_id}/dedup-groups" in paths
    assert "/evidence/pipeline/profiles/{profile_id}/key-moments" in paths


def test_evidence_pipeline_summary_api_returns_facts_only_summary(tmp_path):
    db_path = tmp_path / "monitorme.db"
    ids = _seed_evidence_index(db_path)
    client = TestClient(create_app(str(db_path)))

    summaries = client.get("/evidence/pipeline/summaries", params={"session_id": ids["session_id"]})
    assert summaries.status_code == 200, summaries.text
    payload = summaries.json()
    assert payload["ok"] is True
    assert payload["count"] == 1
    summary = payload["evidence_pipeline_summaries"][0]
    assert summary["profile_id"] == ids["profile_id"]
    assert summary["counts"]["fingerprint_count"] == 2
    assert summary["counts"]["media_fingerprint_count"] == 2
    assert summary["ingestion"]["real_media_ingestion"] is True
    assert summary["ingestion"]["facts_only"] is True
    assert summary["safety"]["violation_count"] == 0
    assert summary["privacy"]["external_upload"] is False
    assert summary["privacy"]["media_decode_in_api"] is False
    assert summary["privacy"]["semantic_claims"] is False

    profile = client.get(
        f"/evidence/pipeline/profiles/{ids['profile_id']}/summary",
        params={"include_fingerprints": "true", "fingerprint_limit": 1},
    )
    assert profile.status_code == 200, profile.text
    profile_summary = profile.json()["evidence_pipeline_summary"]
    assert profile_summary["profile_id"] == ids["profile_id"]
    assert profile_summary["fingerprints_truncated"] is True
    assert len(profile_summary["fingerprints"]) == 1
    assert profile_summary["fingerprints"][0]["from_media"] == 1
    assert profile_summary["fingerprints"][0]["fingerprint_source"] == "decoded_keyframe"
    assert len(profile_summary["key_moments"]) == 1
    assert len(profile_summary["dedup_groups"]) == 1


def test_evidence_pipeline_detail_api_routes_filter_by_profile(tmp_path):
    db_path = tmp_path / "monitorme.db"
    ids = _seed_evidence_index(db_path)
    client = TestClient(create_app(str(db_path)))

    session_summary = client.get(f"/evidence/pipeline/sessions/{ids['session_id']}/summary")
    assert session_summary.status_code == 200, session_summary.text
    assert session_summary.json()["count"] == 1

    fps = client.get(f"/evidence/pipeline/profiles/{ids['profile_id']}/fingerprints", params={"from_media": "true"})
    assert fps.status_code == 200, fps.text
    assert fps.json()["count"] == 2
    assert all(row["from_media"] == 1 for row in fps.json()["evidence_fingerprints"])

    groups = client.get(f"/evidence/pipeline/profiles/{ids['profile_id']}/dedup-groups")
    assert groups.status_code == 200, groups.text
    assert groups.json()["count"] == 1
    assert groups.json()["evidence_dedup_groups"][0]["duplicate_count"] == 1

    moments = client.get(f"/evidence/pipeline/profiles/{ids['profile_id']}/key-moments")
    assert moments.status_code == 200, moments.text
    assert moments.json()["count"] == 1
    assert moments.json()["evidence_key_moments"][0]["rank"] == 1

    missing = client.get("/evidence/pipeline/profiles/missing/summary")
    assert missing.status_code == 404
