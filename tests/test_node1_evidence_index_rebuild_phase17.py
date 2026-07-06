from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from monitor_me.db import MonitorMeDB
from monitor_me.routes import create_app


MODEL_ID = "node1-non-llm-evidence-pipeline-v0.1"


def _evidence_payload(fp_count: int = 3) -> dict:
    fingerprints = []
    for idx in range(fp_count):
        fingerprints.append(
            {
                "clip_id": f"evt_rebuild_clip_{idx}",
                "clip_index": idx,
                "path": f"keyframes/frame_{idx:06d}.jpg",
                "start_ms": idx * 33,
                "duration_ms": 33,
                "from_media": True,
                "fingerprint_source": "decoded_keyframe",
                "decoded_width": 1280,
                "decoded_height": 720,
                "ahash64": str(10000 + idx),
                "dhash64": str(20000 + idx),
                "fingerprint64": str(30000 + idx),
                "fingerprint_hex": f"0x{30000 + idx:016X}",
                "histogram": [idx + 1] * 16,
                "histogram_bins": 16,
                "duplicate_group": -1,
                "duplicate_of": -1,
                "nearest_hamming": idx,
                "fingerprint_score": 1.0,
            }
        )
    return {
        "schema": "node1_non_llm_evidence_pipeline.v0.1",
        "facts_only": True,
        "manifest_entries": fp_count,
        "fingerprint_count": fp_count,
        "media_fingerprint_count": fp_count,
        "synthetic_fingerprint_count": 0,
        "real_media_ingestion": True,
        "duplicate_group_count": 0,
        "duplicate_clip_count": 0,
        "unique_clip_count": fp_count,
        "key_moment_count": 1,
        "planned_read_bytes": 4096 * fp_count,
        "total_manifest_bytes": 4096 * fp_count,
        "timeline": {"clip_count": fp_count, "timeline_start_ms": 0, "timeline_end_ms": fp_count * 33, "total_bytes": 4096 * fp_count},
        "latency": {"total_ms": 0.1, "fingerprint_ms": 0.01, "dedup_ms": 0.01, "key_selection_ms": 0.01, "planned_read_mb_per_s": 1.0},
        "safety": {"ok": True, "violation_count": 0, "facts_only": True, "no_semantic_claims": True},
        "fingerprints": fingerprints,
        "duplicate_groups": [],
        "key_moments": [
            {
                "rank": 1,
                "clip_id": fingerprints[0]["clip_id"],
                "clip_index": 0,
                "start_ms": 0,
                "duration_ms": 33,
                "reason": "priority_score",
                "priority_score": 0.5,
                "motion_score": 0.1,
                "audio_score": 0.0,
                "lighting_delta": 1.0,
                "changed_pixels": 10,
            }
        ],
    }


def _seed_retained_artifacts(db: MonitorMeDB, artifact_root: Path, *, persist_index: bool = True) -> dict[str, str]:
    camera_id = "cam_rebuild"
    session_id = "sess_rebuild"
    db.upsert_camera(camera_id, name="Rebuild Camera", device="/dev/video0")
    db.upsert_model(MODEL_ID, role="evidence_pipeline", provider="local", version="v0.1", metadata={"facts_only": True})
    db.create_session(
        session_id=session_id,
        camera_id=camera_id,
        manifest_path=f"data/captures/{session_id}/manifest.json",
        dataset_path=f"data/captures/{session_id}",
        frames_seen=3,
        frames_written=3,
    )
    evidence_dir = artifact_root / "data" / "captures" / session_id / "evidence_pipeline"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    csv_rel = f"data/captures/{session_id}/evidence_pipeline/capture_evidence_manifest.csv"
    profile_rel = f"data/captures/{session_id}/evidence_pipeline/evidence_pipeline_profile.json"
    (artifact_root / csv_rel).write_text("clip_id,path,start_ms,duration_ms,bytes\n", encoding="utf-8")
    evidence = _evidence_payload(3)
    profile_payload = {
        "schema": "monitorme.node1_evidence_pipeline_profile.v0.1",
        "session_id": session_id,
        "camera_id": camera_id,
        "evidence_manifest_csv": csv_rel,
        "result": {"ok": True, "source": "test", "evidence_pipeline": evidence},
        "facts_only": True,
    }
    (artifact_root / profile_rel).write_text(json.dumps(profile_payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_artifact_id = db.add_artifact(
        session_id=session_id,
        camera_id=camera_id,
        artifact_type="evidence_pipeline_manifest_csv",
        path=csv_rel,
        media_type="text/csv",
        size_bytes=64,
        sha256="a" * 64,
    )
    profile_artifact_id = db.add_artifact(
        session_id=session_id,
        camera_id=camera_id,
        artifact_type="evidence_pipeline_profile",
        path=profile_rel,
        media_type="application/json",
        size_bytes=512,
        sha256="b" * 64,
    )
    event_id = db.insert_event(
        camera_id=camera_id,
        session_id=session_id,
        event_type="evidence_pipeline_indexed",
        label="facts_only_evidence_pipeline",
        confidence=1.0,
        artifact_id=profile_artifact_id,
        model_id=MODEL_ID,
        attrs={
            "schema": "monitorme.node1_evidence_pipeline_profile.v0.1",
            "manifest_artifact_id": manifest_artifact_id,
            "profile_artifact_id": profile_artifact_id,
            "manifest_csv_path": csv_rel,
            "profile_path": profile_rel,
            "capture_manifest_rows": 3,
            "facts_only": True,
        },
    )
    profile_id = ""
    if persist_index:
        profile_id = db.persist_evidence_pipeline_index(
            event_id=event_id,
            session_id=session_id,
            camera_id=camera_id,
            manifest_artifact_id=manifest_artifact_id,
            profile_artifact_id=profile_artifact_id,
            manifest_csv_path=csv_rel,
            profile_path=profile_rel,
            evidence=evidence,
            capture_manifest_rows=3,
        )
    return {
        "camera_id": camera_id,
        "session_id": session_id,
        "event_id": event_id,
        "profile_id": profile_id,
        "manifest_artifact_id": manifest_artifact_id,
        "profile_artifact_id": profile_artifact_id,
        "artifact_root": str(artifact_root),
    }


def test_evidence_index_rebuild_from_retained_artifacts_after_retention(tmp_path: Path) -> None:
    db = MonitorMeDB(tmp_path / "monitorme.db")
    ids = _seed_retained_artifacts(db, tmp_path, persist_index=True)
    assert db.get_evidence_profile(ids["profile_id"]) is not None

    removed = db.apply_evidence_index_retention(dry_run=False, profile_id=ids["profile_id"], keep_last_per_camera=0, keep_last_per_session=0)
    assert removed["ok"] is True
    assert removed["deleted"]["profiles"] == 1
    assert db.get_evidence_profile(ids["profile_id"]) is None
    assert db.get_event(ids["event_id"]) is not None
    assert db.list_artifacts(session_id=ids["session_id"]) != []

    plan = db.plan_evidence_index_rebuild(session_id=ids["session_id"], artifact_root=tmp_path)
    assert plan["ok"] is True
    assert plan["candidates_selected"] == 1
    assert plan["rows_rebuildable"]["fingerprints"] == 3
    assert plan["source_scope"]["native_rerun"] is False
    assert plan["source_scope"]["media_decode"] is False

    dry = db.rebuild_evidence_index_from_artifacts(dry_run=True, session_id=ids["session_id"], artifact_root=tmp_path)
    assert dry["ok"] is True
    assert dry["profiles_rebuilt"] == 0
    assert db.list_evidence_profiles(session_id=ids["session_id"]) == []

    rebuilt = db.rebuild_evidence_index_from_artifacts(dry_run=False, session_id=ids["session_id"], artifact_root=tmp_path)
    assert rebuilt["ok"] is True
    assert rebuilt["profiles_rebuilt"] == 1
    assert rebuilt["rows_rebuilt"]["fingerprints"] == 3
    profiles = db.list_evidence_profiles(session_id=ids["session_id"])
    assert len(profiles) == 1
    new_profile_id = profiles[0]["profile_id"]
    assert new_profile_id != ids["profile_id"]
    assert len(db.list_evidence_fingerprints(profile_id=new_profile_id)) == 3
    assert len(db.list_evidence_key_moments(profile_id=new_profile_id)) == 1
    assert db.get_event(ids["event_id"]) is not None
    assert db.list_artifacts(session_id=ids["session_id"]) != []
    runs = db.list_evidence_index_rebuild_runs(limit=10)
    assert any(row["status"] == "completed" and row["profiles_rebuilt"] == 1 for row in runs)
    db.close()


def test_evidence_index_rebuild_cli_and_api_controls(tmp_path: Path) -> None:
    db_path = tmp_path / "monitorme.db"
    db = MonitorMeDB(db_path)
    ids = _seed_retained_artifacts(db, tmp_path, persist_index=True)
    db.apply_evidence_index_retention(dry_run=False, profile_id=ids["profile_id"], keep_last_per_camera=0, keep_last_per_session=0)
    db.close()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    plan = subprocess.run(
        [
            "python",
            "-m",
            "monitor_me.cli",
            "--db",
            str(db_path),
            "evidence-index-rebuild-plan",
            "--session-id",
            ids["session_id"],
            "--artifact-root",
            str(tmp_path),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    plan_json = json.loads(plan.stdout)
    assert plan_json["ok"] is True
    assert plan_json["candidates_selected"] == 1

    refused = subprocess.run(
        [
            "python",
            "-m",
            "monitor_me.cli",
            "--db",
            str(db_path),
            "evidence-index-rebuild-apply",
            "--session-id",
            ids["session_id"],
            "--artifact-root",
            str(tmp_path),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
    )
    assert refused.returncode == 2
    assert "Refusing" in refused.stdout

    dry = subprocess.run(
        [
            "python",
            "-m",
            "monitor_me.cli",
            "--db",
            str(db_path),
            "evidence-index-rebuild-apply",
            "--dry-run",
            "--session-id",
            ids["session_id"],
            "--artifact-root",
            str(tmp_path),
        ],
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(dry.stdout)["status"] == "dry_run"

    app = create_app(str(db_path))
    client = TestClient(app)
    api_refused = client.post(f"/evidence/pipeline/rebuild/apply?session_id={ids['session_id']}&artifact_root={tmp_path}&dry_run=false")
    assert api_refused.status_code == 400
    api_apply = client.post(f"/evidence/pipeline/rebuild/apply?session_id={ids['session_id']}&artifact_root={tmp_path}&dry_run=false&confirm=true")
    assert api_apply.status_code == 200
    applied = api_apply.json()
    assert applied["ok"] is True
    assert applied["profiles_rebuilt"] == 1
    runs = client.get("/evidence/pipeline/rebuild/runs?limit=10")
    assert runs.status_code == 200
    assert runs.json()["count"] >= 1
    dashboard = client.get("/operator/dashboard/data?limit=10&retention_limit=5")
    assert dashboard.status_code == 200
    assert dashboard.json()["cards"]["rebuild_run_count"] >= 1
    assert dashboard.json()["privacy"]["evidence_index_rebuild_visible"] is True
    assert dashboard.json()["privacy"]["evidence_index_rebuild_apply_from_dashboard"] is False
