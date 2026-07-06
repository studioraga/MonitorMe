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


def _seed_profile(db: MonitorMeDB, *, camera_id: str, session_id: str, suffix: str, created_at: str, fp_count: int = 2) -> dict[str, str]:
    db.upsert_camera(camera_id, name=f"Camera {camera_id}", device="/dev/video0")
    db.upsert_model(
        "node1-non-llm-evidence-pipeline-v0.1",
        role="evidence_pipeline",
        provider="local",
        version="v0.1",
        path="native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab",
        metadata={"facts_only": True},
        enabled=False,
    )
    try:
        db.create_session(
            session_id=session_id,
            camera_id=camera_id,
            manifest_path=f"data/captures/{session_id}/manifest.json",
            dataset_path=f"data/captures/{session_id}",
            frames_seen=fp_count,
            frames_written=fp_count,
        )
    except Exception:
        pass
    manifest_artifact_id = db.add_artifact(
        session_id=session_id,
        camera_id=camera_id,
        artifact_type="evidence_pipeline_manifest_csv",
        path=f"data/captures/{session_id}/evidence_pipeline/{suffix}_manifest.csv",
        media_type="text/csv",
        size_bytes=256,
        sha256=(suffix[:1] or "0") * 64,
    )
    profile_artifact_id = db.add_artifact(
        session_id=session_id,
        camera_id=camera_id,
        artifact_type="evidence_pipeline_profile",
        path=f"data/captures/{session_id}/evidence_pipeline/{suffix}_profile.json",
        media_type="application/json",
        size_bytes=512,
        sha256=(suffix[-1:] or "1") * 64,
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
    fingerprints = []
    for idx in range(fp_count):
        fingerprints.append(
            {
                "clip_id": f"evt_{suffix}_{idx}",
                "clip_index": idx,
                "path": f"keyframes/frame_{idx:06d}.jpg",
                "start_ms": idx * 33,
                "duration_ms": 33,
                "from_media": True,
                "fingerprint_source": "decoded_keyframe",
                "decoded_width": 160,
                "decoded_height": 120,
                "ahash64": str(1000 + idx),
                "dhash64": str(2000 + idx),
                "fingerprint64": str(3000 + idx),
                "fingerprint_hex": f"0x{3000 + idx:016X}",
                "histogram": [idx + 1] * 16,
                "histogram_bins": 16,
                "duplicate_group": -1,
                "duplicate_of": -1,
                "nearest_hamming": idx,
                "fingerprint_score": 1.0,
            }
        )
    evidence = {
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
    profile_id = db.persist_evidence_pipeline_index(
        event_id=event_id,
        session_id=session_id,
        camera_id=camera_id,
        manifest_artifact_id=manifest_artifact_id,
        profile_artifact_id=profile_artifact_id,
        manifest_csv_path=f"data/captures/{session_id}/evidence_pipeline/{suffix}_manifest.csv",
        profile_path=f"data/captures/{session_id}/evidence_pipeline/{suffix}_profile.json",
        evidence=evidence,
        capture_manifest_rows=fp_count,
    )
    db.conn.execute("UPDATE evidence_pipeline_profiles SET created_at=? WHERE profile_id=?", (created_at, profile_id))
    db.conn.commit()
    return {"profile_id": profile_id, "event_id": event_id, "session_id": session_id, "manifest_artifact_id": manifest_artifact_id, "profile_artifact_id": profile_artifact_id}


def test_evidence_retention_migration_and_plan_keep_latest(tmp_path: Path) -> None:
    db = MonitorMeDB(tmp_path / "monitorme.db")
    migrations = {row[0] for row in db.conn.execute("SELECT version FROM schema_migrations").fetchall()}
    assert "006_evidence_index_retention.sql" in migrations
    tables = {row[0] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "evidence_retention_runs" in tables

    old1 = _seed_profile(db, camera_id="cam_a", session_id="sess_old_1", suffix="old1", created_at="2026-01-01T00:00:00+00:00")
    old2 = _seed_profile(db, camera_id="cam_a", session_id="sess_old_2", suffix="old2", created_at="2026-02-01T00:00:00+00:00")
    new1 = _seed_profile(db, camera_id="cam_a", session_id="sess_new_1", suffix="new1", created_at="2026-07-01T00:00:00+00:00")

    plan = db.plan_evidence_index_retention(
        older_than_days=30,
        keep_last_per_camera=1,
        keep_last_per_session=0,
        camera_id="cam_a",
        now="2026-07-06T00:00:00+00:00",
    )
    assert plan["ok"] is True
    assert plan["profiles_scanned"] == 3
    assert plan["profiles_selected"] == 2
    assert set(plan["profile_ids"]) == {old1["profile_id"], old2["profile_id"]}
    assert new1["profile_id"] not in plan["profile_ids"]
    assert plan["rows_selected"]["fingerprints"] == 4
    assert plan["retains_source_events"] is True
    assert plan["retains_capture_artifacts"] is True
    assert plan["retains_keyframe_files"] is True
    db.close()


def test_evidence_retention_apply_deletes_only_index_rows_and_records_run(tmp_path: Path) -> None:
    db = MonitorMeDB(tmp_path / "monitorme.db")
    old = _seed_profile(db, camera_id="cam_a", session_id="sess_old", suffix="old", created_at="2026-01-01T00:00:00+00:00")
    keep = _seed_profile(db, camera_id="cam_a", session_id="sess_keep", suffix="keep", created_at="2026-07-01T00:00:00+00:00")

    dry = db.apply_evidence_index_retention(
        dry_run=True,
        older_than_days=30,
        keep_last_per_camera=1,
        keep_last_per_session=0,
        camera_id="cam_a",
        compact=True,
    )
    assert dry["ok"] is True
    assert dry["dry_run"] is True
    assert dry["deleted"]["profiles"] == 0
    assert db.get_evidence_profile(old["profile_id"]) is not None

    applied = db.apply_evidence_index_retention(
        dry_run=False,
        older_than_days=30,
        keep_last_per_camera=1,
        keep_last_per_session=0,
        camera_id="cam_a",
        compact=True,
    )
    assert applied["ok"] is True
    assert applied["deleted"]["profiles"] == 1
    assert db.get_evidence_profile(old["profile_id"]) is None
    assert db.get_evidence_profile(keep["profile_id"]) is not None
    assert db.list_evidence_fingerprints(profile_id=old["profile_id"]) == []
    assert db.get_event(old["event_id"]) is not None
    assert db.list_artifacts(session_id=old["session_id"]) != []

    runs = db.list_evidence_retention_runs(limit=10)
    assert len(runs) >= 2
    completed = next(row for row in runs if row["status"] == "completed")
    assert completed["profiles_selected"] == 1
    assert completed["fingerprints_selected"] == 2
    assert completed["policy"]["delete_scope"] == "evidence_index_rows_only"
    db.close()


def test_evidence_retention_cli_and_api_controls(tmp_path: Path) -> None:
    db_path = tmp_path / "monitorme.db"
    db = MonitorMeDB(db_path)
    old = _seed_profile(db, camera_id="cam_a", session_id="sess_old", suffix="old", created_at="2026-01-01T00:00:00+00:00")
    _seed_profile(db, camera_id="cam_a", session_id="sess_keep", suffix="keep", created_at="2026-07-01T00:00:00+00:00")
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
            "evidence-retention-plan",
            "--older-than-days",
            "30",
            "--keep-last-per-camera",
            "1",
            "--keep-last-per-session",
            "0",
            "--camera-id",
            "cam_a",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert json.loads(plan.stdout)["profiles_selected"] == 1

    refused = subprocess.run(
        [
            "python",
            "-m",
            "monitor_me.cli",
            "--db",
            str(db_path),
            "evidence-retention-apply",
            "--profile-id",
            old["profile_id"],
            "--keep-last-per-camera",
            "0",
            "--keep-last-per-session",
            "0",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert refused.returncode == 2

    client = TestClient(create_app(str(db_path)))
    api_plan = client.get(
        "/evidence/pipeline/retention/plan",
        params={"profile_id": old["profile_id"], "keep_last_per_camera": 0, "keep_last_per_session": 0},
    )
    assert api_plan.status_code == 200, api_plan.text
    assert api_plan.json()["profiles_selected"] == 1

    blocked = client.post(
        "/evidence/pipeline/retention/apply",
        params={"dry_run": "false", "profile_id": old["profile_id"], "keep_last_per_camera": 0, "keep_last_per_session": 0},
    )
    assert blocked.status_code == 400

    applied = client.post(
        "/evidence/pipeline/retention/apply",
        params={"dry_run": "false", "confirm": "true", "profile_id": old["profile_id"], "keep_last_per_camera": 0, "keep_last_per_session": 0},
    )
    assert applied.status_code == 200, applied.text
    assert applied.json()["deleted"]["profiles"] == 1

    runs = client.get("/evidence/pipeline/retention/runs")
    assert runs.status_code == 200, runs.text
    assert runs.json()["count"] >= 1
