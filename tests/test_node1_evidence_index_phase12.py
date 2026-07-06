from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from pathlib import Path

import pytest

from monitor_me.db import MonitorMeDB
from monitor_me.local_capture import IterableFrameSource, LocalCameraCaptureRunner, LocalCaptureConfig
from tests.helpers import motion_frames


NATIVE_CPU = Path("native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab")


def _require_native_cpu() -> None:
    if not NATIVE_CPU.exists():
        pytest.skip("native CPU binary is not built")


def test_evidence_index_migration_tables_are_applied(tmp_path: Path) -> None:
    db = MonitorMeDB(tmp_path / "monitorme.db")
    tables = {
        row[0]
        for row in db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'evidence_%'"
        ).fetchall()
    }
    assert "evidence_pipeline_profiles" in tables
    assert "evidence_fingerprints" in tables
    assert "evidence_dedup_groups" in tables
    assert "evidence_key_moments" in tables
    migrations = {row[0] for row in db.conn.execute("SELECT version FROM schema_migrations").fetchall()}
    assert "005_evidence_index_persistence.sql" in migrations
    db.close()


def test_capture_run_persists_queryable_evidence_index_rows(tmp_path: Path) -> None:
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
    assert result.evidence_pipeline_event_ids
    event_id = result.evidence_pipeline_event_ids[0]
    profile_rows = db.list_evidence_profiles(event_id=event_id)
    assert len(profile_rows) == 1
    profile = profile_rows[0]
    assert profile["event_id"] == event_id
    assert profile["session_id"] == result.session_id
    assert profile["camera_id"] == "c922_node1_gate"
    assert profile["native_schema"] == "node1_non_llm_evidence_pipeline.v0.1"
    assert profile["capture_manifest_rows"] == len(result.motion_event_ids)
    assert profile["fingerprint_count"] == len(result.motion_event_ids)
    assert profile["media_fingerprint_count"] == len(result.motion_event_ids)
    assert profile["synthetic_fingerprint_count"] == 0
    assert profile["real_media_ingestion"] == 1
    assert profile["safety_ok"] == 1
    assert profile["violation_count"] == 0
    assert profile["facts_only"] == 1
    assert profile["safety"]["ok"] is True
    assert profile["safety"]["violation_count"] == 0
    assert profile["latency"]["total_ms"] >= 0

    profile_id = profile["profile_id"]
    fingerprints = db.list_evidence_fingerprints(profile_id=profile_id, limit=100)
    assert len(fingerprints) == len(result.motion_event_ids)
    assert all(row["from_media"] == 1 for row in fingerprints)
    assert all(row["fingerprint_source"] == "decoded_keyframe" for row in fingerprints)
    assert all(row["decoded_width"] == 160 and row["decoded_height"] == 120 for row in fingerprints)
    assert all(row["fingerprint_hex"].startswith("0x") for row in fingerprints)
    assert all(len(row["histogram"]) == 16 for row in fingerprints)

    key_moments = db.list_evidence_key_moments(profile_id=profile_id)
    assert len(key_moments) == profile["key_moment_count"]
    assert key_moments
    assert [row["rank"] for row in key_moments] == sorted(row["rank"] for row in key_moments)

    dedup_groups = db.list_evidence_dedup_groups(profile_id=profile_id)
    assert len(dedup_groups) == profile["duplicate_group_count"]
    assert all(isinstance(row["clip_ids"], list) for row in dedup_groups)

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["evidence_pipeline"]["last_result"]["evidence_index_profile_id"] == profile_id
    db.close()


def test_evidence_index_cli_lists_persisted_rows(tmp_path: Path) -> None:
    _require_native_cpu()
    db_path = tmp_path / "monitorme.db"
    db = MonitorMeDB(db_path)
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
        evidence_pipeline_real_fingerprint_enabled=True,
    )
    result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(motion_frames())).run()
    assert result.ok is True
    db.close()

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    idx = subprocess.run(
        ["python", "-m", "monitor_me.cli", "--db", str(db_path), "evidence-index", "--session-id", result.session_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert idx.returncode == 0, idx.stdout + idx.stderr
    idx_payload = json.loads(idx.stdout)
    assert idx_payload["count"] == 1
    profile_id = idx_payload["evidence_profiles"][0]["profile_id"]

    fps = subprocess.run(
        ["python", "-m", "monitor_me.cli", "--db", str(db_path), "evidence-fingerprints", "--profile-id", profile_id, "--from-media", "--limit", "100"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert fps.returncode == 0, fps.stdout + fps.stderr
    fp_payload = json.loads(fps.stdout)
    assert fp_payload["count"] == len(result.motion_event_ids)

    km = subprocess.run(
        ["python", "-m", "monitor_me.cli", "--db", str(db_path), "evidence-key-moments", "--profile-id", profile_id],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert km.returncode == 0, km.stdout + km.stderr
    assert json.loads(km.stdout)["count"] >= 1
