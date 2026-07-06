from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from monitor_me.db import MonitorMeDB
from monitor_me.operator_dashboard import build_operator_dashboard_context
from monitor_me.routes import create_app
from tests.test_node1_evidence_retention_phase14 import _seed_profile


def test_evidence_retention_schedule_migration_defaults_and_skip(tmp_path: Path) -> None:
    db = MonitorMeDB(tmp_path / "monitorme.db")
    migrations = {row[0] for row in db.conn.execute("SELECT version FROM schema_migrations").fetchall()}
    assert "007_evidence_retention_schedule.sql" in migrations
    tables = {row[0] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "evidence_retention_schedule" in tables
    assert "evidence_retention_scheduler_runs" in tables

    schedule = db.get_evidence_retention_schedule()
    assert schedule is not None
    assert schedule["schedule_id"] == "default"
    assert schedule["enabled"] == 0
    assert schedule["dry_run"] == 1
    assert schedule["privacy"]["facts_only"] is True

    skipped = db.run_evidence_retention_schedule(now="2026-07-06T00:00:00+00:00")
    assert skipped["ok"] is True
    assert skipped["status"] == "skipped"
    assert skipped["reason"] == "schedule_disabled"
    runs = db.list_evidence_retention_scheduler_runs(limit=5)
    assert len(runs) == 1
    assert runs[0]["status"] == "skipped"
    db.close()


def test_evidence_retention_schedule_forced_dry_run_records_retention_run(tmp_path: Path) -> None:
    db = MonitorMeDB(tmp_path / "monitorme.db")
    _seed_profile(db, camera_id="cam_a", session_id="sess_old", suffix="old", created_at="2026-01-01T00:00:00+00:00")
    _seed_profile(db, camera_id="cam_a", session_id="sess_keep", suffix="keep", created_at="2026-07-01T00:00:00+00:00")

    schedule = db.configure_evidence_retention_schedule(
        enabled=True,
        cadence="daily",
        older_than_days=30,
        keep_last_per_camera=1,
        keep_last_per_session=0,
        camera_id="cam_a",
        dry_run=True,
        next_run_after="2026-07-06T00:00:00+00:00",
    )
    assert schedule["enabled"] == 1
    assert schedule["dry_run"] == 1

    result = db.run_evidence_retention_schedule(now="2026-07-06T00:00:00+00:00")
    assert result["ok"] is True
    assert result["status"] == "dry_run"
    assert result["due"] is True
    assert result["retention_run_id"].startswith("eret_")
    assert result["retention_result"]["dry_run"] is True
    assert result["retention_result"]["plan"]["profiles_selected"] == 1

    updated = db.get_evidence_retention_schedule()
    assert updated["last_retention_run_id"] == result["retention_run_id"]
    assert updated["next_run_after"] is not None

    second = db.run_evidence_retention_schedule(now="2026-07-06T00:01:00+00:00")
    assert second["status"] == "skipped"
    assert second["reason"] == "not_due"
    db.close()


def test_evidence_retention_schedule_apply_requires_explicit_confirmation(tmp_path: Path) -> None:
    db = MonitorMeDB(tmp_path / "monitorme.db")
    with pytest.raises(ValueError):
        db.configure_evidence_retention_schedule(enabled=True, dry_run=False)
    schedule = db.configure_evidence_retention_schedule(enabled=True, dry_run=False, allow_destructive=True)
    assert schedule["dry_run"] == 0
    with pytest.raises(ValueError):
        db.run_evidence_retention_schedule(force=True)
    db.close()


def test_evidence_retention_schedule_cli_and_api_controls(tmp_path: Path) -> None:
    db_path = tmp_path / "monitorme.db"
    db = MonitorMeDB(db_path)
    _seed_profile(db, camera_id="cam_a", session_id="sess_old", suffix="old", created_at="2026-01-01T00:00:00+00:00")
    _seed_profile(db, camera_id="cam_a", session_id="sess_keep", suffix="keep", created_at="2026-07-01T00:00:00+00:00")
    db.close()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    refused = subprocess.run(
        [
            "python",
            "-m",
            "monitor_me.cli",
            "--db",
            str(db_path),
            "evidence-retention-schedule-set",
            "--enable",
            "--apply",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert refused.returncode == 2

    configured = subprocess.run(
        [
            "python",
            "-m",
            "monitor_me.cli",
            "--db",
            str(db_path),
            "evidence-retention-schedule-set",
            "--enable",
            "--cadence",
            "daily",
            "--older-than-days",
            "30",
            "--keep-last-per-camera",
            "1",
            "--keep-last-per-session",
            "0",
            "--camera-id",
            "cam_a",
            "--dry-run",
            "--next-run-after",
            "2026-07-06T00:00:00+00:00",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert configured.returncode == 0, configured.stdout + configured.stderr
    assert json.loads(configured.stdout)["schedule"]["enabled"] == 1

    run = subprocess.run(
        [
            "python",
            "-m",
            "monitor_me.cli",
            "--db",
            str(db_path),
            "evidence-retention-schedule-run",
            "--force",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(run.stdout)["status"] == "dry_run"

    client = TestClient(create_app(str(db_path)))
    show = client.get("/evidence/pipeline/retention/schedule")
    assert show.status_code == 200, show.text
    assert show.json()["schedule"]["enabled"] == 1

    blocked = client.post("/evidence/pipeline/retention/schedule", params={"dry_run": "false"})
    assert blocked.status_code == 400

    api_run = client.post("/evidence/pipeline/retention/schedule/run", params={"force": "true", "dry_run": "true"})
    assert api_run.status_code == 200, api_run.text
    assert api_run.json()["status"] == "dry_run"

    runs = client.get("/evidence/pipeline/retention/scheduler-runs")
    assert runs.status_code == 200, runs.text
    assert runs.json()["count"] >= 2


def test_operator_dashboard_includes_scheduled_retention_status(tmp_path: Path) -> None:
    db = MonitorMeDB(tmp_path / "monitorme.db")
    db.configure_evidence_retention_schedule(enabled=True, cadence="weekly", dry_run=True)
    result = db.run_evidence_retention_schedule(force=True)
    assert result["ok"] is True
    context = build_operator_dashboard_context(db, limit=5, fingerprint_limit=2, retention_limit=5)
    assert context["retention_schedule"]["enabled"] == 1
    assert context["cards"]["scheduler_run_count"] >= 1
    assert context["privacy"]["scheduled_retention_visible"] is True
    assert context["privacy"]["scheduled_retention_apply_from_dashboard"] is False
    db.close()
