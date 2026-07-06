from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .ids import new_id
from .time_utils import now_iso


class MonitorMeDB:
    """Thread-safe SQLite access layer for MonitorMe v0.1.

    The schema intentionally stores detection facts as first-class event rows.
    Assistant answers are generated from these rows and must cite event/session
    references rather than relying on nested JSON blobs.
    """

    def __init__(self, db_path: str | Path, migrations_dir: str | Path | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 10000")
        self.conn.execute("PRAGMA journal_mode = WAL")
        if migrations_dir is None:
            migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
        self.apply_migrations(Path(migrations_dir))

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def apply_migrations(self, migrations_dir: Path) -> None:
        with self._lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations(version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            for path in sorted(migrations_dir.glob("*.sql")):
                version = path.name
                row = self.conn.execute(
                    "SELECT 1 FROM schema_migrations WHERE version=?", (version,)
                ).fetchone()
                if row:
                    continue
                self.conn.executescript(path.read_text(encoding="utf-8"))
                self.conn.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES(?, ?)",
                    (version, now_iso()),
                )
            self.conn.commit()

    @staticmethod
    def _json(data: Any) -> str:
        return json.dumps(data if data is not None else {}, sort_keys=True)

    @staticmethod
    def _decode_row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        data = dict(row)
        for key, value in list(data.items()):
            if key.endswith("_json") and isinstance(value, str) and value:
                out_key = key[:-5]
                try:
                    data[out_key] = json.loads(value)
                except json.JSONDecodeError:
                    data[out_key] = None
        return data

    @staticmethod
    def _decode_rows(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
        return [MonitorMeDB._decode_row(r) or {} for r in rows]

    def audit(
        self,
        action: str,
        *,
        outcome: str = "ok",
        actor: str | None = None,
        camera_id: str | None = None,
        event_id: str | None = None,
        session_id: str | None = None,
        report_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            audit_id = new_id("aud")
            self.conn.execute(
                """
                INSERT INTO audit_log(audit_id, action, actor, outcome, camera_id, event_id,
                                      session_id, report_id, details_json, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    audit_id,
                    action,
                    actor,
                    outcome,
                    camera_id,
                    event_id,
                    session_id,
                    report_id,
                    self._json(details or {}),
                    now_iso(),
                ),
            )
            self.conn.commit()
            return audit_id

    def upsert_camera(
        self,
        camera_id: str,
        *,
        name: str,
        location: str = "",
        source_node: str = "node1",
        source_kind: str = "local_v4l2",
        device: str = "/dev/video0",
        enabled: bool = True,
    ) -> None:
        with self._lock:
            now = now_iso()
            self.conn.execute(
                """
                INSERT INTO cameras(camera_id, name, location, source_node, source_kind, device,
                                    enabled, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    name=excluded.name,
                    location=excluded.location,
                    source_node=excluded.source_node,
                    source_kind=excluded.source_kind,
                    device=excluded.device,
                    enabled=excluded.enabled,
                    updated_at=excluded.updated_at
                """,
                (camera_id, name, location, source_node, source_kind, device, 1 if enabled else 0, now, now),
            )
            self.conn.commit()
            self.audit("camera.upsert", camera_id=camera_id, details={"source_kind": source_kind, "device": device})

    def upsert_model(
        self,
        model_id: str,
        *,
        role: str,
        provider: str = "local",
        version: str = "",
        path: str = "",
        sha256: str = "",
        metadata: dict[str, Any] | None = None,
        enabled: bool = True,
    ) -> None:
        with self._lock:
            now = now_iso()
            self.conn.execute(
                """
                INSERT INTO model_registry(model_id, role, provider, version, path, sha256,
                                           metadata_json, enabled, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(model_id) DO UPDATE SET
                    role=excluded.role,
                    provider=excluded.provider,
                    version=excluded.version,
                    path=excluded.path,
                    sha256=excluded.sha256,
                    metadata_json=excluded.metadata_json,
                    enabled=excluded.enabled,
                    updated_at=excluded.updated_at
                """,
                (model_id, role, provider, version, path, sha256, self._json(metadata or {}), 1 if enabled else 0, now, now),
            )
            self.conn.commit()
            self.audit("model.upsert", details={"model_id": model_id, "role": role})

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM model_registry WHERE model_id=?", (model_id,)).fetchone()
            return self._decode_row(row)

    def list_models(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._decode_rows(self.conn.execute("SELECT * FROM model_registry ORDER BY role, model_id"))

    def create_session(
        self,
        *,
        session_id: str | None = None,
        camera_id: str,
        source_node: str = "node1",
        source_kind: str = "local_v4l2",
        device: str = "/dev/video0",
        status: str = "completed",
        started_at: str | None = None,
        ended_at: str | None = None,
        manifest_path: str = "",
        dataset_path: str = "",
        frames_seen: int = 0,
        frames_written: int = 0,
        bytes_written: int = 0,
        error: str | None = None,
        policy_decision: dict[str, Any] | None = None,
    ) -> str:
        with self._lock:
            sid = session_id or new_id("sess")
            now = now_iso()
            started = started_at or now
            self.conn.execute(
                """
                INSERT INTO capture_sessions(session_id, camera_id, source_node, source_kind, device,
                                             started_at, ended_at, status, manifest_path, dataset_path,
                                             frames_seen, frames_written, bytes_written, error,
                                             policy_decision_json, created_at, updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    sid,
                    camera_id,
                    source_node,
                    source_kind,
                    device,
                    started,
                    ended_at or now,
                    status,
                    manifest_path,
                    dataset_path,
                    frames_seen,
                    frames_written,
                    bytes_written,
                    error,
                    self._json(policy_decision or {"decision": "allow", "reason": "local Node1 capture"}),
                    now,
                    now,
                ),
            )
            self.conn.commit()
            self.audit("session.create", camera_id=camera_id, session_id=sid, details={"status": status})
            return sid


    def update_session(
        self,
        session_id: str,
        *,
        status: str | None = None,
        ended_at: str | None = None,
        manifest_path: str | None = None,
        dataset_path: str | None = None,
        frames_seen: int | None = None,
        frames_written: int | None = None,
        bytes_written: int | None = None,
        error: str | None = None,
    ) -> None:
        """Update a capture session after a real local camera run."""
        with self._lock:
            existing = self.get_session(session_id)
            if not existing:
                raise KeyError(f"session_id not found: {session_id}")
            self.conn.execute(
                """
                UPDATE capture_sessions
                SET status=COALESCE(?, status),
                    ended_at=COALESCE(?, ended_at),
                    manifest_path=COALESCE(?, manifest_path),
                    dataset_path=COALESCE(?, dataset_path),
                    frames_seen=COALESCE(?, frames_seen),
                    frames_written=COALESCE(?, frames_written),
                    bytes_written=COALESCE(?, bytes_written),
                    error=?,
                    updated_at=?
                WHERE session_id=?
                """,
                (
                    status,
                    ended_at,
                    manifest_path,
                    dataset_path,
                    frames_seen,
                    frames_written,
                    bytes_written,
                    error,
                    now_iso(),
                    session_id,
                ),
            )
            self.conn.commit()
            self.audit("session.update", session_id=session_id, outcome=status or "ok", details={"frames_seen": frames_seen, "frames_written": frames_written, "error": error})

    def add_artifact(
        self,
        *,
        artifact_id: str | None = None,
        session_id: str,
        camera_id: str,
        artifact_type: str,
        path: str,
        media_type: str = "application/octet-stream",
        size_bytes: int | None = None,
        sha256: str | None = None,
    ) -> str:
        with self._lock:
            aid = artifact_id or new_id("art")
            self.conn.execute(
                """
                INSERT INTO capture_artifacts(artifact_id, session_id, camera_id, artifact_type, path,
                                              media_type, size_bytes, sha256, created_at)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (aid, session_id, camera_id, artifact_type, path, media_type, size_bytes, sha256, now_iso()),
            )
            self.conn.commit()
            self.audit("artifact.add", camera_id=camera_id, session_id=session_id, details={"artifact_id": aid, "path": path})
            return aid

    def insert_event(
        self,
        *,
        event_id: str | None = None,
        parent_event_id: str | None = None,
        camera_id: str,
        session_id: str | None = None,
        frame_id: int | None = None,
        ts: str | None = None,
        event_type: str,
        severity: str = "info",
        label: str | None = None,
        confidence: float | None = None,
        bbox: list[float] | None = None,
        track_id: str | None = None,
        zone_id: str | None = None,
        source_node: str = "node1",
        source_kind: str = "local_v4l2",
        model_id: str | None = None,
        artifact_id: str | None = None,
        attrs: dict[str, Any] | None = None,
        caption: str | None = None,
    ) -> str:
        with self._lock:
            eid = event_id or new_id("evt")
            now = now_iso()
            self.conn.execute(
                """
                INSERT INTO events(event_id, parent_event_id, camera_id, session_id, frame_id, ts,
                                   event_type, severity, label, confidence, bbox_json, track_id,
                                   zone_id, source_node, source_kind, model_id, artifact_id,
                                   attrs_json, caption, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    eid,
                    parent_event_id,
                    camera_id,
                    session_id,
                    frame_id,
                    ts or now,
                    event_type,
                    severity,
                    label,
                    confidence,
                    self._json(bbox) if bbox is not None else None,
                    track_id,
                    zone_id,
                    source_node,
                    source_kind,
                    model_id,
                    artifact_id,
                    self._json(attrs or {}),
                    caption,
                    now,
                ),
            )
            self.conn.commit()
            self.audit("event.insert", camera_id=camera_id, event_id=eid, session_id=session_id, details={"event_type": event_type, "label": label})
            return eid

    def insert_motion_with_detections(
        self,
        *,
        camera_id: str,
        session_id: str,
        frame_id: int,
        detections: list[dict[str, Any]],
        ts: str | None = None,
        motion_score: float | None = None,
        source_node: str = "node1",
        source_kind: str = "local_v4l2",
        artifact_id: str | None = None,
    ) -> tuple[str, list[str]]:
        parent_id = self.insert_event(
            camera_id=camera_id,
            session_id=session_id,
            frame_id=frame_id,
            ts=ts,
            event_type="motion_detected",
            severity="info",
            label="motion",
            confidence=motion_score,
            source_node=source_node,
            source_kind=source_kind,
            artifact_id=artifact_id,
            attrs={"detection_count": len(detections)},
        )
        child_ids: list[str] = []
        for det in detections:
            child_ids.append(
                self.insert_event(
                    parent_event_id=parent_id,
                    camera_id=camera_id,
                    session_id=session_id,
                    frame_id=det.get("frame_id", frame_id),
                    ts=ts,
                    event_type="object_detected",
                    severity=det.get("severity", "info"),
                    label=det.get("label"),
                    confidence=det.get("confidence"),
                    bbox=det.get("bbox"),
                    track_id=det.get("track_id"),
                    zone_id=det.get("zone_id"),
                    source_node=source_node,
                    source_kind=source_kind,
                    model_id=det.get("model_id"),
                    artifact_id=artifact_id,
                    attrs=det.get("attrs", {}),
                    caption=det.get("caption"),
                )
            )
        return parent_id, child_ids

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
            return self._decode_row(row)

    def list_events(
        self,
        *,
        camera_id: str | None = None,
        event_type: str | None = None,
        label: str | None = None,
        session_id: str | None = None,
        start_ts: str | None = None,
        end_ts: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM events WHERE 1=1"
            args: list[Any] = []
            if camera_id:
                sql += " AND camera_id=?"; args.append(camera_id)
            if event_type:
                sql += " AND event_type=?"; args.append(event_type)
            if label:
                sql += " AND lower(label)=lower(?)"; args.append(label)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            if start_ts:
                sql += " AND ts>=?"; args.append(start_ts)
            if end_ts:
                sql += " AND ts<=?"; args.append(end_ts)
            sql += " ORDER BY ts DESC LIMIT ?"; args.append(int(limit))
            return self._decode_rows(self.conn.execute(sql, args))

    def related_events(self, event_id: str) -> list[dict[str, Any]]:
        event = self.get_event(event_id)
        if not event:
            return []
        parent_id = event.get("parent_event_id") or event_id
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT * FROM events
                WHERE event_id=? OR parent_event_id=? OR event_id=(SELECT parent_event_id FROM events WHERE event_id=?)
                ORDER BY ts, event_id
                """,
                (parent_id, parent_id, event_id),
            )
            return self._decode_rows(rows)

    def get_session(self, session_id: str | None) -> dict[str, Any] | None:
        if not session_id:
            return None
        with self._lock:
            row = self.conn.execute("SELECT * FROM capture_sessions WHERE session_id=?", (session_id,)).fetchone()
            return self._decode_row(row)

    def list_artifacts(self, *, session_id: str | None = None, camera_id: str | None = None, event_id: str | None = None, artifact_type: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT a.* FROM capture_artifacts a WHERE 1=1"
            args: list[Any] = []
            if session_id:
                sql += " AND a.session_id=?"; args.append(session_id)
            if camera_id:
                sql += " AND a.camera_id=?"; args.append(camera_id)
            if event_id:
                sql += " AND a.artifact_id IN (SELECT artifact_id FROM events WHERE event_id=? OR parent_event_id=?)"
                args.extend([event_id, event_id])
            if artifact_type:
                sql += " AND a.artifact_type=?"; args.append(artifact_type)
            sql += " ORDER BY a.created_at DESC"
            return self._decode_rows(self.conn.execute(sql, args))

    def recent_audit(self, *, event_id: str | None = None, session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM audit_log WHERE 1=1"
            args: list[Any] = []
            if event_id:
                sql += " AND event_id=?"; args.append(event_id)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            sql += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
            return self._decode_rows(self.conn.execute(sql, args))

    def create_assistant_run(self, question: str, *, status: str = "pending", model_id: str | None = None, evidence: list[dict[str, Any]] | None = None) -> str:
        with self._lock:
            run_id = new_id("run")
            self.conn.execute(
                """
                INSERT INTO assistant_runs(run_id, question, status, model_id, evidence_json, created_at)
                VALUES(?,?,?,?,?,?)
                """,
                (run_id, question, status, model_id, self._json(evidence or []), now_iso()),
            )
            self.conn.commit()
            self.audit("assistant.run.create", details={"run_id": run_id, "status": status})
            return run_id

    def complete_assistant_run(self, run_id: str, *, status: str, answer: str | None = None, evidence: list[dict[str, Any]] | None = None, error: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE assistant_runs
                SET status=?, answer=?, evidence_json=?, error=?, completed_at=?
                WHERE run_id=?
                """,
                (status, answer, self._json(evidence or []), error, now_iso(), run_id),
            )
            self.conn.commit()
            self.audit("assistant.run.complete", outcome=status, details={"run_id": run_id, "error": error})

    def create_summary(self, *, run_id: str | None, camera_id: str, summary_text: str, facts: dict[str, Any], source_refs: list[dict[str, Any]], event_id: str | None = None, session_id: str | None = None, model_id: str | None = None, status: str = "completed") -> str:
        with self._lock:
            summary_id = new_id("sum")
            self.conn.execute(
                """
                INSERT INTO assistant_summaries(summary_id, run_id, event_id, session_id, camera_id,
                                                summary_text, facts_json, source_refs_json,
                                                model_id, status, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (summary_id, run_id, event_id, session_id, camera_id, summary_text, self._json(facts), self._json(source_refs), model_id, status, now_iso()),
            )
            self.conn.commit()
            self.audit("assistant.summary.create", camera_id=camera_id, event_id=event_id, session_id=session_id, details={"summary_id": summary_id})
            return summary_id


    def record_event_contract(
        self,
        *,
        event_id: str,
        camera_id: str,
        contract: dict[str, Any],
        policy_decision: dict[str, Any],
        parent_event_id: str | None = None,
        session_id: str | None = None,
        schema_version: str = "1.0",
    ) -> str:
        with self._lock:
            contract_id = new_id("ctr")
            self.conn.execute(
                """
                INSERT INTO event_contracts(contract_id, event_id, parent_event_id, session_id, camera_id,
                                            schema_version, contract_json, policy_decision_json, created_at)
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                (
                    contract_id,
                    event_id,
                    parent_event_id,
                    session_id,
                    camera_id,
                    schema_version,
                    self._json(contract),
                    self._json(policy_decision),
                    now_iso(),
                ),
            )
            self.conn.commit()
            self.audit(
                "event_contract.create",
                camera_id=camera_id,
                event_id=event_id,
                session_id=session_id,
                details={"contract_id": contract_id, "policy_action": policy_decision.get("action")},
            )
            return contract_id

    def latest_event_contract(self, event_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM event_contracts WHERE event_id=? ORDER BY created_at DESC LIMIT 1",
                (event_id,),
            ).fetchone()
            return self._decode_row(row)

    def list_event_contracts(self, *, event_id: str | None = None, session_id: str | None = None, camera_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM event_contracts WHERE 1=1"
            args: list[Any] = []
            if event_id:
                sql += " AND event_id=?"; args.append(event_id)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            if camera_id:
                sql += " AND camera_id=?"; args.append(camera_id)
            sql += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
            return self._decode_rows(self.conn.execute(sql, args))

    def list_summaries(self, *, event_id: str | None = None, session_id: str | None = None, camera_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM assistant_summaries WHERE 1=1"
            args: list[Any] = []
            if event_id:
                sql += " AND event_id=?"; args.append(event_id)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            if camera_id:
                sql += " AND camera_id=?"; args.append(camera_id)
            sql += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
            return self._decode_rows(self.conn.execute(sql, args))

    def create_vlm_analysis(
        self,
        *,
        event_id: str,
        camera_id: str,
        artifact_path: str,
        analysis: dict[str, Any],
        source_refs: list[dict[str, Any]],
        status: str = "completed",
        parent_event_id: str | None = None,
        session_id: str | None = None,
        frame_id: int | None = None,
        artifact_id: str | None = None,
        model_id: str | None = None,
        error: str | None = None,
    ) -> str:
        with self._lock:
            analysis_id = new_id("vlm")
            self.conn.execute(
                """
                INSERT INTO vlm_keyframe_analyses(analysis_id, event_id, parent_event_id, session_id,
                                                  camera_id, frame_id, artifact_id, artifact_path,
                                                  model_id, status, analysis_json, source_refs_json,
                                                  error, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    analysis_id,
                    event_id,
                    parent_event_id,
                    session_id,
                    camera_id,
                    frame_id,
                    artifact_id,
                    artifact_path,
                    model_id,
                    status,
                    self._json(analysis),
                    self._json(source_refs),
                    error,
                    now_iso(),
                ),
            )
            self.conn.commit()
            self.audit(
                "vlm.keyframe_analysis.create",
                outcome=status,
                camera_id=camera_id,
                event_id=event_id,
                session_id=session_id,
                details={"analysis_id": analysis_id, "model_id": model_id, "artifact_id": artifact_id, "error": error},
            )
            return analysis_id

    def list_vlm_analyses(
        self,
        *,
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        artifact_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM vlm_keyframe_analyses WHERE 1=1"
            args: list[Any] = []
            if event_id:
                sql += " AND event_id=?"; args.append(event_id)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            if camera_id:
                sql += " AND camera_id=?"; args.append(camera_id)
            if artifact_id:
                sql += " AND artifact_id=?"; args.append(artifact_id)
            if status:
                sql += " AND status=?"; args.append(status)
            sql += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
            return self._decode_rows(self.conn.execute(sql, args))


    def create_smolvlm2_clip_experiment(
        self,
        *,
        event_id: str,
        parent_event_id: str | None,
        session_id: str | None,
        camera_id: str,
        trigger_frame_id: int | None,
        clip_artifact_id: str | None,
        clip_path: str,
        model_id: str | None,
        status: str,
        experiment: dict[str, Any],
        source_refs: list[dict[str, Any]],
        error: str | None,
    ) -> str:
        with self._lock:
            experiment_id = new_id("svlm")
            self.conn.execute(
                """
                INSERT INTO smolvlm2_clip_experiments(experiment_id, event_id, parent_event_id, session_id,
                                                       camera_id, trigger_frame_id, clip_artifact_id, clip_path,
                                                       model_id, status, experiment_json, source_refs_json,
                                                       error, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    experiment_id,
                    event_id,
                    parent_event_id,
                    session_id,
                    camera_id,
                    trigger_frame_id,
                    clip_artifact_id,
                    clip_path,
                    model_id,
                    status,
                    self._json(experiment),
                    self._json(source_refs),
                    error,
                    now_iso(),
                ),
            )
            self.conn.commit()
            self.audit(
                "smolvlm2.short_clip_experiment.create",
                outcome=status,
                camera_id=camera_id,
                event_id=event_id,
                session_id=session_id,
                details={"experiment_id": experiment_id, "model_id": model_id, "clip_artifact_id": clip_artifact_id, "error": error},
            )
            return experiment_id

    def list_smolvlm2_clip_experiments(
        self,
        *,
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        clip_artifact_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM smolvlm2_clip_experiments WHERE 1=1"
            args: list[Any] = []
            if event_id:
                sql += " AND event_id=?"; args.append(event_id)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            if camera_id:
                sql += " AND camera_id=?"; args.append(camera_id)
            if clip_artifact_id:
                sql += " AND clip_artifact_id=?"; args.append(clip_artifact_id)
            if status:
                sql += " AND status=?"; args.append(status)
            sql += " ORDER BY created_at DESC LIMIT ?"; args.append(limit)
            return self._decode_rows(self.conn.execute(sql, args))


    def persist_evidence_pipeline_index(
        self,
        *,
        event_id: str,
        session_id: str | None,
        camera_id: str,
        manifest_artifact_id: str | None,
        profile_artifact_id: str | None,
        manifest_csv_path: str,
        profile_path: str,
        evidence: dict[str, Any],
        capture_manifest_rows: int = 0,
    ) -> str:
        """Persist a queryable facts-only evidence index for one evidence pipeline event.

        The original event row remains the compact session-level record. These
        tables make fingerprints, duplicate groups, and key moments queryable
        without reparsing a large profile artifact.
        """
        with self._lock:
            event = self.get_event(event_id)
            if not event:
                raise KeyError(f"event_id not found: {event_id}")
            now = now_iso()
            existing = self.conn.execute(
                "SELECT profile_id FROM evidence_pipeline_profiles WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if existing:
                old_profile_id = str(existing["profile_id"])
                self.conn.execute("DELETE FROM evidence_key_moments WHERE profile_id=?", (old_profile_id,))
                self.conn.execute("DELETE FROM evidence_dedup_groups WHERE profile_id=?", (old_profile_id,))
                self.conn.execute("DELETE FROM evidence_fingerprints WHERE profile_id=?", (old_profile_id,))
                self.conn.execute("DELETE FROM evidence_pipeline_profiles WHERE profile_id=?", (old_profile_id,))
            profile_id = new_id("eidx")
            safety = evidence.get("safety") if isinstance(evidence.get("safety"), dict) else {}
            latency = evidence.get("latency") if isinstance(evidence.get("latency"), dict) else {}
            timeline = evidence.get("timeline") if isinstance(evidence.get("timeline"), dict) else {}
            self.conn.execute(
                """
                INSERT INTO evidence_pipeline_profiles(
                    profile_id, event_id, session_id, camera_id, manifest_artifact_id, profile_artifact_id,
                    manifest_csv_path, profile_path, native_schema, capture_manifest_rows, fingerprint_count,
                    media_fingerprint_count, synthetic_fingerprint_count, real_media_ingestion,
                    duplicate_group_count, duplicate_clip_count, unique_clip_count, key_moment_count,
                    planned_read_bytes, total_manifest_bytes, safety_ok, violation_count, facts_only,
                    timeline_json, latency_json, safety_json, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    profile_id,
                    event_id,
                    session_id,
                    camera_id,
                    manifest_artifact_id,
                    profile_artifact_id,
                    manifest_csv_path,
                    profile_path,
                    str(evidence.get("schema") or ""),
                    int(capture_manifest_rows or evidence.get("manifest_entries") or 0),
                    int(evidence.get("fingerprint_count") or 0),
                    int(evidence.get("media_fingerprint_count") or 0),
                    int(evidence.get("synthetic_fingerprint_count") or 0),
                    1 if evidence.get("real_media_ingestion") else 0,
                    int(evidence.get("duplicate_group_count") or 0),
                    int(evidence.get("duplicate_clip_count") or 0),
                    int(evidence.get("unique_clip_count") or 0),
                    int(evidence.get("key_moment_count") or 0),
                    int(evidence.get("planned_read_bytes") or 0),
                    int(evidence.get("total_manifest_bytes") or 0),
                    1 if safety.get("ok") else 0,
                    int(safety.get("violation_count") or 0),
                    1 if evidence.get("facts_only", True) else 0,
                    self._json(timeline),
                    self._json(latency),
                    self._json(safety),
                    now,
                ),
            )
            for item in evidence.get("fingerprints") or []:
                if not isinstance(item, dict):
                    continue
                hist = item.get("histogram16") or item.get("histogram") or []
                self.conn.execute(
                    """
                    INSERT INTO evidence_fingerprints(
                        fingerprint_id, profile_id, event_id, session_id, camera_id, clip_id, clip_index,
                        path, start_ms, duration_ms, from_media, fingerprint_source, decoded_width,
                        decoded_height, ahash64, dhash64, fingerprint64, fingerprint_hex, histogram_json,
                        histogram_bins, duplicate_group, duplicate_of, nearest_hamming, fingerprint_score,
                        created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        new_id("efp"),
                        profile_id,
                        event_id,
                        session_id,
                        camera_id,
                        str(item.get("clip_id") or ""),
                        int(item.get("clip_index") if item.get("clip_index") is not None else -1),
                        str(item.get("path") or ""),
                        int(item.get("start_ms") or 0),
                        int(item.get("duration_ms") or 0),
                        1 if item.get("from_media") else 0,
                        str(item.get("fingerprint_source") or ("decoded_keyframe" if item.get("from_media") else "metadata_synthetic")),
                        int(item.get("decoded_width") or 0),
                        int(item.get("decoded_height") or 0),
                        str(item.get("ahash64") or ""),
                        str(item.get("dhash64") or ""),
                        str(item.get("fingerprint64") or ""),
                        str(item.get("fingerprint_hex") or ""),
                        self._json(hist if isinstance(hist, list) else []),
                        int(item.get("histogram_bins") or (len(hist) if isinstance(hist, list) else 0)),
                        int(item.get("duplicate_group") if item.get("duplicate_group") is not None else -1),
                        int(item.get("duplicate_of") if item.get("duplicate_of") is not None else -1),
                        int(item.get("nearest_hamming") if item.get("nearest_hamming") is not None else -1),
                        float(item.get("fingerprint_score") or 0.0),
                        now,
                    ),
                )
            for group in evidence.get("duplicate_groups") or []:
                if not isinstance(group, dict):
                    continue
                self.conn.execute(
                    """
                    INSERT INTO evidence_dedup_groups(
                        dedup_id, profile_id, event_id, session_id, camera_id, group_id,
                        representative_clip_id, representative_clip_index, group_size, duplicate_count,
                        min_hamming, max_hamming, clip_ids_json, clip_indices_json, created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        new_id("edup"),
                        profile_id,
                        event_id,
                        session_id,
                        camera_id,
                        int(group.get("group_id") or 0),
                        str(group.get("representative_clip_id") or ""),
                        int(group.get("representative_clip_index") if group.get("representative_clip_index") is not None else -1),
                        int(group.get("group_size") or 0),
                        int(group.get("duplicate_count") or 0),
                        int(group.get("min_hamming") if group.get("min_hamming") is not None else -1),
                        int(group.get("max_hamming") if group.get("max_hamming") is not None else -1),
                        self._json(group.get("clip_ids") if isinstance(group.get("clip_ids"), list) else []),
                        self._json(group.get("clip_indices") if isinstance(group.get("clip_indices"), list) else []),
                        now,
                    ),
                )
            by_clip_index: dict[int, dict[str, Any]] = {}
            for item in evidence.get("fingerprints") or []:
                if isinstance(item, dict):
                    try:
                        by_clip_index[int(item.get("clip_index") or -1)] = item
                    except (TypeError, ValueError):
                        pass
            for item in evidence.get("key_moments") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    clip_index = int(item.get("clip_index") if item.get("clip_index") is not None else -1)
                except (TypeError, ValueError):
                    clip_index = -1
                fp = by_clip_index.get(clip_index, {})
                self.conn.execute(
                    """
                    INSERT INTO evidence_key_moments(
                        key_moment_id, profile_id, event_id, session_id, camera_id, rank, clip_id,
                        clip_index, start_ms, duration_ms, reason, priority_score, motion_score,
                        audio_score, lighting_delta, changed_pixels, duplicate_group, created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        new_id("ekm"),
                        profile_id,
                        event_id,
                        session_id,
                        camera_id,
                        int(item.get("rank") or 0),
                        str(item.get("clip_id") or ""),
                        clip_index,
                        int(item.get("start_ms") or 0),
                        int(item.get("duration_ms") or 0),
                        str(item.get("reason") or ""),
                        float(item.get("priority_score") or 0.0),
                        float(item.get("motion_score") or 0.0),
                        float(item.get("audio_score") or 0.0),
                        float(item.get("lighting_delta") or 0.0),
                        int(item.get("changed_pixels") or 0),
                        int(fp.get("duplicate_group") if fp.get("duplicate_group") is not None else -1),
                        now,
                    ),
                )
            self.conn.commit()
            self.audit(
                "evidence_index.persist",
                camera_id=camera_id,
                event_id=event_id,
                session_id=session_id,
                details={
                    "profile_id": profile_id,
                    "fingerprint_count": int(evidence.get("fingerprint_count") or 0),
                    "duplicate_group_count": int(evidence.get("duplicate_group_count") or 0),
                    "key_moment_count": int(evidence.get("key_moment_count") or 0),
                    "facts_only": True,
                },
            )
            return profile_id

    def list_evidence_profiles(
        self,
        *,
        profile_id: str | None = None,
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM evidence_pipeline_profiles WHERE 1=1"
            args: list[Any] = []
            if profile_id:
                sql += " AND profile_id=?"; args.append(profile_id)
            if event_id:
                sql += " AND event_id=?"; args.append(event_id)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            if camera_id:
                sql += " AND camera_id=?"; args.append(camera_id)
            sql += " ORDER BY created_at DESC LIMIT ?"; args.append(int(limit))
            return self._decode_rows(self.conn.execute(sql, args))

    def get_evidence_profile(self, profile_id: str) -> dict[str, Any] | None:
        rows = self.list_evidence_profiles(profile_id=profile_id, limit=1)
        return rows[0] if rows else None

    @staticmethod
    def _evidence_profile_summary(profile: dict[str, Any], *, detailed: bool = False) -> dict[str, Any]:
        safety = profile.get("safety") if isinstance(profile.get("safety"), dict) else {}
        timeline = profile.get("timeline") if isinstance(profile.get("timeline"), dict) else {}
        latency = profile.get("latency") if isinstance(profile.get("latency"), dict) else {}
        return {
            "ok": bool(profile.get("safety_ok")) and int(profile.get("violation_count") or 0) == 0,
            "schema": "monitorme.evidence_pipeline_summary.v0.1",
            "profile_id": profile.get("profile_id"),
            "event_id": profile.get("event_id"),
            "session_id": profile.get("session_id"),
            "camera_id": profile.get("camera_id"),
            "created_at": profile.get("created_at"),
            "native_schema": profile.get("native_schema"),
            "artifact_refs": {
                "manifest_artifact_id": profile.get("manifest_artifact_id"),
                "profile_artifact_id": profile.get("profile_artifact_id"),
                "manifest_csv_path": profile.get("manifest_csv_path"),
                "profile_path": profile.get("profile_path"),
            },
            "counts": {
                "capture_manifest_rows": int(profile.get("capture_manifest_rows") or 0),
                "fingerprint_count": int(profile.get("fingerprint_count") or 0),
                "media_fingerprint_count": int(profile.get("media_fingerprint_count") or 0),
                "synthetic_fingerprint_count": int(profile.get("synthetic_fingerprint_count") or 0),
                "duplicate_group_count": int(profile.get("duplicate_group_count") or 0),
                "duplicate_clip_count": int(profile.get("duplicate_clip_count") or 0),
                "unique_clip_count": int(profile.get("unique_clip_count") or 0),
                "key_moment_count": int(profile.get("key_moment_count") or 0),
            },
            "storage": {
                "planned_read_bytes": int(profile.get("planned_read_bytes") or 0),
                "total_manifest_bytes": int(profile.get("total_manifest_bytes") or 0),
            },
            "ingestion": {
                "real_media_ingestion": bool(profile.get("real_media_ingestion")),
                "facts_only": bool(profile.get("facts_only", 1)),
                "safety_ok": bool(profile.get("safety_ok")),
                "violation_count": int(profile.get("violation_count") or 0),
            },
            "timeline": timeline if detailed else {
                "timeline_start_ms": timeline.get("timeline_start_ms"),
                "timeline_end_ms": timeline.get("timeline_end_ms"),
                "timeline_span_ms": timeline.get("timeline_span_ms"),
                "covered_duration_ms": timeline.get("covered_duration_ms"),
                "max_gap_ms": timeline.get("max_gap_ms"),
                "clip_count": timeline.get("clip_count"),
            },
            "latency": latency if detailed else {
                "total_ms": latency.get("total_ms"),
                "fingerprint_ms": latency.get("fingerprint_ms"),
                "dedup_ms": latency.get("dedup_ms"),
                "key_selection_ms": latency.get("key_selection_ms"),
                "planned_read_mb_per_s": latency.get("planned_read_mb_per_s"),
            },
            "safety": safety if detailed else {
                "ok": safety.get("ok"),
                "violation_count": safety.get("violation_count"),
                "no_semantic_claims": safety.get("no_semantic_claims"),
                "facts_only": safety.get("facts_only"),
            },
            "privacy": {
                "external_upload": False,
                "raw_frame_upload": False,
                "media_decode_in_api": False,
                "identity": False,
                "intent": False,
                "speech_content": False,
                "semantic_claims": False,
            },
        }

    def summarize_evidence_pipeline_profiles(
        self,
        *,
        profile_id: str | None = None,
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 100,
        detailed: bool = False,
    ) -> list[dict[str, Any]]:
        profiles = self.list_evidence_profiles(
            profile_id=profile_id, event_id=event_id, session_id=session_id, camera_id=camera_id, limit=limit
        )
        return [self._evidence_profile_summary(profile, detailed=detailed) for profile in profiles]

    def get_evidence_pipeline_summary(
        self,
        profile_id: str,
        *,
        include_fingerprints: bool = False,
        include_dedup_groups: bool = True,
        include_key_moments: bool = True,
        fingerprint_limit: int = 20,
        detailed: bool = True,
    ) -> dict[str, Any] | None:
        profile = self.get_evidence_profile(profile_id)
        if not profile:
            return None
        summary = self._evidence_profile_summary(profile, detailed=detailed)
        if include_key_moments:
            summary["key_moments"] = self.list_evidence_key_moments(profile_id=profile_id, limit=100)
        if include_dedup_groups:
            summary["dedup_groups"] = self.list_evidence_dedup_groups(profile_id=profile_id, limit=100)
        if include_fingerprints:
            summary["fingerprints"] = self.list_evidence_fingerprints(profile_id=profile_id, limit=fingerprint_limit)
            summary["fingerprints_truncated"] = int(profile.get("fingerprint_count") or 0) > int(fingerprint_limit)
        return summary

    def list_evidence_fingerprints(
        self,
        *,
        profile_id: str | None = None,
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        from_media: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM evidence_fingerprints WHERE 1=1"
            args: list[Any] = []
            if profile_id:
                sql += " AND profile_id=?"; args.append(profile_id)
            if event_id:
                sql += " AND event_id=?"; args.append(event_id)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            if camera_id:
                sql += " AND camera_id=?"; args.append(camera_id)
            if from_media is not None:
                sql += " AND from_media=?"; args.append(1 if from_media else 0)
            sql += " ORDER BY profile_id, clip_index LIMIT ?"; args.append(int(limit))
            return self._decode_rows(self.conn.execute(sql, args))

    def list_evidence_dedup_groups(
        self,
        *,
        profile_id: str | None = None,
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM evidence_dedup_groups WHERE 1=1"
            args: list[Any] = []
            if profile_id:
                sql += " AND profile_id=?"; args.append(profile_id)
            if event_id:
                sql += " AND event_id=?"; args.append(event_id)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            if camera_id:
                sql += " AND camera_id=?"; args.append(camera_id)
            sql += " ORDER BY profile_id, group_id LIMIT ?"; args.append(int(limit))
            return self._decode_rows(self.conn.execute(sql, args))

    def list_evidence_key_moments(
        self,
        *,
        profile_id: str | None = None,
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM evidence_key_moments WHERE 1=1"
            args: list[Any] = []
            if profile_id:
                sql += " AND profile_id=?"; args.append(profile_id)
            if event_id:
                sql += " AND event_id=?"; args.append(event_id)
            if session_id:
                sql += " AND session_id=?"; args.append(session_id)
            if camera_id:
                sql += " AND camera_id=?"; args.append(camera_id)
            sql += " ORDER BY profile_id, rank LIMIT ?"; args.append(int(limit))
            return self._decode_rows(self.conn.execute(sql, args))

    @staticmethod
    def _parse_iso_datetime(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(str(value))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _db_total_size_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            try:
                if path.exists() and path.is_file():
                    total += int(path.stat().st_size)
            except OSError:
                pass
        return total

    def _selected_evidence_index_counts(self, profile_ids: list[str]) -> dict[str, int]:
        if not profile_ids:
            return {
                "profiles": 0,
                "fingerprints": 0,
                "dedup_groups": 0,
                "key_moments": 0,
                "index_payload_bytes_estimate": 0,
            }
        placeholders = ",".join("?" for _ in profile_ids)
        fp = self.conn.execute(
            f"SELECT COUNT(*) AS c, COALESCE(SUM(LENGTH(COALESCE(histogram_json,'')) + LENGTH(COALESCE(ahash64,'')) + LENGTH(COALESCE(dhash64,'')) + LENGTH(COALESCE(fingerprint64,'')) + LENGTH(COALESCE(fingerprint_hex,''))),0) AS b FROM evidence_fingerprints WHERE profile_id IN ({placeholders})",
            profile_ids,
        ).fetchone()
        dg = self.conn.execute(
            f"SELECT COUNT(*) AS c, COALESCE(SUM(LENGTH(COALESCE(clip_ids_json,'')) + LENGTH(COALESCE(clip_indices_json,''))),0) AS b FROM evidence_dedup_groups WHERE profile_id IN ({placeholders})",
            profile_ids,
        ).fetchone()
        km = self.conn.execute(
            f"SELECT COUNT(*) AS c FROM evidence_key_moments WHERE profile_id IN ({placeholders})",
            profile_ids,
        ).fetchone()
        prof = self.conn.execute(
            f"SELECT COUNT(*) AS c, COALESCE(SUM(LENGTH(COALESCE(timeline_json,'')) + LENGTH(COALESCE(latency_json,'')) + LENGTH(COALESCE(safety_json,''))),0) AS b FROM evidence_pipeline_profiles WHERE profile_id IN ({placeholders})",
            profile_ids,
        ).fetchone()
        return {
            "profiles": int(prof["c"] or 0),
            "fingerprints": int(fp["c"] or 0),
            "dedup_groups": int(dg["c"] or 0),
            "key_moments": int(km["c"] or 0),
            "index_payload_bytes_estimate": int(prof["b"] or 0) + int(fp["b"] or 0) + int(dg["b"] or 0),
        }

    def plan_evidence_index_retention(
        self,
        *,
        older_than_days: int | None = None,
        keep_last_per_camera: int = 1,
        keep_last_per_session: int = 1,
        profile_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 1000,
        now: str | None = None,
    ) -> dict[str, Any]:
        """Plan a facts-only evidence-index retention operation without deleting data.

        The policy prunes only normalized evidence index rows. Source events,
        capture artifacts, keyframe files, manifests, and profile JSON artifacts
        remain intact and auditable.
        """
        with self._lock:
            safe_limit = max(1, min(int(limit), 100000))
            keep_last_per_camera = max(0, int(keep_last_per_camera))
            keep_last_per_session = max(0, int(keep_last_per_session))
            now_dt = self._parse_iso_datetime(now) or datetime.now(timezone.utc).astimezone()
            cutoff_dt: datetime | None = None
            cutoff_at: str | None = None
            if older_than_days is not None:
                cutoff_dt = now_dt - timedelta(days=max(0, int(older_than_days)))
                cutoff_at = cutoff_dt.isoformat(timespec="seconds")

            profiles = self.list_evidence_profiles(
                profile_id=profile_id,
                session_id=session_id,
                camera_id=camera_id,
                limit=safe_limit,
            )
            by_camera: dict[str, int] = {}
            by_session: dict[str, int] = {}
            protected: set[str] = set()
            for profile in profiles:
                pid = str(profile.get("profile_id") or "")
                cam = str(profile.get("camera_id") or "")
                sess = str(profile.get("session_id") or "")
                if cam:
                    seen = by_camera.get(cam, 0)
                    if seen < keep_last_per_camera:
                        protected.add(pid)
                    by_camera[cam] = seen + 1
                if sess:
                    seen = by_session.get(sess, 0)
                    if seen < keep_last_per_session:
                        protected.add(pid)
                    by_session[sess] = seen + 1

            selected: list[dict[str, Any]] = []
            explicit_scope_without_age = bool(profile_id or session_id) and cutoff_dt is None
            for profile in profiles:
                pid = str(profile.get("profile_id") or "")
                created_dt = self._parse_iso_datetime(str(profile.get("created_at") or ""))
                age_match = bool(cutoff_dt and created_dt and created_dt < cutoff_dt)
                explicit_match = bool(explicit_scope_without_age)
                if (age_match or explicit_match) and pid not in protected:
                    selected.append(profile)

            profile_ids = [str(p.get("profile_id")) for p in selected if p.get("profile_id")]
            counts = self._selected_evidence_index_counts(profile_ids)
            selected_summaries = [
                {
                    "profile_id": p.get("profile_id"),
                    "event_id": p.get("event_id"),
                    "session_id": p.get("session_id"),
                    "camera_id": p.get("camera_id"),
                    "created_at": p.get("created_at"),
                    "fingerprint_count": int(p.get("fingerprint_count") or 0),
                    "key_moment_count": int(p.get("key_moment_count") or 0),
                    "facts_only": bool(p.get("facts_only", 1)),
                    "safety_ok": bool(p.get("safety_ok")),
                }
                for p in selected
            ]
            return {
                "ok": True,
                "schema": "monitorme.evidence_index_retention_plan.v0.1",
                "policy": {
                    "older_than_days": older_than_days,
                    "cutoff_at": cutoff_at,
                    "keep_last_per_camera": keep_last_per_camera,
                    "keep_last_per_session": keep_last_per_session,
                    "profile_id": profile_id,
                    "session_id": session_id,
                    "camera_id": camera_id,
                    "limit": safe_limit,
                    "delete_scope": "evidence_index_rows_only",
                },
                "profiles_scanned": len(profiles),
                "profiles_selected": len(profile_ids),
                "profile_ids": profile_ids,
                "selected_profiles": selected_summaries,
                "rows_selected": counts,
                "index_payload_bytes_estimate": counts["index_payload_bytes_estimate"],
                "retains_source_events": True,
                "retains_capture_artifacts": True,
                "retains_keyframe_files": True,
                "facts_only": True,
                "privacy": {
                    "external_upload": False,
                    "raw_frame_upload": False,
                    "media_decode": False,
                    "semantic_claims": False,
                },
            }

    def apply_evidence_index_retention(
        self,
        *,
        dry_run: bool = True,
        older_than_days: int | None = None,
        keep_last_per_camera: int = 1,
        keep_last_per_session: int = 1,
        profile_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 1000,
        compact: bool = True,
        vacuum: bool = False,
        now: str | None = None,
    ) -> dict[str, Any]:
        """Apply or dry-run an evidence-index retention policy.

        Deletion is intentionally limited to evidence_pipeline_profiles and its
        child evidence index tables. It does not remove source events, artifact
        rows, keyframe files, manifests, or JSON profile artifacts.
        """
        with self._lock:
            plan = self.plan_evidence_index_retention(
                older_than_days=older_than_days,
                keep_last_per_camera=keep_last_per_camera,
                keep_last_per_session=keep_last_per_session,
                profile_id=profile_id,
                session_id=session_id,
                camera_id=camera_id,
                limit=limit,
                now=now,
            )
            run_id = new_id("eret")
            now = now_iso()
            db_size_before = self._db_total_size_bytes()
            profile_ids = list(plan.get("profile_ids") or [])
            status = "dry_run" if dry_run else "completed"
            error: str | None = None
            wal_checkpoint = 0
            vacuum_completed = 0
            try:
                if not dry_run and profile_ids:
                    placeholders = ",".join("?" for _ in profile_ids)
                    self.conn.execute(
                        f"DELETE FROM evidence_pipeline_profiles WHERE profile_id IN ({placeholders})",
                        profile_ids,
                    )
                    self.conn.commit()
                if compact:
                    try:
                        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                        wal_checkpoint = 1
                    except sqlite3.DatabaseError:
                        wal_checkpoint = 0
                if (not dry_run) and vacuum:
                    try:
                        self.conn.execute("VACUUM")
                        vacuum_completed = 1
                    except sqlite3.DatabaseError as exc:
                        error = str(exc)
                        status = "completed_with_compaction_error"
            except Exception as exc:  # pragma: no cover - defensive status capture
                self.conn.rollback()
                status = "error"
                error = str(exc)
            db_size_after = self._db_total_size_bytes()
            row_counts = plan.get("rows_selected") or {}
            self.conn.execute(
                """
                INSERT INTO evidence_retention_runs(
                    run_id, dry_run, status, policy_json, cutoff_at, profiles_scanned,
                    profiles_selected, fingerprints_selected, dedup_groups_selected,
                    key_moments_selected, index_payload_bytes_estimate, db_size_before_bytes,
                    db_size_after_bytes, wal_checkpoint, vacuum_requested, vacuum_completed,
                    selected_profiles_json, error, created_at, completed_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    run_id,
                    1 if dry_run else 0,
                    status,
                    self._json(plan.get("policy") or {}),
                    (plan.get("policy") or {}).get("cutoff_at"),
                    int(plan.get("profiles_scanned") or 0),
                    int(plan.get("profiles_selected") or 0),
                    int(row_counts.get("fingerprints") or 0),
                    int(row_counts.get("dedup_groups") or 0),
                    int(row_counts.get("key_moments") or 0),
                    int(plan.get("index_payload_bytes_estimate") or 0),
                    db_size_before,
                    db_size_after,
                    wal_checkpoint,
                    1 if vacuum else 0,
                    vacuum_completed,
                    self._json(plan.get("selected_profiles") or []),
                    error,
                    now,
                    now_iso(),
                ),
            )
            self.conn.commit()
            result = {
                "ok": status != "error",
                "schema": "monitorme.evidence_index_retention_result.v0.1",
                "run_id": run_id,
                "dry_run": bool(dry_run),
                "status": status,
                "plan": plan,
                "deleted": {
                    "profiles": 0 if dry_run else int(plan.get("profiles_selected") or 0),
                    "fingerprints": 0 if dry_run else int(row_counts.get("fingerprints") or 0),
                    "dedup_groups": 0 if dry_run else int(row_counts.get("dedup_groups") or 0),
                    "key_moments": 0 if dry_run else int(row_counts.get("key_moments") or 0),
                },
                "compaction": {
                    "compact_requested": bool(compact),
                    "wal_checkpoint": bool(wal_checkpoint),
                    "vacuum_requested": bool(vacuum),
                    "vacuum_completed": bool(vacuum_completed),
                    "db_size_before_bytes": db_size_before,
                    "db_size_after_bytes": db_size_after,
                },
                "retains_source_events": True,
                "retains_capture_artifacts": True,
                "retains_keyframe_files": True,
                "error": error,
            }
            self.audit(
                "evidence_index.retention",
                outcome=status,
                camera_id=camera_id,
                session_id=session_id,
                details={
                    "run_id": run_id,
                    "dry_run": bool(dry_run),
                    "profiles_selected": int(plan.get("profiles_selected") or 0),
                    "profiles_deleted": result["deleted"]["profiles"],
                    "delete_scope": "evidence_index_rows_only",
                },
            )
            return result

    def list_evidence_retention_runs(
        self,
        *,
        run_id: str | None = None,
        dry_run: bool | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM evidence_retention_runs WHERE 1=1"
            args: list[Any] = []
            if run_id:
                sql += " AND run_id=?"; args.append(run_id)
            if dry_run is not None:
                sql += " AND dry_run=?"; args.append(1 if dry_run else 0)
            if status:
                sql += " AND status=?"; args.append(status)
            sql += " ORDER BY created_at DESC LIMIT ?"; args.append(max(1, min(int(limit), 1000)))
            return self._decode_rows(self.conn.execute(sql, args))


    @staticmethod
    def _retention_schedule_delta(cadence: str) -> timedelta | None:
        normalized = str(cadence or "daily").strip().lower()
        if normalized == "hourly":
            return timedelta(hours=1)
        if normalized == "daily":
            return timedelta(days=1)
        if normalized == "weekly":
            return timedelta(days=7)
        if normalized == "monthly":
            return timedelta(days=30)
        if normalized == "manual":
            return None
        raise ValueError("cadence must be one of: hourly, daily, weekly, monthly, manual")

    @staticmethod
    def _retention_schedule_policy_from_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "older_than_days": row.get("older_than_days"),
            "keep_last_per_camera": int(row.get("keep_last_per_camera") or 0),
            "keep_last_per_session": int(row.get("keep_last_per_session") or 0),
            "profile_id": row.get("profile_id"),
            "session_id": row.get("session_id"),
            "camera_id": row.get("camera_id"),
            "limit": int(row.get("limit_profiles") or 1000),
        }

    def get_evidence_retention_schedule(self, schedule_id: str = "default") -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM evidence_retention_schedule WHERE schedule_id=?",
                (schedule_id,),
            ).fetchone()
            data = self._decode_row(row)
            if not data:
                return None
            data["policy"] = self._retention_schedule_policy_from_row(data)
            data["privacy"] = {
                "facts_only": True,
                "external_upload": False,
                "raw_frame_upload": False,
                "media_decode": False,
                "semantic_claims": False,
                "destructive_apply_requires_explicit_configuration": True,
            }
            data["schema"] = "monitorme.evidence_index_retention_schedule.v0.1"
            return data

    def configure_evidence_retention_schedule(
        self,
        *,
        schedule_id: str = "default",
        enabled: bool | None = None,
        cadence: str | None = None,
        older_than_days: int | None = None,
        keep_last_per_camera: int | None = None,
        keep_last_per_session: int | None = None,
        profile_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int | None = None,
        dry_run: bool | None = None,
        compact: bool | None = None,
        vacuum: bool | None = None,
        next_run_after: str | None = None,
        notes: str | None = None,
        allow_destructive: bool = False,
    ) -> dict[str, Any]:
        """Create or update the scheduled evidence-index retention policy.

        Scheduled retention defaults to dry-run. Setting dry_run=False requires
        allow_destructive=True from the CLI/API confirmation layer.
        """
        with self._lock:
            current = self.get_evidence_retention_schedule(schedule_id) or {}
            now = now_iso()
            new_cadence = str(cadence if cadence is not None else current.get("cadence", "daily")).strip().lower()
            self._retention_schedule_delta(new_cadence)
            new_dry_run = bool(current.get("dry_run", 1)) if dry_run is None else bool(dry_run)
            if not new_dry_run and not allow_destructive:
                raise ValueError("Scheduling destructive retention requires explicit confirmation")
            # Validate next_run_after when supplied but keep NULL valid for manual/first-run schedules.
            if next_run_after:
                parsed = self._parse_iso_datetime(next_run_after)
                if parsed is None:
                    raise ValueError("next_run_after must be ISO-8601 when provided")
                next_run_after = parsed.isoformat(timespec="seconds")
            values = {
                "schedule_id": schedule_id,
                "enabled": int(bool(current.get("enabled", 0)) if enabled is None else bool(enabled)),
                "cadence": new_cadence,
                "older_than_days": current.get("older_than_days", 30) if older_than_days is None else max(0, int(older_than_days)),
                "keep_last_per_camera": int(current.get("keep_last_per_camera", 1) if keep_last_per_camera is None else max(0, int(keep_last_per_camera))),
                "keep_last_per_session": int(current.get("keep_last_per_session", 1) if keep_last_per_session is None else max(0, int(keep_last_per_session))),
                "profile_id": current.get("profile_id") if profile_id is None else profile_id,
                "session_id": current.get("session_id") if session_id is None else session_id,
                "camera_id": current.get("camera_id") if camera_id is None else camera_id,
                "limit_profiles": int(current.get("limit_profiles", 1000) if limit is None else max(1, min(int(limit), 100000))),
                "dry_run": int(new_dry_run),
                "compact": int(bool(current.get("compact", 1)) if compact is None else bool(compact)),
                "vacuum": int(bool(current.get("vacuum", 0)) if vacuum is None else bool(vacuum)),
                "next_run_after": current.get("next_run_after") if next_run_after is None else next_run_after,
                "notes": str(current.get("notes", "") if notes is None else notes),
                "created_at": current.get("created_at") or now,
                "updated_at": now,
            }
            self.conn.execute(
                """
                INSERT INTO evidence_retention_schedule(
                    schedule_id, enabled, cadence, older_than_days, keep_last_per_camera,
                    keep_last_per_session, profile_id, session_id, camera_id, limit_profiles,
                    dry_run, compact, vacuum, next_run_after, notes, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(schedule_id) DO UPDATE SET
                    enabled=excluded.enabled,
                    cadence=excluded.cadence,
                    older_than_days=excluded.older_than_days,
                    keep_last_per_camera=excluded.keep_last_per_camera,
                    keep_last_per_session=excluded.keep_last_per_session,
                    profile_id=excluded.profile_id,
                    session_id=excluded.session_id,
                    camera_id=excluded.camera_id,
                    limit_profiles=excluded.limit_profiles,
                    dry_run=excluded.dry_run,
                    compact=excluded.compact,
                    vacuum=excluded.vacuum,
                    next_run_after=excluded.next_run_after,
                    notes=excluded.notes,
                    updated_at=excluded.updated_at
                """,
                (
                    values["schedule_id"], values["enabled"], values["cadence"], values["older_than_days"],
                    values["keep_last_per_camera"], values["keep_last_per_session"], values["profile_id"],
                    values["session_id"], values["camera_id"], values["limit_profiles"], values["dry_run"],
                    values["compact"], values["vacuum"], values["next_run_after"], values["notes"],
                    values["created_at"], values["updated_at"],
                ),
            )
            self.conn.commit()
            self.audit(
                "evidence_retention.schedule.configure",
                outcome="ok",
                details={
                    "schedule_id": schedule_id,
                    "enabled": bool(values["enabled"]),
                    "cadence": values["cadence"],
                    "dry_run": bool(values["dry_run"]),
                    "compact": bool(values["compact"]),
                    "vacuum": bool(values["vacuum"]),
                },
            )
            return self.get_evidence_retention_schedule(schedule_id) or {}

    def _insert_evidence_retention_scheduler_run(
        self,
        *,
        scheduler_run_id: str,
        schedule_id: str,
        forced: bool,
        due: bool,
        status: str,
        reason: str,
        retention_run_id: str | None,
        policy: dict[str, Any],
        dry_run: bool,
        compact: bool,
        vacuum: bool,
        checked_at: str,
        next_run_after: str | None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO evidence_retention_scheduler_runs(
                scheduler_run_id, schedule_id, forced, due, status, reason,
                retention_run_id, policy_json, dry_run, compact, vacuum,
                checked_at, next_run_after, result_json, error, completed_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                scheduler_run_id,
                schedule_id,
                1 if forced else 0,
                1 if due else 0,
                status,
                reason,
                retention_run_id,
                self._json(policy),
                1 if dry_run else 0,
                1 if compact else 0,
                1 if vacuum else 0,
                checked_at,
                next_run_after,
                self._json(result or {}),
                error,
                now_iso(),
            ),
        )

    def run_evidence_retention_schedule(
        self,
        *,
        schedule_id: str = "default",
        force: bool = False,
        now: str | None = None,
        dry_run_override: bool | None = None,
        allow_destructive: bool = False,
    ) -> dict[str, Any]:
        """Run the scheduled evidence-index retention policy when it is due.

        The scheduler executes no media decode and touches only evidence index rows
        through the Phase 14 retention apply path.
        """
        with self._lock:
            schedule = self.get_evidence_retention_schedule(schedule_id)
            checked_dt = self._parse_iso_datetime(now) or datetime.now(timezone.utc).astimezone()
            checked_at = checked_dt.isoformat(timespec="seconds")
            if not schedule:
                schedule = self.configure_evidence_retention_schedule(schedule_id=schedule_id)
            scheduler_run_id = new_id("ers")
            policy = self._retention_schedule_policy_from_row(schedule)
            dry_run = bool(schedule.get("dry_run", 1)) if dry_run_override is None else bool(dry_run_override)
            compact = bool(schedule.get("compact", 1))
            vacuum = bool(schedule.get("vacuum", 0))
            if not dry_run and not allow_destructive:
                raise ValueError("Running destructive scheduled retention requires explicit confirmation")
            enabled = bool(schedule.get("enabled"))
            next_run_after = schedule.get("next_run_after")
            due_at = self._parse_iso_datetime(str(next_run_after)) if next_run_after else None
            due = bool(force or (enabled and (due_at is None or checked_dt >= due_at)))
            reason = "forced" if force else "due" if due else "schedule_disabled" if not enabled else "not_due"
            if not due:
                self.conn.execute(
                    "UPDATE evidence_retention_schedule SET last_checked_at=?, updated_at=? WHERE schedule_id=?",
                    (checked_at, checked_at, schedule_id),
                )
                result = {
                    "ok": True,
                    "schema": "monitorme.evidence_index_retention_scheduler_result.v0.1",
                    "scheduler_run_id": scheduler_run_id,
                    "schedule_id": schedule_id,
                    "status": "skipped",
                    "reason": reason,
                    "due": False,
                    "forced": bool(force),
                    "retention_run_id": None,
                    "schedule": schedule,
                    "privacy": {
                        "facts_only": True,
                        "external_upload": False,
                        "raw_frame_upload": False,
                        "media_decode": False,
                        "semantic_claims": False,
                    },
                }
                self._insert_evidence_retention_scheduler_run(
                    scheduler_run_id=scheduler_run_id,
                    schedule_id=schedule_id,
                    forced=force,
                    due=False,
                    status="skipped",
                    reason=reason,
                    retention_run_id=None,
                    policy=policy,
                    dry_run=dry_run,
                    compact=compact,
                    vacuum=vacuum,
                    checked_at=checked_at,
                    next_run_after=next_run_after,
                    result=result,
                )
                self.conn.commit()
                return result

            cadence_delta = self._retention_schedule_delta(str(schedule.get("cadence") or "daily"))
            computed_next = (checked_dt + cadence_delta).isoformat(timespec="seconds") if cadence_delta else None
            try:
                retention = self.apply_evidence_index_retention(
                    dry_run=dry_run,
                    compact=compact,
                    vacuum=vacuum,
                    older_than_days=policy.get("older_than_days"),
                    keep_last_per_camera=int(policy.get("keep_last_per_camera") or 0),
                    keep_last_per_session=int(policy.get("keep_last_per_session") or 0),
                    profile_id=policy.get("profile_id"),
                    session_id=policy.get("session_id"),
                    camera_id=policy.get("camera_id"),
                    limit=int(policy.get("limit") or 1000),
                    now=checked_at,
                )
                status = "dry_run" if dry_run else "completed"
                retention_run_id = str(retention.get("run_id") or "") or None
                self.conn.execute(
                    """
                    UPDATE evidence_retention_schedule
                    SET last_checked_at=?, last_run_at=?, last_retention_run_id=?, next_run_after=?, updated_at=?
                    WHERE schedule_id=?
                    """,
                    (checked_at, checked_at, retention_run_id, computed_next, checked_at, schedule_id),
                )
                result = {
                    "ok": True,
                    "schema": "monitorme.evidence_index_retention_scheduler_result.v0.1",
                    "scheduler_run_id": scheduler_run_id,
                    "schedule_id": schedule_id,
                    "status": status,
                    "reason": reason,
                    "due": True,
                    "forced": bool(force),
                    "retention_run_id": retention_run_id,
                    "next_run_after": computed_next,
                    "retention_result": retention,
                    "privacy": {
                        "facts_only": True,
                        "external_upload": False,
                        "raw_frame_upload": False,
                        "media_decode": False,
                        "semantic_claims": False,
                    },
                }
                self._insert_evidence_retention_scheduler_run(
                    scheduler_run_id=scheduler_run_id,
                    schedule_id=schedule_id,
                    forced=force,
                    due=True,
                    status=status,
                    reason=reason,
                    retention_run_id=retention_run_id,
                    policy=policy,
                    dry_run=dry_run,
                    compact=compact,
                    vacuum=vacuum,
                    checked_at=checked_at,
                    next_run_after=computed_next,
                    result=result,
                )
                self.conn.commit()
                self.audit(
                    "evidence_retention.schedule.run",
                    outcome="ok",
                    details={
                        "scheduler_run_id": scheduler_run_id,
                        "schedule_id": schedule_id,
                        "status": status,
                        "retention_run_id": retention_run_id,
                        "forced": bool(force),
                        "dry_run": dry_run,
                    },
                )
                return result
            except Exception as exc:
                self.conn.execute(
                    "UPDATE evidence_retention_schedule SET last_checked_at=?, updated_at=? WHERE schedule_id=?",
                    (checked_at, checked_at, schedule_id),
                )
                result = {
                    "ok": False,
                    "schema": "monitorme.evidence_index_retention_scheduler_result.v0.1",
                    "scheduler_run_id": scheduler_run_id,
                    "schedule_id": schedule_id,
                    "status": "error",
                    "reason": reason,
                    "due": True,
                    "forced": bool(force),
                    "error": str(exc),
                    "privacy": {
                        "facts_only": True,
                        "external_upload": False,
                        "raw_frame_upload": False,
                        "media_decode": False,
                        "semantic_claims": False,
                    },
                }
                self._insert_evidence_retention_scheduler_run(
                    scheduler_run_id=scheduler_run_id,
                    schedule_id=schedule_id,
                    forced=force,
                    due=True,
                    status="error",
                    reason=reason,
                    retention_run_id=None,
                    policy=policy,
                    dry_run=dry_run,
                    compact=compact,
                    vacuum=vacuum,
                    checked_at=checked_at,
                    next_run_after=next_run_after,
                    result=result,
                    error=str(exc),
                )
                self.conn.commit()
                self.audit("evidence_retention.schedule.run", outcome="error", details=result)
                return result

    def list_evidence_retention_scheduler_runs(
        self,
        *,
        schedule_id: str | None = None,
        scheduler_run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            sql = "SELECT * FROM evidence_retention_scheduler_runs WHERE 1=1"
            args: list[Any] = []
            if schedule_id:
                sql += " AND schedule_id=?"; args.append(schedule_id)
            if scheduler_run_id:
                sql += " AND scheduler_run_id=?"; args.append(scheduler_run_id)
            if status:
                sql += " AND status=?"; args.append(status)
            sql += " ORDER BY checked_at DESC LIMIT ?"; args.append(max(1, min(int(limit), 1000)))
            return self._decode_rows(self.conn.execute(sql, args))

    def create_feedback(self, event_id: str, *, label: str, reason: str = "", operator: str = "operator") -> str:
        with self._lock:
            event = self.get_event(event_id)
            if not event:
                raise KeyError(f"event_id not found: {event_id}")
            feedback_id = new_id("fb")
            self.conn.execute(
                """
                INSERT INTO event_feedback(feedback_id, event_id, label, reason, operator, created_at)
                VALUES(?,?,?,?,?,?)
                """,
                (feedback_id, event_id, label, reason, operator, now_iso()),
            )
            self.conn.commit()
            self.audit("feedback.create", camera_id=event.get("camera_id"), event_id=event_id, details={"feedback_id": feedback_id, "label": label})
            return feedback_id

    def list_feedback(self, *, label: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            if label:
                rows = self.conn.execute("SELECT * FROM event_feedback WHERE label=? ORDER BY created_at DESC LIMIT ?", (label, limit))
            else:
                rows = self.conn.execute("SELECT * FROM event_feedback ORDER BY created_at DESC LIMIT ?", (limit,))
            return self._decode_rows(rows)

    def record_evidence_pack(self, *, pack_id: str, event_id: str | None, session_id: str | None, camera_id: str, pack_path: str, manifest_path: str, sha256: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO evidence_packs(pack_id, event_id, session_id, camera_id, pack_path, manifest_path, sha256, created_at)
                VALUES(?,?,?,?,?,?,?,?)
                """,
                (pack_id, event_id, session_id, camera_id, pack_path, manifest_path, sha256, now_iso()),
            )
            self.conn.commit()
            self.audit("evidence_pack.create", camera_id=camera_id, event_id=event_id, session_id=session_id, details={"pack_id": pack_id, "pack_path": pack_path})

    def create_incident_report(self, *, report_id: str, camera_id: str, title: str, severity: str, event_ids: list[str], session_ids: list[str], evidence_pack_ids: list[str], report_path: str, start_ts: str | None = None, end_ts: str | None = None) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO incident_reports(report_id, camera_id, start_ts, end_ts, title, severity,
                                             event_ids_json, session_ids_json, evidence_pack_ids_json,
                                             report_path, created_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (report_id, camera_id, start_ts, end_ts, title, severity, self._json(event_ids), self._json(session_ids), self._json(evidence_pack_ids), report_path, now_iso()),
            )
            self.conn.commit()
            self.audit("incident_report.create", camera_id=camera_id, report_id=report_id, details={"event_ids": event_ids, "report_path": report_path})
