from __future__ import annotations

import json
import sqlite3
import threading
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
