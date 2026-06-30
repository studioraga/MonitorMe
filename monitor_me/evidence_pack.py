from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .db import MonitorMeDB
from .hash_utils import sha256_file, sha256_text
from .ids import new_id
from .time_utils import now_iso


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


class EvidencePackBuilder:
    """Generate file-based evidence packs for incident review and audit."""

    def __init__(self, db: MonitorMeDB, root: str | Path = "data/evidence_packs"):
        self.db = db
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def build_for_event(self, event_id: str) -> dict[str, Any]:
        event = self.db.get_event(event_id)
        if not event:
            raise KeyError(f"event_id not found: {event_id}")
        related = self.db.related_events(event_id)
        session = self.db.get_session(event.get("session_id"))
        artifacts = self.db.list_artifacts(session_id=event.get("session_id")) if event.get("session_id") else []
        model = self.db.get_model(event.get("model_id")) if event.get("model_id") else None
        audit = self.db.recent_audit(event_id=event_id, session_id=event.get("session_id"), limit=100)
        pack_id = new_id("pack")
        pack_dir = self.root / pack_id
        pack_dir.mkdir(parents=True, exist_ok=False)

        policy_decision = (session or {}).get("policy_decision", {})
        assistant_summary = self._latest_summary(event_id)

        _write_json(pack_dir / "event.json", event)
        _write_json(pack_dir / "related_events.json", related)
        _write_json(pack_dir / "capture_session.json", session or {})
        _write_json(pack_dir / "artifacts.json", artifacts)
        _write_json(pack_dir / "model_metadata.json", model or {})
        _write_json(pack_dir / "policy_decision.json", policy_decision)
        _write_json(pack_dir / "audit.json", audit)
        _write_json(pack_dir / "assistant_summary.json", assistant_summary or {})

        report_md = self._render_event_report(event, related, session, artifacts, model, policy_decision, audit, assistant_summary)
        (pack_dir / "report.md").write_text(report_md, encoding="utf-8")

        manifest = {
            "pack_id": pack_id,
            "created_at": now_iso(),
            "event_id": event.get("event_id"),
            "session_id": event.get("session_id"),
            "camera_id": event.get("camera_id"),
            "files": {},
            "artifact_paths": [a.get("path") for a in artifacts if a.get("path")],
            "privacy": {
                "external_upload": False,
                "face_recognition": False,
                "notes": "Evidence pack references local files only. Raw frames are not uploaded by MonitorMe v0.1.",
            },
        }
        for file_path in sorted(pack_dir.iterdir()):
            if file_path.is_file() and file_path.name != "manifest.json":
                manifest["files"][file_path.name] = {
                    "size_bytes": file_path.stat().st_size,
                    "sha256": sha256_file(file_path),
                }
        _write_json(pack_dir / "manifest.json", manifest)
        manifest_sha = sha256_file(pack_dir / "manifest.json")
        self.db.record_evidence_pack(
            pack_id=pack_id,
            event_id=event.get("event_id"),
            session_id=event.get("session_id"),
            camera_id=event.get("camera_id"),
            pack_path=str(pack_dir),
            manifest_path=str(pack_dir / "manifest.json"),
            sha256=manifest_sha,
        )
        manifest["manifest_sha256"] = manifest_sha
        return manifest

    def _latest_summary(self, event_id: str) -> dict[str, Any] | None:
        with self.db._lock:  # Internal helper; still serialized.
            row = self.db.conn.execute(
                "SELECT * FROM assistant_summaries WHERE event_id=? ORDER BY created_at DESC LIMIT 1",
                (event_id,),
            ).fetchone()
            return MonitorMeDB._decode_row(row)

    @staticmethod
    def _render_event_report(
        event: dict[str, Any],
        related: list[dict[str, Any]],
        session: dict[str, Any] | None,
        artifacts: list[dict[str, Any]],
        model: dict[str, Any] | None,
        policy_decision: dict[str, Any],
        audit: list[dict[str, Any]],
        assistant_summary: dict[str, Any] | None,
    ) -> str:
        lines = [
            f"# MonitorMe Evidence Pack Report",
            "",
            f"- event_id: `{event.get('event_id')}`",
            f"- session_id: `{event.get('session_id')}`",
            f"- frame_id: `{event.get('frame_id')}`",
            f"- camera_id: `{event.get('camera_id')}`",
            f"- event_type: `{event.get('event_type')}`",
            f"- label: `{event.get('label')}`",
            f"- confidence: `{event.get('confidence')}`",
            f"- model_id: `{event.get('model_id')}`",
            "",
            "## Policy decision",
            "",
            "```json",
            json.dumps(policy_decision or {}, indent=2, sort_keys=True),
            "```",
            "",
            "## Local artifacts",
        ]
        if artifacts:
            for artifact in artifacts:
                lines.append(f"- `{artifact.get('artifact_type')}`: `{artifact.get('path')}` sha256=`{artifact.get('sha256')}`")
        else:
            lines.append("- No artifacts registered for this session.")
        lines.extend(["", "## Related events"])
        for row in related:
            lines.append(
                f"- event_id=`{row.get('event_id')}` type=`{row.get('event_type')}` label=`{row.get('label')}` frame_id=`{row.get('frame_id')}`"
            )
        lines.extend(["", "## Model metadata", "", "```json", json.dumps(model or {}, indent=2, sort_keys=True), "```"])
        lines.extend(["", "## Assistant summary"])
        lines.append((assistant_summary or {}).get("summary_text", "No assistant summary was recorded before this evidence pack was generated."))
        lines.extend(["", "## Audit records"])
        for item in audit[:25]:
            lines.append(f"- audit_id=`{item.get('audit_id')}` action=`{item.get('action')}` outcome=`{item.get('outcome')}`")
        lines.extend([
            "",
            "## Safety limits",
            "",
            "MonitorMe v0.1 does not perform face recognition, identity matching, or intent inference.",
        ])
        return "\n".join(lines) + "\n"
