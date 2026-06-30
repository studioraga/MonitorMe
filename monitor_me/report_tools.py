from __future__ import annotations

from pathlib import Path
from typing import Any

from .db import MonitorMeDB
from .evidence_pack import EvidencePackBuilder
from .ids import new_id
from .time_utils import now_iso


class IncidentReportBuilder:
    def __init__(self, db: MonitorMeDB, reports_root: str | Path = "data/reports", evidence_root: str | Path = "data/evidence_packs"):
        self.db = db
        self.reports_root = Path(reports_root)
        self.reports_root.mkdir(parents=True, exist_ok=True)
        self.evidence_builder = EvidencePackBuilder(db, root=evidence_root)

    def build(self, *, event_ids: list[str], title: str | None = None, severity: str = "info") -> dict[str, Any]:
        if not event_ids:
            raise ValueError("event_ids must not be empty")
        events = [self.db.get_event(eid) for eid in event_ids]
        missing = [eid for eid, event in zip(event_ids, events) if not event]
        if missing:
            raise KeyError(f"missing event_ids: {missing}")
        valid_events = [e for e in events if e]
        camera_id = str(valid_events[0]["camera_id"])
        session_ids = sorted({str(e.get("session_id")) for e in valid_events if e.get("session_id")})
        pack_ids: list[str] = []
        for eid in event_ids:
            manifest = self.evidence_builder.build_for_event(eid)
            pack_ids.append(str(manifest["pack_id"]))
        report_id = new_id("ir")
        title = title or f"MonitorMe incident report for {camera_id}"
        path = self.reports_root / f"{report_id}.md"
        content = self._render(report_id, title, severity, valid_events, session_ids, pack_ids)
        path.write_text(content, encoding="utf-8")
        self.db.create_incident_report(
            report_id=report_id,
            camera_id=camera_id,
            title=title,
            severity=severity,
            event_ids=event_ids,
            session_ids=session_ids,
            evidence_pack_ids=pack_ids,
            report_path=str(path),
            start_ts=min(str(e.get("ts")) for e in valid_events if e.get("ts")),
            end_ts=max(str(e.get("ts")) for e in valid_events if e.get("ts")),
        )
        return {"report_id": report_id, "report_path": str(path), "evidence_pack_ids": pack_ids, "session_ids": session_ids}

    @staticmethod
    def _render(report_id: str, title: str, severity: str, events: list[dict[str, Any]], session_ids: list[str], pack_ids: list[str]) -> str:
        lines = [
            f"# {title}",
            "",
            f"- report_id: `{report_id}`",
            f"- severity: `{severity}`",
            f"- created_at: `{now_iso()}`",
            f"- session_ids: `{', '.join(session_ids)}`",
            f"- evidence_pack_ids: `{', '.join(pack_ids)}`",
            "",
            "## Events",
        ]
        for event in events:
            lines.append(
                f"- event_id=`{event.get('event_id')}` session_id=`{event.get('session_id')}` frame_id=`{event.get('frame_id')}` "
                f"type=`{event.get('event_type')}` label=`{event.get('label')}` confidence=`{event.get('confidence')}` model_id=`{event.get('model_id')}`"
            )
        lines.extend([
            "",
            "## Operator notes",
            "",
            "- Review each evidence pack before escalation.",
            "- MonitorMe v0.1 does not claim identity, face recognition, or intent.",
        ])
        return "\n".join(lines) + "\n"
