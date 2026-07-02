from __future__ import annotations

from collections import Counter
from typing import Any

from .capture_policy import evaluate_node1_event_policy
from .db import MonitorMeDB
from .event_contract import build_event_contract
from .event_tools import build_evidence_refs
from .fact_guard import FactGuard


class AssistantSummaryService:
    """Create deterministic operator summaries after Node1 evidence events."""

    model_id = "deterministic-null-llm"

    def __init__(self, db: MonitorMeDB, *, fact_guard: FactGuard | None = None):
        self.db = db
        self.fact_guard = fact_guard or FactGuard()

    def summarize_event(self, event_id: str) -> dict[str, Any]:
        event = self.db.get_event(event_id)
        if not event:
            raise KeyError(f"event_id not found: {event_id}")
        contract = build_event_contract(self.db, event_id)
        policy = evaluate_node1_event_policy(contract).as_dict()
        self.db.record_event_contract(
            event_id=event_id,
            parent_event_id=contract.get("motion_event_id") if contract.get("motion_event_id") != event_id else None,
            session_id=event.get("session_id"),
            camera_id=str(event.get("camera_id")),
            contract=contract,
            policy_decision=policy,
        )
        related = self.db.related_events(event_id)
        # Prefer the event itself plus its event group for validation/source refs.
        evidence = build_evidence_refs(self.db, related if related else [event])
        summary_text = self._compose_summary(contract, policy)
        validation = self.fact_guard.validate(summary_text, evidence)
        status = "completed" if validation.ok else "failed"
        if not validation.ok:
            summary_text = "Assistant summary rejected by fact guard: " + "; ".join(validation.violations)
        summary_id = self.db.create_summary(
            run_id=None,
            event_id=event_id,
            session_id=event.get("session_id"),
            camera_id=str(event.get("camera_id")),
            summary_text=summary_text,
            facts={"event_contract": contract, "policy_decision": policy},
            source_refs=evidence,
            model_id=self.model_id if self.db.get_model(self.model_id) else None,
            status=status,
        )
        self.db.audit(
            "assistant.summary.auto_create",
            outcome=status,
            camera_id=str(event.get("camera_id")),
            event_id=event_id,
            session_id=event.get("session_id"),
            details={"summary_id": summary_id, "policy_action": policy.get("action")},
        )
        return {"summary_id": summary_id, "status": status, "summary_text": summary_text, "event_contract": contract, "policy_decision": policy}

    def summarize_event_group(self, parent_event_id: str, child_event_ids: list[str] | None = None) -> list[dict[str, Any]]:
        results = [self.summarize_event(parent_event_id)]
        for event_id in child_event_ids or []:
            results.append(self.summarize_event(event_id))
        return results

    @staticmethod
    def _compose_summary(contract: dict[str, Any], policy: dict[str, Any]) -> str:
        camera_id = contract.get("camera_id")
        event_id = contract.get("event_id")
        session_id = contract.get("session_id")
        frame_id = contract.get("frame_id")
        event_type = contract.get("event_type")
        detections = contract.get("detections") or []
        if detections:
            counts = Counter(str(det.get("class_name") or "unknown") for det in detections)
            labels = ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))
            best = max(float(det.get("confidence") or 0.0) for det in detections)
            return (
                f"Node1 camera {camera_id} recorded {event_type} evidence at frame_id={frame_id} "
                f"with labels {labels}; highest confidence={best:.2f}. "
                f"event_id={event_id} session_id={session_id}. Deterministic policy action={policy.get('action')} "
                f"because {policy.get('reason')}."
            )
        label = contract.get("label") or "motion"
        conf = contract.get("confidence")
        conf_text = f" confidence={float(conf):.2f}" if conf is not None else ""
        return (
            f"Node1 camera {camera_id} recorded {event_type} evidence label={label}{conf_text} at frame_id={frame_id}. "
            f"event_id={event_id} session_id={session_id}. Deterministic policy action={policy.get('action')} "
            f"because {policy.get('reason')}."
        )
