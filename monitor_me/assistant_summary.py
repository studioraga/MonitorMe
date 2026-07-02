from __future__ import annotations

from collections import Counter
from typing import Any

from .capture_policy import evaluate_node1_event_policy
from .db import MonitorMeDB
from .event_contract import build_event_contract
from .event_tools import build_evidence_refs
from .fact_guard import FactGuard
from .llm_client import EventSummaryLLMClient, build_default_event_summary_llm


class AssistantSummaryService:
    """Create operator summaries after Node1 evidence events.

    v0.2 can use a configured local Gemma/MAX client, but the deterministic
    summary remains the safety fallback whenever MAX is unavailable, returns
    malformed JSON, or fails validation.
    """

    deterministic_model_id = "deterministic-null-llm"

    def __init__(
        self,
        db: MonitorMeDB,
        *,
        fact_guard: FactGuard | None = None,
        llm: EventSummaryLLMClient | None = None,
    ):
        self.db = db
        self.fact_guard = fact_guard or FactGuard()
        self.llm = llm if llm is not None else build_default_event_summary_llm()

    def summarize_event(self, event_id: str) -> dict[str, Any]:
        event = self.db.get_event(event_id)
        if not event:
            raise KeyError(f"event_id not found: {event_id}")
        contract = build_event_contract(self.db, event_id)
        policy = evaluate_node1_event_policy(contract).as_dict()
        contract_id = self.db.record_event_contract(
            event_id=event_id,
            parent_event_id=contract.get("motion_event_id") if contract.get("motion_event_id") != event_id else None,
            session_id=event.get("session_id"),
            camera_id=str(event.get("camera_id")),
            contract=contract,
            policy_decision=policy,
        )
        related = self.db.related_events(event_id)
        evidence = build_evidence_refs(self.db, related if related else [event])
        deterministic_text = self._compose_summary(contract, policy)
        summary_text = deterministic_text
        summary_payload: dict[str, Any] | None = None
        fallback_reason = "not_configured"
        selected_model_id = self.deterministic_model_id
        summary_source = "deterministic"

        if self.llm is not None:
            try:
                candidate = self.llm.summarize_event_contract(
                    event_contract=contract,
                    policy_decision=policy,
                    evidence=evidence,
                )
                candidate_text = self._render_llm_summary(candidate)
                validation = self.fact_guard.validate(candidate_text, evidence)
                if not validation.ok:
                    raise ValueError("; ".join(validation.violations))
                summary_payload = candidate
                summary_text = candidate_text
                selected_model_id = getattr(self.llm, "model_id", "local-gemma-max")
                summary_source = "gemma_max"
                fallback_reason = ""
            except Exception as exc:  # keep capture path robust
                fallback_reason = str(exc)
                self.db.audit(
                    "assistant.gemma_summary.fallback",
                    outcome="warning",
                    camera_id=str(event.get("camera_id")),
                    event_id=event_id,
                    session_id=event.get("session_id"),
                    details={"reason": fallback_reason, "fallback_model_id": self.deterministic_model_id},
                )

        validation = self.fact_guard.validate(summary_text, evidence)
        status = "completed" if validation.ok else "failed"
        if not validation.ok:
            summary_text = "Assistant summary rejected by fact guard: " + "; ".join(validation.violations)
            selected_model_id = self.deterministic_model_id
            summary_source = "rejected"

        model_id_for_db = selected_model_id if self.db.get_model(selected_model_id) else None
        summary_id = self.db.create_summary(
            run_id=None,
            event_id=event_id,
            session_id=event.get("session_id"),
            camera_id=str(event.get("camera_id")),
            summary_text=summary_text,
            facts={
                "event_contract": contract,
                "policy_decision": policy,
                "summary_source": summary_source,
                "llm_model_id": selected_model_id if summary_source == "gemma_max" else None,
                "gemma_summary_json": summary_payload,
                "fallback_reason": fallback_reason,
                "deterministic_fallback_summary": deterministic_text,
                "contract_id": contract_id,
            },
            source_refs=evidence,
            model_id=model_id_for_db,
            status=status,
        )
        self.db.audit(
            "assistant.summary.auto_create",
            outcome=status,
            camera_id=str(event.get("camera_id")),
            event_id=event_id,
            session_id=event.get("session_id"),
            details={"summary_id": summary_id, "policy_action": policy.get("action"), "summary_source": summary_source},
        )
        return {
            "summary_id": summary_id,
            "status": status,
            "summary_text": summary_text,
            "event_contract": contract,
            "policy_decision": policy,
            "contract_id": contract_id,
            "summary_source": summary_source,
            "llm_model_id": selected_model_id if summary_source == "gemma_max" else None,
            "fallback_reason": fallback_reason,
            "gemma_summary_json": summary_payload,
        }

    def summarize_event_group(self, parent_event_id: str, child_event_ids: list[str] | None = None) -> list[dict[str, Any]]:
        results = [self.summarize_event(parent_event_id)]
        for event_id in child_event_ids or []:
            results.append(self.summarize_event(event_id))
        return results

    @staticmethod
    def _render_llm_summary(summary_json: dict[str, Any]) -> str:
        return (
            f"{summary_json['operator_summary']} "
            f"Reason: {summary_json['event_reason']} "
            f"Next step: {summary_json['recommended_next_step']} "
            f"Dashboard tag: {summary_json['dashboard_tag']}. "
            f"Severity: {summary_json['severity_label']}. "
            f"Cited events: {', '.join(summary_json['cited_event_ids'])}."
        )

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
