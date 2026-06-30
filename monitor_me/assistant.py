from __future__ import annotations

from collections import Counter
from typing import Any

from .db import MonitorMeDB
from .event_tools import build_evidence_refs, extract_labels, is_correlation_question, query_events_for_question, sessions_with_all_labels
from .fact_guard import FactGuard
from .llm_client import LLMClient, NullLLMClient
from .schemas import AssistantAnswer


class MonitorMeAssistant:
    """DB-grounded assistant for CCTV evidence.

    v0.1 never treats the LLM as the database. It first queries normalized local
    SQLite evidence, then optionally uses an LLM to summarize those facts. A
    FactGuard validates the final text before it is saved or returned.
    """

    def __init__(self, db: MonitorMeDB, *, llm: LLMClient | None = None, fact_guard: FactGuard | None = None):
        self.db = db
        self.llm = llm or NullLLMClient()
        self.fact_guard = fact_guard or FactGuard()

    def ask(self, question: str, *, camera_id: str | None = None, limit: int = 100, use_llm: bool = False) -> AssistantAnswer:
        llm_model_id = getattr(self.llm, "model_id", None)
        if llm_model_id and not self.db.get_model(llm_model_id):
            llm_model_id = None
        run_id = self.db.create_assistant_run(question, status="running", model_id=llm_model_id)
        q_lower = question.lower()
        unsupported_terms = ("weapon", "gun", "knife", "face", "recognized", "identity", "intent", "suspicious")
        if any(term in q_lower for term in unsupported_terms):
            answer = "I do not have local evidence for that request. MonitorMe v0.1 does not support that claim without normalized local evidence."
            evidence: list[dict[str, Any]] = []
            self.db.complete_assistant_run(run_id, status="completed", answer=answer, evidence=evidence)
            return AssistantAnswer(answer=answer, evidence=evidence, limits=self._limits(), run_id=run_id)

        events = query_events_for_question(self.db, question, camera_id=camera_id, limit=limit)
        labels = extract_labels(question)

        # Support explicit co-occurrence questions such as "person + vehicle" or
        # "person and vehicle in the same clip". Plain wording like
        # "person and vehicle events" uses union semantics in event_tools.
        if len(labels) >= 2 and is_correlation_question(question):
            matched = sessions_with_all_labels(events, labels)
            events = [row for rows in matched.values() for row in rows]

        evidence = build_evidence_refs(self.db, events)
        answer = self._compose_answer(question, evidence, labels)

        if use_llm and evidence:
            proposed = self.llm.summarize(question, evidence)
            validation = self.fact_guard.validate(proposed, evidence)
            if validation.ok:
                answer = proposed

        validation = self.fact_guard.validate(answer, evidence)
        if not validation.ok:
            safe_answer = (
                "I cannot safely answer that from local evidence. "
                f"Fact guard violations: {'; '.join(validation.violations)}"
            )
            self.db.complete_assistant_run(run_id, status="rejected", answer=safe_answer, evidence=evidence, error="; ".join(validation.violations))
            return AssistantAnswer(answer=safe_answer, evidence=evidence, limits=self._limits(), run_id=run_id)

        self.db.complete_assistant_run(run_id, status="completed", answer=answer, evidence=evidence)
        if evidence:
            first = evidence[0]
            self.db.create_summary(
                run_id=run_id,
                camera_id=str(first.get("camera_id")),
                event_id=first.get("event_id"),
                session_id=first.get("session_id"),
                summary_text=answer,
                facts={"question": question, "evidence_count": len(evidence)},
                source_refs=evidence,
                model_id=llm_model_id,
            )
        return AssistantAnswer(answer=answer, evidence=evidence, limits=self._limits(), run_id=run_id)

    def _compose_answer(self, question: str, evidence: list[dict[str, Any]], labels: list[str]) -> str:
        q = question.lower()
        if not evidence:
            return (
                "I do not have local evidence for that request. "
                "No matching event_id/session_id/frame_id rows were found in the MonitorMe evidence DB."
            )

        if "weapon" in q or "face" in q or "recognized" in q or "identity" in q:
            return "I do not have local evidence for that request. MonitorMe v0.1 does not support that claim without normalized local evidence."

        label_counts = Counter(str(item.get("label") or "unknown") for item in evidence)
        event_type_counts = Counter(str(item.get("event_type") or "unknown") for item in evidence)
        missing_labels = [label for label in labels if label not in label_counts]
        session_ids = sorted({str(item.get("session_id")) for item in evidence if item.get("session_id")})
        camera_ids = sorted({str(item.get("camera_id")) for item in evidence if item.get("camera_id")})
        first_refs = evidence[:5]
        refs = "; ".join(
            f"event_id={item.get('event_id')} session_id={item.get('session_id')} frame_id={item.get('frame_id')} label={item.get('label')}"
            for item in first_refs
        )
        label_text = ", ".join(f"{label}={count}" for label, count in sorted(label_counts.items()))
        event_type_text = ", ".join(f"{kind}={count}" for kind, count in sorted(event_type_counts.items()))

        if len(labels) >= 2:
            missing_text = f" No local evidence found for requested label(s): {', '.join(missing_labels)}." if missing_labels else ""
            mode_text = "co-occurring " if is_correlation_question(question) else ""
            return (
                f"I found {mode_text}local evidence for requested label(s) {', '.join(labels)} across {len(session_ids)} session(s). "
                f"Event types: {event_type_text}. Labels: {label_text}. Evidence: {refs}." + missing_text
            )
        return (
            f"I found {len(evidence)} local evidence item(s) for camera(s) {', '.join(camera_ids)}. "
            f"Event types: {event_type_text}. Labels: {label_text}. Evidence: {refs}."
        )

    @staticmethod
    def _limits() -> list[str]:
        return [
            "Answer is based only on local MonitorMe SQLite evidence and artifact metadata.",
            "No private CCTV frames are uploaded to external services.",
            "No face recognition, identity claim, or intent claim is made in v0.1.",
        ]
