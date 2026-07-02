from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class FactGuardResult:
    ok: bool
    violations: list[str]


class FactGuard:
    """Reject assistant answers that assert facts absent from local evidence.

    v0.1 is intentionally conservative. It supports event/detection facts such as
    label, confidence, frame_id, session_id, model_id, artifact path, and policy
    decision. It blocks identity, face-recognition, intent, and weapon claims
    unless those labels are explicitly present as normalized evidence rows.
    """

    SENSITIVE_OR_UNSUPPORTED_TERMS = {
        "face": {"requires_label": "face"},
        "recognized": {"requires_label": "face"},
        "identity": {"requires_label": "identity"},
        "name": {"requires_label": "identity"},
        "weapon": {"requires_label": "weapon"},
        "gun": {"requires_label": "weapon"},
        "knife": {"requires_label": "weapon"},
        "suspicious": {"requires_label": "suspicious_behavior"},
        "intent": {"requires_label": "intent"},
    }

    def validate(self, answer: str, evidence: list[dict[str, Any]]) -> FactGuardResult:
        violations: list[str] = []
        answer_l = answer.lower()
        labels = {str(item.get("label", "")).lower() for item in evidence if item.get("label")}
        event_ids = {item.get("event_id") for item in evidence if item.get("event_id")}

        if not evidence and not any(phrase in answer_l for phrase in ("do not have local evidence", "no matching local evidence", "no local evidence")):
            violations.append("answer has no evidence but does not clearly state the evidence limit")

        for term, rule in self.SENSITIVE_OR_UNSUPPORTED_TERMS.items():
            # Use token-aware matching so random evidence IDs such as
            # "evt_face..." do not trigger a face-recognition violation.
            if re.search(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", answer_l):
                required = rule["requires_label"]
                if required not in labels:
                    violations.append(f"unsupported claim term '{term}' requires normalized label '{required}'")

        for item in evidence:
            if not item.get("event_id") and not item.get("session_id"):
                violations.append("evidence item missing event_id/session_id reference")
            if item.get("event_id") and not item.get("camera_id"):
                violations.append(f"evidence event {item.get('event_id')} missing camera_id")

        # If events are cited in evidence, the answer should expose at least one
        # event/session reference or explicitly state that details follow below.
        if event_ids and not any(str(eid) in answer for eid in event_ids):
            if "evidence" not in answer_l and "event_id" not in answer_l:
                violations.append("answer does not surface event_id references")

        return FactGuardResult(ok=not violations, violations=violations)
