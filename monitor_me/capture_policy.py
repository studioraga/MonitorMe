from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

ALLOWED_SEVERITIES = {"info", "review", "urgent"}


@dataclass(frozen=True)
class Node1PolicyDecision:
    decision: str
    action: str
    severity_label: str
    duration_sec: int
    reason: str
    matched_labels: list[str]
    policy_version: str = "node1-ai-camera-assistant-v0.1"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["severity_label"] not in ALLOWED_SEVERITIES:
            data["severity_label"] = "review"
        return data


def _detections(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = contract.get("detections")
    return rows if isinstance(rows, list) else []


def evaluate_node1_event_policy(
    contract: dict[str, Any],
    *,
    person_conf_threshold: float = 0.60,
    capture_duration_sec: int = 90,
) -> Node1PolicyDecision:
    """Deterministic Node1 decision layer.

    This policy intentionally stays simple for v0.1. It never asks the LLM to
    decide whether to act. It only uses normalized local facts in the event
    contract.
    """

    detections = _detections(contract)
    labels = [str(det.get("class_name") or det.get("label") or "").lower() for det in detections]
    person_hits = [
        det
        for det in detections
        if str(det.get("class_name") or det.get("label") or "").lower() == "person"
        and float(det.get("confidence") or 0.0) >= person_conf_threshold
    ]
    if person_hits:
        best = max(float(det.get("confidence") or 0.0) for det in person_hits)
        return Node1PolicyDecision(
            decision="allow",
            action="request_capture_review",
            severity_label="review",
            duration_sec=int(capture_duration_sec),
            reason=f"person confidence {best:.2f} >= {person_conf_threshold:.2f}; review local capture evidence",
            matched_labels=sorted(set(labels)),
        )

    if detections:
        return Node1PolicyDecision(
            decision="allow",
            action="record_evidence_only",
            severity_label="info",
            duration_sec=0,
            reason="object evidence exists but no configured action trigger matched",
            matched_labels=sorted(set(labels)),
        )

    if str(contract.get("event_type")) == "motion_detected":
        return Node1PolicyDecision(
            decision="allow",
            action="record_motion_only",
            severity_label="info",
            duration_sec=0,
            reason="motion evidence exists but no object evidence was produced",
            matched_labels=["motion"],
        )

    return Node1PolicyDecision(
        decision="allow",
        action="record_evidence_only",
        severity_label="info",
        duration_sec=0,
        reason="no configured action trigger matched",
        matched_labels=sorted(set(labels)),
    )
