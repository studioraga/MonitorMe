from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float | None = None
    bbox: list[float] | None = None
    frame_id: int | None = None
    track_id: str | None = None
    zone_id: str | None = None
    model_id: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceRef:
    event_id: str | None = None
    session_id: str | None = None
    frame_id: int | None = None
    camera_id: str | None = None
    artifact_paths: list[str] = field(default_factory=list)
    model_id: str | None = None
    policy_decision: dict[str, Any] = field(default_factory=dict)
    audit_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class AssistantAnswer:
    answer: str
    evidence: list[dict[str, Any]]
    limits: list[str]
    run_id: str | None = None
