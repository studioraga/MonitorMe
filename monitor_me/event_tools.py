from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from .db import MonitorMeDB

LABEL_ALIASES = {
    "person": {"person", "people", "human"},
    "vehicle": {"vehicle", "car", "truck", "bike", "bicycle", "motorbike", "motorcycle", "bus", "auto", "train", "boat"},
    "motion": {"motion", "movement"},
    "chair": {"chair", "chairs"},
    "bed": {"bed", "beds"},
}

OBJECT_WORDS = {"object", "objects", "detected", "detection", "detections", "yolo"}
CORRELATION_WORDS = (
    "+",
    "same clip",
    "same session",
    "same frame",
    "with both",
    "both person and vehicle",
    "had person + vehicle",
    "had both",
)


def _iso_minutes_ago(minutes: int) -> str:
    return (datetime.now(timezone.utc).astimezone() - timedelta(minutes=minutes)).isoformat(timespec="seconds")


def _today_start_iso() -> str:
    now = datetime.now(timezone.utc).astimezone()
    return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat(timespec="seconds")


def extract_labels(question: str) -> list[str]:
    q = question.lower()
    labels: list[str] = []
    for canonical, words in LABEL_ALIASES.items():
        if any(re.search(rf"\b{re.escape(word)}\b", q) for word in words):
            labels.append(canonical)
    return labels


def is_correlation_question(question: str) -> bool:
    """Return True only when the user asks for co-occurrence.

    Plain wording such as "person and vehicle events" means a union of those
    event labels. Correlation/co-occurrence requires explicit hints such as
    "+", "same clip", "same session", or "both".
    """

    q = question.lower()
    return any(token in q for token in CORRELATION_WORDS)


def wants_object_events(question: str) -> bool:
    q_words = set(re.findall(r"\b[a-z0-9_]+\b", question.lower()))
    return bool(q_words & OBJECT_WORDS)


def extract_time_filter(question: str) -> tuple[str | None, str | None]:
    q = question.lower()
    if "today" in q:
        return _today_start_iso(), None
    m = re.search(r"last\s+(\d+)\s*(minute|minutes|min|hour|hours|hr|hrs)", q)
    if m:
        value = int(m.group(1))
        unit = m.group(2)
        minutes = value * 60 if unit.startswith(("hour", "hr")) else value
        return _iso_minutes_ago(minutes), None
    return None, None


def _label_matches(row: dict[str, Any], labels: set[str]) -> bool:
    label = str(row.get("label") or "").lower()
    if label in labels:
        return True
    if "vehicle" in labels and label in LABEL_ALIASES["vehicle"]:
        return True
    return False


def query_events_for_question(db: MonitorMeDB, question: str, *, camera_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    labels = extract_labels(question)
    start_ts, end_ts = extract_time_filter(question)

    if is_correlation_question(question) and len(labels) >= 2:
        return db.list_events(camera_id=camera_id, event_type="object_detected", start_ts=start_ts, end_ts=end_ts, limit=limit)

    if len(labels) >= 2:
        # Union semantics for questions like: "What person and vehicle events happened today?"
        rows = db.list_events(camera_id=camera_id, event_type="object_detected", start_ts=start_ts, end_ts=end_ts, limit=limit)
        wanted = set(labels)
        return [row for row in rows if _label_matches(row, wanted)]

    if labels:
        label = labels[0]
        event_type = "motion_detected" if label == "motion" else "object_detected"
        return db.list_events(camera_id=camera_id, event_type=event_type, label=label, start_ts=start_ts, end_ts=end_ts, limit=limit)

    if wants_object_events(question):
        return db.list_events(camera_id=camera_id, event_type="object_detected", start_ts=start_ts, end_ts=end_ts, limit=limit)

    return db.list_events(camera_id=camera_id, start_ts=start_ts, end_ts=end_ts, limit=limit)


def sessions_with_all_labels(events: list[dict[str, Any]], labels: list[str]) -> dict[str, list[dict[str, Any]]]:
    wanted = set(labels)
    by_session: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        sid = event.get("session_id")
        if not sid:
            continue
        by_session.setdefault(str(sid), []).append(event)
    matched: dict[str, list[dict[str, Any]]] = {}
    for sid, rows in by_session.items():
        labels_in_session = {str(row.get("label")) for row in rows if row.get("label")}
        # vehicle is a canonical bucket: direct vehicle rows or common vehicle labels count.
        if "vehicle" in wanted:
            labels_in_session |= {"vehicle" for row in rows if str(row.get("label", "")).lower() in LABEL_ALIASES["vehicle"]}
        if wanted.issubset(labels_in_session):
            matched[sid] = rows
    return matched


def build_evidence_refs(db: MonitorMeDB, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for event in events:
        session = db.get_session(event.get("session_id")) if event.get("session_id") else None
        artifacts = db.list_artifacts(session_id=event.get("session_id")) if event.get("session_id") else []
        audits = db.recent_audit(event_id=event.get("event_id"), session_id=event.get("session_id"), limit=10)
        evidence.append(
            {
                "event_id": event.get("event_id"),
                "parent_event_id": event.get("parent_event_id"),
                "session_id": event.get("session_id"),
                "frame_id": event.get("frame_id"),
                "camera_id": event.get("camera_id"),
                "event_type": event.get("event_type"),
                "label": event.get("label"),
                "confidence": event.get("confidence"),
                "bbox": event.get("bbox"),
                "model_id": event.get("model_id"),
                "artifact_paths": [a.get("path") for a in artifacts if a.get("path")],
                "policy_decision": (session or {}).get("policy_decision", {}),
                "audit_ids": [a.get("audit_id") for a in audits if a.get("audit_id")],
                "ts": event.get("ts"),
            }
        )
    return evidence
