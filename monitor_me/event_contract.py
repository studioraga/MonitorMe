from __future__ import annotations

from typing import Any

from .db import MonitorMeDB


def _as_detection(row: dict[str, Any]) -> dict[str, Any]:
    attrs = row.get("attrs") or {}
    raw_label = attrs.get("raw_label") or row.get("label")
    return {
        "event_id": row.get("event_id"),
        "class_name": row.get("label"),
        "raw_label": raw_label,
        "confidence": row.get("confidence"),
        "bbox_xyxy_norm": row.get("bbox"),
        "track_id": row.get("track_id"),
        "duration_sec": attrs.get("duration_sec"),
        "model_id": row.get("model_id"),
        "artifact_id": row.get("artifact_id"),
        "frame_id": row.get("frame_id"),
    }


def build_event_contract(db: MonitorMeDB, event_id: str) -> dict[str, Any]:
    """Build the strict JSON fact contract consumed by policy/Gemma layers."""

    event = db.get_event(event_id)
    if not event:
        raise KeyError(f"event_id not found: {event_id}")

    parent_id = event.get("parent_event_id") or event.get("event_id")
    related = db.related_events(event_id)
    children = [row for row in related if row.get("parent_event_id") == parent_id and row.get("event_type") == "object_detected"]
    if event.get("event_type") == "object_detected" and not children:
        children = [event]
    motion = next((row for row in related if row.get("event_id") == parent_id), event)
    session = db.get_session(event.get("session_id")) if event.get("session_id") else None
    artifacts = db.list_artifacts(session_id=event.get("session_id")) if event.get("session_id") else []
    detector_model = next((row.get("model_id") for row in children if row.get("model_id")), event.get("model_id"))

    contract = {
        "schema_version": "1.0",
        "contract_type": "monitorme.node1_ai_camera_event",
        "event_id": event.get("event_id"),
        "parent_event_id": parent_id if parent_id != event.get("event_id") else None,
        "motion_event_id": parent_id,
        "source_node": event.get("source_node") or "node1",
        "source_kind": event.get("source_kind") or "local_v4l2",
        "camera_id": event.get("camera_id"),
        "timestamp": event.get("ts"),
        "event_type": event.get("event_type"),
        "label": event.get("label"),
        "confidence": event.get("confidence"),
        "frame_id": event.get("frame_id"),
        "bbox_xyxy_norm": event.get("bbox"),
        "session_id": event.get("session_id"),
        "detector": {
            "model_id": detector_model,
            "runtime": "onnxruntime" if detector_model else None,
            "input_size": [640, 640] if detector_model else None,
            "role": "fast_visual_facts_only",
        },
        "motion": {
            "event_id": motion.get("event_id"),
            "motion_score": motion.get("confidence"),
            "bbox_xyxy_norm": motion.get("bbox"),
            "frame_id": motion.get("frame_id"),
        },
        "detections": [_as_detection(row) for row in children],
        "artifacts": [
            {
                "artifact_id": item.get("artifact_id"),
                "artifact_type": item.get("artifact_type"),
                "path": item.get("path"),
                "sha256": item.get("sha256"),
                "media_type": item.get("media_type"),
            }
            for item in artifacts
        ],
        "session": {
            "session_id": (session or {}).get("session_id"),
            "status": (session or {}).get("status"),
            "manifest_path": (session or {}).get("manifest_path"),
            "dataset_path": (session or {}).get("dataset_path"),
            "policy_decision": (session or {}).get("policy_decision", {}),
        },
        "privacy": {
            "external_upload": False,
            "face_recognition": False,
            "raw_frame_upload": False,
            "identity_claim": False,
        },
    }
    return contract
