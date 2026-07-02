from __future__ import annotations

from pathlib import Path
from typing import Any

from .db import MonitorMeDB
from .vlm_client import KeyframeVLMClient, build_default_keyframe_vlm, validate_qwen_keyframe_json


class KeyframeVLMAnalysisService:
    """Run optional Qwen VLM analysis on stored trigger keyframes.

    v0.3 intentionally runs after the trigger and after a keyframe artifact has
    been stored. The VLM produces companion visual-facts JSON only. It cannot
    create YOLO object rows or override deterministic policy decisions.
    """

    def __init__(self, db: MonitorMeDB, *, vlm: KeyframeVLMClient | None = None):
        self.db = db
        self.vlm = vlm if vlm is not None else build_default_keyframe_vlm()

    def analyze_event(self, event_id: str) -> dict[str, Any]:
        event = self.db.get_event(event_id)
        if not event:
            raise KeyError(f"event_id not found: {event_id}")
        artifact = self._keyframe_artifact(event)
        if not artifact:
            raise ValueError(f"event_id has no keyframe artifact: {event_id}")
        artifact_path = str(artifact.get("path") or "")
        source_refs = self._source_refs(event, artifact)
        if self.vlm is None:
            analysis = {
                "schema_version": "monitorme.qwen_vlm_keyframe.v0.3",
                "scene_summary": "Qwen VLM is not configured; no keyframe visual analysis was run.",
                "visible_entities": [],
                "text_visible": "unknown",
                "image_quality": "unknown",
                "safety_notes": "No VLM call was made.",
                "cited_event_ids": [str(event_id)],
                "cited_artifact_ids": [str(artifact.get("artifact_id"))] if artifact.get("artifact_id") else [],
                "limitations": ["Qwen VLM disabled or not configured."],
                "_validated": True,
            }
            analysis_id = self.db.create_vlm_analysis(
                event_id=event_id,
                parent_event_id=event.get("parent_event_id"),
                session_id=event.get("session_id"),
                camera_id=event["camera_id"],
                frame_id=event.get("frame_id"),
                artifact_id=artifact.get("artifact_id"),
                artifact_path=artifact_path,
                model_id=None,
                status="skipped",
                analysis=analysis,
                source_refs=source_refs,
                error="not_configured",
            )
            return {"status": "skipped", "analysis_id": analysis_id, "analysis": analysis, "source_refs": source_refs, "model_id": "qwen-vlm-disabled"}
        try:
            related = self.db.related_events(event_id)
            raw_analysis = self.vlm.analyze_keyframe(image_path=artifact_path, event=event, related_events=related, artifact=artifact)
            analysis = validate_qwen_keyframe_json(raw_analysis, event=event, related_events=related, artifact=artifact)
            analysis_id = self.db.create_vlm_analysis(
                event_id=event_id,
                parent_event_id=event.get("parent_event_id"),
                session_id=event.get("session_id"),
                camera_id=event["camera_id"],
                frame_id=event.get("frame_id"),
                artifact_id=artifact.get("artifact_id"),
                artifact_path=artifact_path,
                model_id=getattr(self.vlm, "model_id", "qwen-vlm"),
                status="completed",
                analysis=analysis,
                source_refs=source_refs,
                error=None,
            )
            return {"status": "completed", "analysis_id": analysis_id, "analysis": analysis, "source_refs": source_refs, "model_id": getattr(self.vlm, "model_id", "qwen-vlm")}
        except Exception as exc:
            analysis_id = self.db.create_vlm_analysis(
                event_id=event_id,
                parent_event_id=event.get("parent_event_id"),
                session_id=event.get("session_id"),
                camera_id=event["camera_id"],
                frame_id=event.get("frame_id"),
                artifact_id=artifact.get("artifact_id"),
                artifact_path=artifact_path,
                model_id=getattr(self.vlm, "model_id", "qwen-vlm"),
                status="failed",
                analysis={},
                source_refs=source_refs,
                error=str(exc),
            )
            return {"status": "failed", "analysis_id": analysis_id, "analysis": {}, "source_refs": source_refs, "model_id": getattr(self.vlm, "model_id", "qwen-vlm"), "error": str(exc)}

    def _keyframe_artifact(self, event: dict[str, Any]) -> dict[str, Any] | None:
        artifact_id = event.get("artifact_id")
        artifacts = self.db.list_artifacts(event_id=str(event.get("event_id")), artifact_type="keyframe")
        if artifact_id:
            artifacts.extend([row for row in self.db.list_artifacts(session_id=event.get("session_id"), artifact_type="keyframe") if row.get("artifact_id") == artifact_id])
        for artifact in artifacts:
            path = Path(str(artifact.get("path") or ""))
            if artifact.get("artifact_type") == "keyframe" and artifact.get("path") and path.exists():
                return artifact
        return artifacts[0] if artifacts else None

    @staticmethod
    def _source_refs(event: dict[str, Any], artifact: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "kind": "event",
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "camera_id": event.get("camera_id"),
                "session_id": event.get("session_id"),
                "frame_id": event.get("frame_id"),
            },
            {
                "kind": "artifact",
                "artifact_id": artifact.get("artifact_id"),
                "artifact_type": artifact.get("artifact_type"),
                "path": artifact.get("path"),
                "sha256": artifact.get("sha256"),
            },
        ]
