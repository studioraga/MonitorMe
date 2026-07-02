from __future__ import annotations

from pathlib import Path
from typing import Any

from .db import MonitorMeDB
from .smolvlm2_client import SMOLVLM2_SCHEMA_VERSION, ShortClipVLMClient, build_default_smolvlm2_client, validate_smolvlm2_short_clip_json


class ShortClipVLMExperimentService:
    """Run optional SmolVLM2 experiments on locally stored short clip bundles.

    v0.4 is deliberately experimental. It runs after a motion/YOLO trigger and
    after MonitorMe has written a short clip manifest artifact. Results are
    stored as companion temporal observations only; they cannot create YOLO rows,
    override policy, or make identity/intent/threat claims.
    """

    def __init__(self, db: MonitorMeDB, *, vlm: ShortClipVLMClient | None = None):
        self.db = db
        self.vlm = vlm if vlm is not None else build_default_smolvlm2_client()

    def analyze_event(self, event_id: str) -> dict[str, Any]:
        event = self.db.get_event(event_id)
        if not event:
            raise KeyError(f"event_id not found: {event_id}")
        clip_artifact = self._short_clip_artifact(event)
        if not clip_artifact:
            raise ValueError(f"event_id has no short clip manifest artifact: {event_id}")
        clip_path = str(clip_artifact.get("path") or "")
        frame_artifacts = self._frame_artifacts_from_manifest(clip_path, event)
        source_refs = self._source_refs(event, clip_artifact, frame_artifacts)
        if self.vlm is None:
            experiment = {
                "schema_version": SMOLVLM2_SCHEMA_VERSION,
                "event_id": str(event_id),
                "artifact_id": str(clip_artifact.get("artifact_id")),
                "visible_scene": "unclear",
                "person_like_presence": "unclear",
                "vehicle_like_presence": "unclear",
                "motion_claim": "single_frame_only_no_motion_claim",
                "safe_observation": "visible content unclear",
                "unsupported_claims": [],
                "_validated": True,
            }
            experiment_id = self.db.create_smolvlm2_clip_experiment(
                event_id=event_id,
                parent_event_id=event.get("parent_event_id"),
                session_id=event.get("session_id"),
                camera_id=event["camera_id"],
                trigger_frame_id=event.get("frame_id"),
                clip_artifact_id=clip_artifact.get("artifact_id"),
                clip_path=clip_path,
                model_id=None,
                status="skipped",
                experiment=experiment,
                source_refs=source_refs,
                error="not_configured",
            )
            return {"status": "skipped", "experiment_id": experiment_id, "experiment": experiment, "source_refs": source_refs, "model_id": "smolvlm2-disabled"}
        try:
            related = self.db.related_events(event_id)
            raw = self.vlm.analyze_clip(
                clip_manifest_path=clip_path,
                event=event,
                related_events=related,
                clip_artifact=clip_artifact,
                frame_artifacts=frame_artifacts,
            )
            import json
            manifest = json.loads(Path(clip_path).read_text(encoding="utf-8"))
            raw_for_validation = dict(raw)
            raw_for_validation.pop("_validated", None)
            experiment = validate_smolvlm2_short_clip_json(
                raw_for_validation,
                event=event,
                related_events=related,
                clip_artifact=clip_artifact,
                frame_artifacts=frame_artifacts,
                clip_manifest=manifest,
            )
            experiment_id = self.db.create_smolvlm2_clip_experiment(
                event_id=event_id,
                parent_event_id=event.get("parent_event_id"),
                session_id=event.get("session_id"),
                camera_id=event["camera_id"],
                trigger_frame_id=event.get("frame_id"),
                clip_artifact_id=clip_artifact.get("artifact_id"),
                clip_path=clip_path,
                model_id=getattr(self.vlm, "model_id", "smolvlm2"),
                status="completed",
                experiment=experiment,
                source_refs=source_refs,
                error=None,
            )
            return {"status": "completed", "experiment_id": experiment_id, "experiment": experiment, "source_refs": source_refs, "model_id": getattr(self.vlm, "model_id", "smolvlm2")}
        except Exception as exc:
            experiment_id = self.db.create_smolvlm2_clip_experiment(
                event_id=event_id,
                parent_event_id=event.get("parent_event_id"),
                session_id=event.get("session_id"),
                camera_id=event["camera_id"],
                trigger_frame_id=event.get("frame_id"),
                clip_artifact_id=clip_artifact.get("artifact_id"),
                clip_path=clip_path,
                model_id=getattr(self.vlm, "model_id", "smolvlm2"),
                status="failed",
                experiment={},
                source_refs=source_refs,
                error=str(exc),
            )
            return {"status": "failed", "experiment_id": experiment_id, "experiment": {}, "source_refs": source_refs, "model_id": getattr(self.vlm, "model_id", "smolvlm2"), "error": str(exc)}

    def _short_clip_artifact(self, event: dict[str, Any]) -> dict[str, Any] | None:
        session_id = event.get("session_id")
        event_id = str(event.get("event_id"))
        candidates = self.db.list_artifacts(session_id=session_id, artifact_type="short_clip_manifest") if session_id else []
        for artifact in candidates:
            path = Path(str(artifact.get("path") or ""))
            if not path.exists():
                continue
            try:
                import json
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if manifest.get("trigger_event_id") == event_id:
                return artifact
        return None

    @staticmethod
    def _frame_artifacts_from_manifest(clip_path: str, event: dict[str, Any]) -> list[dict[str, Any]]:
        import json
        path = Path(clip_path)
        if not path.exists():
            return []
        manifest = json.loads(path.read_text(encoding="utf-8"))
        out: list[dict[str, Any]] = []
        for frame in manifest.get("frames", []):
            if not isinstance(frame, dict):
                continue
            if frame.get("artifact_id"):
                out.append({
                    "artifact_id": frame.get("artifact_id"),
                    "artifact_type": "short_clip_frame",
                    "path": frame.get("path"),
                    "sha256": frame.get("sha256"),
                    "media_type": "image/jpeg",
                    "frame_id": frame.get("frame_id"),
                    "camera_id": event.get("camera_id"),
                    "session_id": event.get("session_id"),
                })
        return out

    @staticmethod
    def _source_refs(event: dict[str, Any], clip_artifact: dict[str, Any], frame_artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                "artifact_id": clip_artifact.get("artifact_id"),
                "artifact_type": clip_artifact.get("artifact_type"),
                "path": clip_artifact.get("path"),
                "sha256": clip_artifact.get("sha256"),
            },
            *[
                {
                    "kind": "artifact",
                    "artifact_id": row.get("artifact_id"),
                    "artifact_type": row.get("artifact_type"),
                    "path": row.get("path"),
                    "sha256": row.get("sha256"),
                    "frame_id": row.get("frame_id"),
                }
                for row in frame_artifacts
            ],
        ]
