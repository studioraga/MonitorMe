import os
from typing import Any

from . import __version__
from .assistant import MonitorMeAssistant
from .camera_devices import camera_start_hint, list_video_devices
from .db import MonitorMeDB
from .detector_health import check_detector_health
from .evidence_pack import EvidencePackBuilder
from .local_capture import LocalCameraCaptureRunner, LocalCaptureConfig
from .model_registry import register_default_models
from .report_tools import IncidentReportBuilder
from .tracker_tools import TrackerTools


def create_app(db_path: str | None = None):
    """Create the optional FastAPI app.

    The default API is local-only (`127.0.0.1`) when launched through
    `scripts/run_api.sh`. It never uploads CCTV frames or events externally.
    """
    try:
        from fastapi import Body, FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except Exception as exc:  # pragma: no cover - covered by optional deployment
        raise RuntimeError("FastAPI is optional. Install with: pip install -e .[api]") from exc

    db = MonitorMeDB(db_path or os.getenv("MONITORME_DB", "data/events/monitorme.db"))
    assistant = MonitorMeAssistant(db)
    evidence_builder = EvidencePackBuilder(db)
    report_builder = IncidentReportBuilder(db)
    trackers = TrackerTools(db)

    class AskRequest(BaseModel):
        question: str
        camera_id: str | None = None
        limit: int = 100
        use_llm: bool = False

    class FeedbackRequest(BaseModel):
        label: str
        reason: str = ""
        operator: str = "operator"

    class IncidentRequest(BaseModel):
        event_ids: list[str]
        title: str | None = None
        severity: str = "info"

    class CaptureRequest(BaseModel):
        camera_id: str = "c922_node1_gate"
        device: str = "/dev/video0"
        width: int = 1280
        height: int = 720
        fps: int = 30
        fourcc: str = "MJPG"
        duration_sec: float = Field(default=10.0, ge=0.1, le=3600.0)
        max_frames: int | None = Field(default=None, ge=1)
        motion_threshold: float = Field(default=1.5, ge=0.0)
        motion_pixel_threshold: int = Field(default=30, ge=1, le=255)
        min_event_gap_sec: float = Field(default=2.0, ge=0.0)
        data_root: str = "data"
        detector_enabled: bool = False
        detector_model_id: str = "yolo11n-coco-onnx"
        detector_model_path: str = "models/object_detection/yolo11n.onnx"
        detector_conf_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
        detector_iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
        detector_max_detections: int = Field(default=20, ge=1, le=100)
        detector_input_size: int = Field(default=640, ge=64, le=2048)
        overlay_enabled: bool = True
        overlay_dir_name: str = "overlays"

    app = FastAPI(title="MonitorMe Node1 Local Evidence Assistant", version=__version__)

    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "MonitorMe",
            "version": __version__,
            "description": "Node1 real C922 local evidence assistant API",
            "db_path": str(db.db_path),
            "routes": {
                "health": "/health",
                "docs": "/docs",
                "openapi": "/openapi.json",
                "models": "/models",
                "detector_health": "/models/detector/health",
                "camera_devices": "/camera/devices",
                "capture_run": "POST /camera/capture/run",
                "events": "GET /events",
                "artifacts": "GET /artifacts",
                "ask": "POST /assistant/ask",
                "evidence_pack": "POST /assistant/events/{event_id}/evidence-pack",
                "incident_report": "POST /assistant/reports/incident",
                "feedback": "POST /events/{event_id}/feedback",
            },
            "privacy": {
                "external_upload": False,
                "face_recognition": False,
                "raw_frame_upload": False,
                "object_labels_fabricated": False,
                "step17c_yolo_onnx_after_motion_gate": True,
                "step17e_evidence_overlays": True,
            },
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "service": "MonitorMe", "version": __version__, "db_path": str(db.db_path)}

    @app.get("/camera/devices")
    def camera_devices(probe: bool = False) -> dict[str, Any]:
        devices = list_video_devices(probe=probe)
        return {
            "ok": True,
            "devices": devices,
            "count": len(devices),
            "hint": camera_start_hint(devices),
            "recommended_next_commands": [
                "v4l2-ctl --list-devices",
                "v4l2-ctl --device=/dev/video0 --list-formats-ext",
                "v4l2-ctl --device=/dev/video1 --list-formats-ext",
            ],
        }

    @app.post("/camera/capture/run")
    def capture_run(req: CaptureRequest = Body(...)) -> dict[str, Any]:
        config = LocalCaptureConfig(**req.model_dump())
        result = LocalCameraCaptureRunner(db, config).run().as_dict()
        if not result["ok"]:
            raise HTTPException(status_code=500, detail=result)
        return result

    @app.post("/models/register-defaults")
    def models_register_defaults() -> dict[str, Any]:
        register_default_models(db)
        return {"models": db.list_models()}

    @app.get("/models")
    def models() -> dict[str, Any]:
        return {"models": db.list_models()}

    @app.get("/models/detector/health")
    def detector_health(
        model_path: str = "models/object_detection/yolo11n.onnx",
        model_id: str = "yolo11n-coco-onnx",
        sha256: str = "",
        load_model: bool = True,
    ) -> dict[str, Any]:
        return check_detector_health(
            model_path=model_path,
            model_id=model_id,
            expected_sha256=sha256 or None,
            load_model=load_model,
        )

    @app.get("/events")
    def events(camera_id: str | None = None, event_type: str | None = None, label: str | None = None, session_id: str | None = None, limit: int = 100) -> dict[str, Any]:
        return {"events": db.list_events(camera_id=camera_id, event_type=event_type, label=label, session_id=session_id, limit=limit)}


    @app.get("/artifacts")
    def artifacts(session_id: str | None = None, camera_id: str | None = None, event_id: str | None = None, artifact_type: str | None = None, limit: int = 100) -> dict[str, Any]:
        items = db.list_artifacts(session_id=session_id, camera_id=camera_id, event_id=event_id, artifact_type=artifact_type)
        return {"artifacts": items[:limit], "count": min(len(items), limit)}

    @app.get("/sessions/{session_id}")
    def session(session_id: str) -> dict[str, Any]:
        row = db.get_session(session_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"session_id not found: {session_id}")
        return row

    @app.post("/assistant/ask")
    def ask(req: AskRequest = Body(...)) -> dict[str, Any]:
        answer = assistant.ask(req.question, camera_id=req.camera_id, limit=req.limit, use_llm=req.use_llm)
        return {"run_id": answer.run_id, "answer": answer.answer, "evidence": answer.evidence, "limits": answer.limits}

    @app.post("/events/{event_id}/feedback")
    def feedback(event_id: str, req: FeedbackRequest = Body(...)) -> dict[str, Any]:
        try:
            feedback_id = trackers.mark_event(event_id, label=req.label, reason=req.reason, operator=req.operator)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"feedback_id": feedback_id, "event_id": event_id, "label": req.label}

    @app.get("/trackers/false-positives")
    def false_positives() -> dict[str, Any]:
        return {"items": trackers.false_positive_tracker()}

    @app.post("/assistant/events/{event_id}/evidence-pack")
    def evidence_pack(event_id: str) -> dict[str, Any]:
        try:
            return evidence_builder.build_for_event(event_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/assistant/reports/incident")
    def incident(req: IncidentRequest = Body(...)) -> dict[str, Any]:
        try:
            return report_builder.build(event_ids=req.event_ids, title=req.title, severity=req.severity)
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


app = create_app() if os.getenv("MONITORME_CREATE_APP_AT_IMPORT") == "1" else None
