import os
from typing import Any

from . import __version__
from .assistant import MonitorMeAssistant
from .assistant_summary import AssistantSummaryService
from .camera_devices import camera_start_hint, list_video_devices
from .db import MonitorMeDB
from .detector_health import check_detector_health
from .evidence_pack import EvidencePackBuilder
from .local_capture import LocalCameraCaptureRunner, LocalCaptureConfig
from .llm_client import gemma_max_health
from .keyframe_vlm import KeyframeVLMAnalysisService
from .vlm_client import qwen_vlm_health
from .short_clip_vlm import ShortClipVLMExperimentService
from .smolvlm2_client import smolvlm2_health
from .model_registry import register_default_models
from .operator_dashboard import build_operator_dashboard_context, render_operator_dashboard_html
from .report_tools import IncidentReportBuilder
from .tracker_tools import TrackerTools


def create_app(db_path: str | None = None):
    """Create the optional FastAPI app.

    The default API is local-only (`127.0.0.1`) when launched through
    `scripts/run_api.sh`. It never uploads CCTV frames or events externally.
    """
    try:
        from fastapi import Body, FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
        from pydantic import BaseModel, Field
    except Exception as exc:  # pragma: no cover - covered by optional deployment
        raise RuntimeError("FastAPI is optional. Install with: pip install -e .[api]") from exc

    db = MonitorMeDB(db_path or os.getenv("MONITORME_DB", "data/events/monitorme.db"))
    assistant = MonitorMeAssistant(db)
    evidence_builder = EvidencePackBuilder(db)
    report_builder = IncidentReportBuilder(db)
    trackers = TrackerTools(db)
    summary_service = AssistantSummaryService(db)
    vlm_service = KeyframeVLMAnalysisService(db)
    smolvlm2_service = ShortClipVLMExperimentService(db)

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
        vlm_enabled: bool = False
        vlm_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
        smolvlm2_enabled: bool = False
        smolvlm2_model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        smolvlm2_clip_frame_count: int = Field(default=8, ge=1, le=64)

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
                "evidence_pipeline_summaries": "GET /evidence/pipeline/summaries",
                "evidence_pipeline_session_summary": "GET /evidence/pipeline/sessions/{session_id}/summary",
                "evidence_pipeline_profile_summary": "GET /evidence/pipeline/profiles/{profile_id}/summary",
                "evidence_pipeline_profile_fingerprints": "GET /evidence/pipeline/profiles/{profile_id}/fingerprints",
                "evidence_pipeline_profile_dedup_groups": "GET /evidence/pipeline/profiles/{profile_id}/dedup-groups",
                "evidence_pipeline_profile_key_moments": "GET /evidence/pipeline/profiles/{profile_id}/key-moments",
                "evidence_pipeline_retention_plan": "GET /evidence/pipeline/retention/plan",
                "evidence_pipeline_retention_apply": "POST /evidence/pipeline/retention/apply",
                "evidence_pipeline_retention_runs": "GET /evidence/pipeline/retention/runs",
                "evidence_retention_schedule": "GET /evidence/pipeline/retention/schedule",
                "evidence_retention_schedule_update": "POST /evidence/pipeline/retention/schedule",
                "evidence_retention_schedule_run": "POST /evidence/pipeline/retention/schedule/run",
                "evidence_retention_scheduler_runs": "GET /evidence/pipeline/retention/scheduler-runs",
                "evidence_index_rebuild_plan": "GET /evidence/pipeline/rebuild/plan",
                "evidence_index_rebuild_apply": "POST /evidence/pipeline/rebuild/apply",
                "evidence_index_rebuild_runs": "GET /evidence/pipeline/rebuild/runs",
                "operator_dashboard": "GET /operator/dashboard",
                "operator_dashboard_data": "GET /operator/dashboard/data",
                "ask": "POST /assistant/ask",
                "assistant_summary": "POST /assistant/events/{event_id}/summary",
                "assistant_summaries": "GET /assistant/summaries",
                "event_contracts": "GET /assistant/event-contracts",
                "llm_health": "GET /assistant/llm/health",
                "vlm_health": "GET /assistant/vlm/health",
                "vlm_analysis": "POST /assistant/events/{event_id}/vlm-analysis",
                "vlm_analyses": "GET /assistant/vlm-analyses",
                "smolvlm2_health": "GET /assistant/smolvlm2/health",
                "smolvlm2_experiment": "POST /assistant/events/{event_id}/smolvlm2-experiment",
                "smolvlm2_experiments": "GET /assistant/smolvlm2-experiments",
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
                "node1_ai_camera_assistant_v0_1": True,
                "node1_ai_camera_assistant_v0_2": True,
                "node1_ai_camera_assistant_v0_3": True,
                "node1_ai_camera_assistant_v0_4": True,
                "gemma_max_raw_frame_upload": False,
                "qwen_vlm_after_trigger_only": True,
                "qwen_vlm_enabled_by_default": False,
                "smolvlm2_after_trigger_only": True,
                "smolvlm2_enabled_by_default": False,
                "smolvlm2_short_clip_experimental": True,
                "evidence_pipeline_api_summaries": True,
                "evidence_pipeline_api_media_decode": False,
                "evidence_pipeline_api_external_upload": False,
                "evidence_index_retention_rows_only": True,
                "evidence_index_retention_deletes_media": False,
                "evidence_index_retention_scheduler_enabled": True,
                "evidence_index_retention_scheduler_default_dry_run": True,
                "evidence_index_retention_scheduler_external_upload": False,
                "evidence_index_rebuild_from_retained_artifacts": True,
                "evidence_index_rebuild_media_decode": False,
                "evidence_index_rebuild_native_rerun": False,
                "operator_dashboard_enabled": True,
                "operator_dashboard_external_assets": False,
                "operator_dashboard_media_decode": False,
                "operator_dashboard_destructive_actions": False,
                "operator_dashboard_charts": True,
                "operator_dashboard_external_chart_assets": False,
                "operator_dashboard_client_side_chart_library": False,
            },
        }

    @app.get("/operator/dashboard", response_class=HTMLResponse)
    def operator_dashboard(
        session_id: str | None = None,
        camera_id: str | None = None,
        profile_id: str | None = None,
        limit: int = 10,
        fingerprint_limit: int = 5,
        retention_limit: int = 5,
    ) -> HTMLResponse:
        context = build_operator_dashboard_context(
            db,
            session_id=session_id,
            camera_id=camera_id,
            profile_id=profile_id,
            limit=limit,
            fingerprint_limit=fingerprint_limit,
            retention_limit=retention_limit,
        )
        return HTMLResponse(render_operator_dashboard_html(context))

    @app.get("/operator/dashboard/data")
    def operator_dashboard_data(
        session_id: str | None = None,
        camera_id: str | None = None,
        profile_id: str | None = None,
        limit: int = 10,
        fingerprint_limit: int = 5,
        retention_limit: int = 5,
    ) -> dict[str, Any]:
        return build_operator_dashboard_context(
            db,
            session_id=session_id,
            camera_id=camera_id,
            profile_id=profile_id,
            limit=limit,
            fingerprint_limit=fingerprint_limit,
            retention_limit=retention_limit,
        )

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

    @app.get("/evidence/pipeline/summaries")
    def evidence_pipeline_summaries(
        profile_id: str | None = None,
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 50,
        detailed: bool = False,
    ) -> dict[str, Any]:
        safe_limit = max(1, min(int(limit), 500))
        items = db.summarize_evidence_pipeline_profiles(
            profile_id=profile_id,
            event_id=event_id,
            session_id=session_id,
            camera_id=camera_id,
            limit=safe_limit,
            detailed=detailed,
        )
        return {
            "ok": True,
            "schema": "monitorme.api.evidence_pipeline_summaries.v0.1",
            "evidence_pipeline_summaries": items,
            "count": len(items),
            "filters": {
                "profile_id": profile_id,
                "event_id": event_id,
                "session_id": session_id,
                "camera_id": camera_id,
                "limit": safe_limit,
                "detailed": detailed,
            },
            "privacy": {
                "external_upload": False,
                "raw_frame_upload": False,
                "media_decode_in_api": False,
                "semantic_claims": False,
                "facts_only": True,
            },
        }

    @app.get("/evidence/pipeline/sessions/{session_id}/summary")
    def evidence_pipeline_session_summary(
        session_id: str,
        latest_only: bool = True,
        detailed: bool = True,
        include_key_moments: bool = True,
        include_dedup_groups: bool = True,
        include_fingerprints: bool = False,
        fingerprint_limit: int = 20,
    ) -> dict[str, Any]:
        profiles = db.list_evidence_profiles(session_id=session_id, limit=1 if latest_only else 100)
        if not profiles:
            raise HTTPException(status_code=404, detail=f"evidence pipeline profile not found for session_id: {session_id}")
        summaries: list[dict[str, Any]] = []
        safe_fingerprint_limit = max(1, min(int(fingerprint_limit), 500))
        for profile in profiles:
            summary = db.get_evidence_pipeline_summary(
                str(profile["profile_id"]),
                include_fingerprints=include_fingerprints,
                include_dedup_groups=include_dedup_groups,
                include_key_moments=include_key_moments,
                fingerprint_limit=safe_fingerprint_limit,
                detailed=detailed,
            )
            if summary:
                summaries.append(summary)
        return {
            "ok": True,
            "schema": "monitorme.api.evidence_pipeline_session_summary.v0.1",
            "session_id": session_id,
            "latest_only": latest_only,
            "evidence_pipeline_summaries": summaries,
            "count": len(summaries),
            "privacy": {"external_upload": False, "media_decode_in_api": False, "facts_only": True},
        }

    @app.get("/evidence/pipeline/profiles/{profile_id}/summary")
    def evidence_pipeline_profile_summary(
        profile_id: str,
        detailed: bool = True,
        include_key_moments: bool = True,
        include_dedup_groups: bool = True,
        include_fingerprints: bool = False,
        fingerprint_limit: int = 20,
    ) -> dict[str, Any]:
        safe_fingerprint_limit = max(1, min(int(fingerprint_limit), 500))
        summary = db.get_evidence_pipeline_summary(
            profile_id,
            include_fingerprints=include_fingerprints,
            include_dedup_groups=include_dedup_groups,
            include_key_moments=include_key_moments,
            fingerprint_limit=safe_fingerprint_limit,
            detailed=detailed,
        )
        if not summary:
            raise HTTPException(status_code=404, detail=f"evidence pipeline profile_id not found: {profile_id}")
        return {
            "ok": True,
            "schema": "monitorme.api.evidence_pipeline_profile_summary.v0.1",
            "evidence_pipeline_summary": summary,
            "privacy": {"external_upload": False, "media_decode_in_api": False, "facts_only": True},
        }

    @app.get("/evidence/pipeline/profiles/{profile_id}/fingerprints")
    def evidence_pipeline_profile_fingerprints(
        profile_id: str,
        from_media: bool | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        if not db.get_evidence_profile(profile_id):
            raise HTTPException(status_code=404, detail=f"evidence pipeline profile_id not found: {profile_id}")
        safe_limit = max(1, min(int(limit), 1000))
        items = db.list_evidence_fingerprints(profile_id=profile_id, from_media=from_media, limit=safe_limit)
        return {
            "ok": True,
            "schema": "monitorme.api.evidence_pipeline_fingerprints.v0.1",
            "profile_id": profile_id,
            "evidence_fingerprints": items,
            "count": len(items),
            "privacy": {"external_upload": False, "media_decode_in_api": False, "facts_only": True},
        }

    @app.get("/evidence/pipeline/profiles/{profile_id}/dedup-groups")
    def evidence_pipeline_profile_dedup_groups(profile_id: str, limit: int = 100) -> dict[str, Any]:
        if not db.get_evidence_profile(profile_id):
            raise HTTPException(status_code=404, detail=f"evidence pipeline profile_id not found: {profile_id}")
        safe_limit = max(1, min(int(limit), 1000))
        items = db.list_evidence_dedup_groups(profile_id=profile_id, limit=safe_limit)
        return {
            "ok": True,
            "schema": "monitorme.api.evidence_pipeline_dedup_groups.v0.1",
            "profile_id": profile_id,
            "evidence_dedup_groups": items,
            "count": len(items),
            "privacy": {"external_upload": False, "media_decode_in_api": False, "facts_only": True},
        }

    @app.get("/evidence/pipeline/profiles/{profile_id}/key-moments")
    def evidence_pipeline_profile_key_moments(profile_id: str, limit: int = 100) -> dict[str, Any]:
        if not db.get_evidence_profile(profile_id):
            raise HTTPException(status_code=404, detail=f"evidence pipeline profile_id not found: {profile_id}")
        safe_limit = max(1, min(int(limit), 1000))
        items = db.list_evidence_key_moments(profile_id=profile_id, limit=safe_limit)
        return {
            "ok": True,
            "schema": "monitorme.api.evidence_pipeline_key_moments.v0.1",
            "profile_id": profile_id,
            "evidence_key_moments": items,
            "count": len(items),
            "privacy": {"external_upload": False, "media_decode_in_api": False, "facts_only": True},
        }

    @app.get("/evidence/pipeline/retention/plan")
    def evidence_pipeline_retention_plan(
        older_than_days: int | None = None,
        keep_last_per_camera: int = 1,
        keep_last_per_session: int = 1,
        profile_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        return db.plan_evidence_index_retention(
            older_than_days=older_than_days,
            keep_last_per_camera=keep_last_per_camera,
            keep_last_per_session=keep_last_per_session,
            profile_id=profile_id,
            session_id=session_id,
            camera_id=camera_id,
            limit=limit,
        )

    @app.post("/evidence/pipeline/retention/apply")
    def evidence_pipeline_retention_apply(
        dry_run: bool = True,
        confirm: bool = False,
        older_than_days: int | None = None,
        keep_last_per_camera: int = 1,
        keep_last_per_session: int = 1,
        profile_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int = 1000,
        compact: bool = True,
        vacuum: bool = False,
    ) -> dict[str, Any]:
        if not dry_run and not confirm:
            raise HTTPException(status_code=400, detail="confirm=true is required when dry_run=false")
        return db.apply_evidence_index_retention(
            dry_run=dry_run,
            older_than_days=older_than_days,
            keep_last_per_camera=keep_last_per_camera,
            keep_last_per_session=keep_last_per_session,
            profile_id=profile_id,
            session_id=session_id,
            camera_id=camera_id,
            limit=limit,
            compact=compact,
            vacuum=vacuum,
        )

    @app.get("/evidence/pipeline/retention/runs")
    def evidence_pipeline_retention_runs(
        run_id: str | None = None,
        dry_run: bool | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        items = db.list_evidence_retention_runs(run_id=run_id, dry_run=dry_run, status=status, limit=limit)
        return {
            "ok": True,
            "schema": "monitorme.api.evidence_index_retention_runs.v0.1",
            "evidence_retention_runs": items,
            "count": len(items),
            "privacy": {"external_upload": False, "media_decode_in_api": False, "facts_only": True},
        }


    @app.get("/evidence/pipeline/retention/schedule")
    def evidence_pipeline_retention_schedule(schedule_id: str = "default") -> dict[str, Any]:
        schedule = db.get_evidence_retention_schedule(schedule_id)
        if not schedule:
            raise HTTPException(status_code=404, detail=f"retention schedule not found: {schedule_id}")
        return {"ok": True, "schema": "monitorme.api.evidence_index_retention_schedule.v0.1", "schedule": schedule}

    @app.post("/evidence/pipeline/retention/schedule")
    def evidence_pipeline_retention_schedule_update(
        schedule_id: str = "default",
        enabled: bool | None = None,
        cadence: str | None = None,
        older_than_days: int | None = None,
        keep_last_per_camera: int | None = None,
        keep_last_per_session: int | None = None,
        profile_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        limit: int | None = None,
        dry_run: bool | None = None,
        compact: bool | None = None,
        vacuum: bool | None = None,
        next_run_after: str | None = None,
        notes: str | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if dry_run is False and not confirm:
            raise HTTPException(status_code=400, detail="confirm=true is required to schedule destructive retention apply")
        try:
            schedule = db.configure_evidence_retention_schedule(
                schedule_id=schedule_id,
                enabled=enabled,
                cadence=cadence,
                older_than_days=older_than_days,
                keep_last_per_camera=keep_last_per_camera,
                keep_last_per_session=keep_last_per_session,
                profile_id=profile_id,
                session_id=session_id,
                camera_id=camera_id,
                limit=limit,
                dry_run=dry_run,
                compact=compact,
                vacuum=vacuum,
                next_run_after=next_run_after,
                notes=notes,
                allow_destructive=confirm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "schema": "monitorme.api.evidence_index_retention_schedule.v0.1", "schedule": schedule}

    @app.post("/evidence/pipeline/retention/schedule/run")
    def evidence_pipeline_retention_schedule_run(
        schedule_id: str = "default",
        force: bool = False,
        dry_run: bool | None = None,
        confirm: bool = False,
    ) -> dict[str, Any]:
        if dry_run is False and not confirm:
            raise HTTPException(status_code=400, detail="confirm=true is required to run destructive scheduled retention")
        try:
            return db.run_evidence_retention_schedule(
                schedule_id=schedule_id,
                force=force,
                dry_run_override=dry_run,
                allow_destructive=confirm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/evidence/pipeline/retention/scheduler-runs")
    def evidence_pipeline_retention_scheduler_runs(
        schedule_id: str | None = None,
        scheduler_run_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        items = db.list_evidence_retention_scheduler_runs(
            schedule_id=schedule_id,
            scheduler_run_id=scheduler_run_id,
            status=status,
            limit=limit,
        )
        return {
            "ok": True,
            "schema": "monitorme.api.evidence_index_retention_scheduler_runs.v0.1",
            "evidence_retention_scheduler_runs": items,
            "count": len(items),
            "privacy": {"external_upload": False, "media_decode_in_api": False, "facts_only": True},
        }


    @app.get("/evidence/pipeline/rebuild/plan")
    def evidence_pipeline_rebuild_plan(
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        include_existing: bool = False,
        artifact_root: str = ".",
        limit: int = 100,
    ) -> dict[str, Any]:
        return db.plan_evidence_index_rebuild(
            event_id=event_id,
            session_id=session_id,
            camera_id=camera_id,
            missing_only=not include_existing,
            artifact_root=artifact_root,
            limit=max(1, min(int(limit), 1000)),
        )

    @app.post("/evidence/pipeline/rebuild/apply")
    def evidence_pipeline_rebuild_apply(
        dry_run: bool = True,
        confirm: bool = False,
        replace_existing: bool = False,
        event_id: str | None = None,
        session_id: str | None = None,
        camera_id: str | None = None,
        artifact_root: str = ".",
        limit: int = 100,
    ) -> dict[str, Any]:
        if not dry_run and not confirm:
            raise HTTPException(status_code=400, detail="confirm=true is required when dry_run=false")
        if replace_existing and not confirm:
            raise HTTPException(status_code=400, detail="confirm=true is required when replace_existing=true")
        return db.rebuild_evidence_index_from_artifacts(
            dry_run=dry_run,
            replace_existing=replace_existing,
            event_id=event_id,
            session_id=session_id,
            camera_id=camera_id,
            artifact_root=artifact_root,
            limit=max(1, min(int(limit), 1000)),
        )

    @app.get("/evidence/pipeline/rebuild/runs")
    def evidence_pipeline_rebuild_runs(
        run_id: str | None = None,
        dry_run: bool | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        items = db.list_evidence_index_rebuild_runs(run_id=run_id, dry_run=dry_run, status=status, limit=max(1, min(int(limit), 1000)))
        return {
            "ok": True,
            "schema": "monitorme.api.evidence_index_rebuild_runs.v0.1",
            "evidence_index_rebuild_runs": items,
            "count": len(items),
            "privacy": {"external_upload": False, "media_decode_in_api": False, "native_rerun": False, "facts_only": True},
        }

    @app.get("/sessions/{session_id}")
    def session(session_id: str) -> dict[str, Any]:
        row = db.get_session(session_id)
        if not row:
            raise HTTPException(status_code=404, detail=f"session_id not found: {session_id}")
        return row


    @app.get("/assistant/llm/health")
    def assistant_llm_health(probe: bool = False) -> dict[str, Any]:
        return gemma_max_health(probe=probe)

    @app.get("/assistant/vlm/health")
    def assistant_vlm_health(probe: bool = False) -> dict[str, Any]:
        return qwen_vlm_health(probe=probe)

    @app.get("/assistant/smolvlm2/health")
    def assistant_smolvlm2_health(probe: bool = False) -> dict[str, Any]:
        return smolvlm2_health(probe=probe)

    @app.post("/assistant/ask")
    def ask(req: AskRequest = Body(...)) -> dict[str, Any]:
        answer = assistant.ask(req.question, camera_id=req.camera_id, limit=req.limit, use_llm=req.use_llm)
        return {"run_id": answer.run_id, "answer": answer.answer, "evidence": answer.evidence, "limits": answer.limits}

    @app.post("/assistant/events/{event_id}/summary")
    def assistant_event_summary(event_id: str) -> dict[str, Any]:
        try:
            return summary_service.summarize_event(event_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/assistant/summaries")
    def assistant_summaries(event_id: str | None = None, session_id: str | None = None, camera_id: str | None = None, limit: int = 100) -> dict[str, Any]:
        items = db.list_summaries(event_id=event_id, session_id=session_id, camera_id=camera_id, limit=limit)
        return {"summaries": items, "count": len(items)}

    @app.get("/assistant/event-contracts")
    def assistant_event_contracts(event_id: str | None = None, session_id: str | None = None, camera_id: str | None = None, limit: int = 100) -> dict[str, Any]:
        items = db.list_event_contracts(event_id=event_id, session_id=session_id, camera_id=camera_id, limit=limit)
        return {"event_contracts": items, "count": len(items)}

    @app.post("/assistant/events/{event_id}/vlm-analysis")
    def assistant_event_vlm_analysis(event_id: str) -> dict[str, Any]:
        try:
            return vlm_service.analyze_event(event_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/assistant/vlm-analyses")
    def assistant_vlm_analyses(event_id: str | None = None, session_id: str | None = None, camera_id: str | None = None, artifact_id: str | None = None, status: str | None = None, limit: int = 100) -> dict[str, Any]:
        items = db.list_vlm_analyses(event_id=event_id, session_id=session_id, camera_id=camera_id, artifact_id=artifact_id, status=status, limit=limit)
        return {"vlm_analyses": items, "count": len(items)}

    @app.post("/assistant/events/{event_id}/smolvlm2-experiment")
    def assistant_event_smolvlm2_experiment(event_id: str) -> dict[str, Any]:
        try:
            return smolvlm2_service.analyze_event(event_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/assistant/smolvlm2-experiments")
    def assistant_smolvlm2_experiments(event_id: str | None = None, session_id: str | None = None, camera_id: str | None = None, clip_artifact_id: str | None = None, status: str | None = None, limit: int = 100) -> dict[str, Any]:
        items = db.list_smolvlm2_clip_experiments(event_id=event_id, session_id=session_id, camera_id=camera_id, clip_artifact_id=clip_artifact_id, status=status, limit=limit)
        return {"smolvlm2_clip_experiments": items, "count": len(items)}

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
