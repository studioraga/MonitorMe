from __future__ import annotations

import argparse
import json

from .assistant import MonitorMeAssistant
from .assistant_summary import AssistantSummaryService
from .camera_devices import camera_start_hint, list_video_devices
from .db import MonitorMeDB
from .detector_health import check_detector_health
from .evidence_pack import EvidencePackBuilder
from .local_capture import LocalCameraCaptureRunner, LocalCaptureConfig
from .llm_client import GemmaMaxConfig, gemma_max_health
from .keyframe_vlm import KeyframeVLMAnalysisService
from .vlm_client import QwenVLMConfig, qwen_vlm_health
from .short_clip_vlm import ShortClipVLMExperimentService
from .smolvlm2_client import SmolVLM2Config, smolvlm2_health
from .model_registry import register_default_models
from .non_llm_gpu_lab import GpuLabConfig, Node1NonLLMGpuLabRunner, gpu_lab_health
from .report_tools import IncidentReportBuilder
from .tracker_tools import TrackerTools


def _db(args: argparse.Namespace) -> MonitorMeDB:
    return MonitorMeDB(args.db)


def cmd_camera_devices(args: argparse.Namespace) -> int:
    devices = list_video_devices(probe=args.probe)
    print(json.dumps({"ok": True, "devices": devices, "count": len(devices), "hint": camera_start_hint(devices)}, indent=2, sort_keys=True))
    return 0



def cmd_detector_health(args: argparse.Namespace) -> int:
    result = check_detector_health(
        model_path=args.model_path,
        model_id=args.model_id,
        expected_sha256=args.sha256 or None,
        load_model=not args.skip_load,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else (0 if args.allow_unhealthy else 3)


def cmd_llm_health(args: argparse.Namespace) -> int:
    config = GemmaMaxConfig.from_env()
    result = gemma_max_health(config, probe=args.probe)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else (0 if args.allow_unconfigured else 3)


def cmd_vlm_health(args: argparse.Namespace) -> int:
    config = QwenVLMConfig.from_env()
    result = qwen_vlm_health(config, probe=args.probe)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else (0 if args.allow_unconfigured else 3)


def cmd_smolvlm2_health(args: argparse.Namespace) -> int:
    config = SmolVLM2Config.from_env()
    result = smolvlm2_health(config, probe=args.probe)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else (0 if args.allow_unconfigured else 3)


def cmd_gpu_lab_health(args: argparse.Namespace) -> int:
    result = gpu_lab_health(probe=args.probe, enabled=args.enabled)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else (0 if args.allow_unavailable else 3)


def cmd_gpu_lab_synthetic(args: argparse.Namespace) -> int:
    config = GpuLabConfig.from_env(enabled=True)
    runner = Node1NonLLMGpuLabRunner(config)
    result = runner.run_synthetic(scenario=args.scenario)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3


def cmd_gpu_lab_sparse_roi_synthetic(args: argparse.Namespace) -> int:
    config = GpuLabConfig.from_env(enabled=True)
    runner = Node1NonLLMGpuLabRunner(config)
    result = runner.run_sparse_roi_synthetic(
        scenario=args.scenario,
        width=args.width,
        height=args.height,
        target_width=args.target_width,
        target_height=args.target_height,
        max_rois=args.max_rois,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3

def cmd_gpu_lab_mixed_region_synthetic(args: argparse.Namespace) -> int:
    config = GpuLabConfig.from_env(enabled=True)
    runner = Node1NonLLMGpuLabRunner(config)
    result = runner.run_mixed_region_synthetic(
        scenario=args.scenario,
        width=args.width,
        height=args.height,
        target_width=args.target_width,
        target_height=args.target_height,
        max_groups=args.max_groups,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3


def cmd_gpu_lab_dense_full_frame_synthetic(args: argparse.Namespace) -> int:
    config = GpuLabConfig.from_env(enabled=True)
    runner = Node1NonLLMGpuLabRunner(config)
    result = runner.run_dense_full_frame_synthetic(
        scenario=args.scenario,
        width=args.width,
        height=args.height,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3


def cmd_gpu_lab_overlay_heavy_synthetic(args: argparse.Namespace) -> int:
    config = GpuLabConfig.from_env(enabled=True)
    runner = Node1NonLLMGpuLabRunner(config)
    result = runner.run_overlay_heavy_synthetic(
        scenario=args.scenario,
        width=args.width,
        height=args.height,
        thumbnail_width=args.thumbnail_width,
        thumbnail_height=args.thumbnail_height,
        alpha=args.alpha,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3

def cmd_gpu_lab_audiobox_synthetic(args: argparse.Namespace) -> int:
    config = GpuLabConfig.from_env(enabled=True)
    runner = Node1NonLLMGpuLabRunner(config)
    result = runner.run_audiobox_synthetic(
        audio_samples=args.audio_samples,
        sample_rate=args.sample_rate,
        window_samples=args.window_samples,
        silence_threshold=args.silence_threshold,
        onset_threshold=args.onset_threshold,
        max_windows=args.max_windows,
        max_lag=args.max_lag,
        sync_drift_samples=args.sync_drift_samples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3


def cmd_gpu_lab_storage_batch_synthetic(args: argparse.Namespace) -> int:
    config = GpuLabConfig.from_env(enabled=True)
    runner = Node1NonLLMGpuLabRunner(config)
    result = runner.run_storage_batch_synthetic(
        clips=args.clips,
        max_batch_bytes=args.max_batch_bytes,
        max_batch_clips=args.max_batch_clips,
        key_moments=args.key_moments,
        min_key_gap_ms=args.min_key_gap_ms,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3


def cmd_gpu_lab_evidence_pipeline_synthetic(args: argparse.Namespace) -> int:
    config = GpuLabConfig.from_env(enabled=True)
    runner = Node1NonLLMGpuLabRunner(config)
    result = runner.run_evidence_pipeline_synthetic(
        clips=args.clips,
        max_batch_bytes=args.max_batch_bytes,
        max_batch_clips=args.max_batch_clips,
        key_moments=args.key_moments,
        min_key_gap_ms=args.min_key_gap_ms,
        dedup_hamming_threshold=args.dedup_hamming_threshold,
        fingerprint_width=args.fingerprint_width,
        fingerprint_height=args.fingerprint_height,
        fingerprint_cycle=args.fingerprint_cycle,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3

def cmd_gpu_lab_isp_synthetic(args: argparse.Namespace) -> int:
    config = GpuLabConfig.from_env(enabled=True)
    runner = Node1NonLLMGpuLabRunner(config)
    result = runner.run_isp_synthetic(filter_name=args.filter, width=args.width, height=args.height)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 3


def cmd_init_db(args: argparse.Namespace) -> int:
    db = _db(args)
    register_default_models(db)
    print(json.dumps({"ok": True, "db": str(db.db_path), "models": db.list_models()}, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_capture_run(args: argparse.Namespace) -> int:
    db = _db(args)
    config = LocalCaptureConfig(
        camera_id=args.camera_id,
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        fourcc=args.fourcc,
        duration_sec=args.duration_sec,
        max_frames=args.max_frames,
        motion_threshold=args.motion_threshold,
        motion_pixel_threshold=args.motion_pixel_threshold,
        min_event_gap_sec=args.min_event_gap_sec,
        data_root=args.data_root,
        detector_enabled=args.detector_enabled,
        detector_model_id=args.detector_model_id,
        detector_model_path=args.detector_model_path,
        detector_conf_threshold=args.detector_conf_threshold,
        detector_iou_threshold=args.detector_iou_threshold,
        detector_max_detections=args.detector_max_detections,
        detector_input_size=args.detector_input_size,
        overlay_enabled=not args.no_overlays,
        overlay_dir_name=args.overlay_dir_name,
        vlm_enabled=args.vlm_enabled,
        vlm_model_id=args.vlm_model_id,
        smolvlm2_enabled=args.smolvlm2_enabled,
        smolvlm2_model_id=args.smolvlm2_model_id,
        smolvlm2_clip_frame_count=args.smolvlm2_clip_frame_count,
        gpu_lab_enabled=args.gpu_lab_enabled,
        gpu_lab_binary=args.gpu_lab_binary,
        gpu_lab_tile_cols=args.gpu_lab_tile_cols,
        gpu_lab_tile_rows=args.gpu_lab_tile_rows,
        gpu_lab_pixel_threshold=args.gpu_lab_pixel_threshold,
        gpu_lab_sparse_threshold=args.gpu_lab_sparse_threshold,
        gpu_lab_dense_threshold=args.gpu_lab_dense_threshold,
        gpu_lab_prefer_cuda=not args.gpu_lab_no_cuda,
        gpu_lab_allow_python_fallback=not args.gpu_lab_no_python_fallback,
        evidence_pipeline_enabled=args.evidence_pipeline_enabled,
        evidence_pipeline_binary=args.evidence_pipeline_binary or args.gpu_lab_binary,
        evidence_pipeline_max_batch_bytes=args.evidence_pipeline_max_batch_bytes,
        evidence_pipeline_max_batch_clips=args.evidence_pipeline_max_batch_clips,
        evidence_pipeline_key_moments=args.evidence_pipeline_key_moments,
        evidence_pipeline_min_key_gap_ms=args.evidence_pipeline_min_key_gap_ms,
        evidence_pipeline_dedup_hamming_threshold=args.evidence_pipeline_dedup_hamming_threshold,
        evidence_pipeline_fingerprint_width=args.evidence_pipeline_fingerprint_width,
        evidence_pipeline_fingerprint_height=args.evidence_pipeline_fingerprint_height,
        evidence_pipeline_fingerprint_cycle=args.evidence_pipeline_fingerprint_cycle,
        evidence_pipeline_real_fingerprint_enabled=not args.evidence_pipeline_no_real_fingerprints,
    )
    result = LocalCameraCaptureRunner(db, config).run().as_dict()
    print(json.dumps(result, indent=2, sort_keys=True))
    db.close()
    return 0 if result.get("ok") else 2


def cmd_events(args: argparse.Namespace) -> int:
    db = _db(args)
    result = db.list_events(camera_id=args.camera_id, event_type=args.event_type, label=args.label, session_id=args.session_id, limit=args.limit)
    print(json.dumps({"events": result, "count": len(result)}, indent=2, sort_keys=True))
    db.close()
    return 0



def cmd_artifacts(args: argparse.Namespace) -> int:
    db = _db(args)
    result = db.list_artifacts(
        session_id=args.session_id,
        camera_id=args.camera_id,
        event_id=args.event_id,
        artifact_type=args.artifact_type,
    )[: args.limit]
    print(json.dumps({"artifacts": result, "count": len(result)}, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_evidence_index(args: argparse.Namespace) -> int:
    db = _db(args)
    result = db.list_evidence_profiles(
        event_id=args.event_id,
        session_id=args.session_id,
        camera_id=args.camera_id,
        limit=args.limit,
    )
    print(json.dumps({"evidence_profiles": result, "count": len(result)}, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_evidence_fingerprints(args: argparse.Namespace) -> int:
    db = _db(args)
    result = db.list_evidence_fingerprints(
        profile_id=args.profile_id,
        event_id=args.event_id,
        session_id=args.session_id,
        camera_id=args.camera_id,
        from_media=args.from_media,
        limit=args.limit,
    )
    print(json.dumps({"evidence_fingerprints": result, "count": len(result)}, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_evidence_dedup_groups(args: argparse.Namespace) -> int:
    db = _db(args)
    result = db.list_evidence_dedup_groups(
        profile_id=args.profile_id,
        event_id=args.event_id,
        session_id=args.session_id,
        camera_id=args.camera_id,
        limit=args.limit,
    )
    print(json.dumps({"evidence_dedup_groups": result, "count": len(result)}, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_evidence_key_moments(args: argparse.Namespace) -> int:
    db = _db(args)
    result = db.list_evidence_key_moments(
        profile_id=args.profile_id,
        event_id=args.event_id,
        session_id=args.session_id,
        camera_id=args.camera_id,
        limit=args.limit,
    )
    print(json.dumps({"evidence_key_moments": result, "count": len(result)}, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    db = _db(args)
    assistant = MonitorMeAssistant(db)
    result = assistant.ask(args.question, camera_id=args.camera_id, limit=args.limit, use_llm=args.use_llm)
    print(json.dumps({"run_id": result.run_id, "answer": result.answer, "evidence": result.evidence, "limits": result.limits}, indent=2, sort_keys=True))
    db.close()
    return 0



def cmd_assistant_summarize_event(args: argparse.Namespace) -> int:
    db = _db(args)
    result = AssistantSummaryService(db).summarize_event(args.event_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    db.close()
    return 0 if result.get("status") == "completed" else 3


def cmd_summaries(args: argparse.Namespace) -> int:
    db = _db(args)
    items = db.list_summaries(event_id=args.event_id, session_id=args.session_id, camera_id=args.camera_id, limit=args.limit)
    print(json.dumps({"summaries": items, "count": len(items)}, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_event_contracts(args: argparse.Namespace) -> int:
    db = _db(args)
    items = db.list_event_contracts(event_id=args.event_id, session_id=args.session_id, camera_id=args.camera_id, limit=args.limit)
    print(json.dumps({"event_contracts": items, "count": len(items)}, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_vlm_analyze_event(args: argparse.Namespace) -> int:
    db = _db(args)
    result = KeyframeVLMAnalysisService(db).analyze_event(args.event_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    db.close()
    return 0 if result.get("status") in {"completed", "skipped"} else 3


def cmd_vlm_analyses(args: argparse.Namespace) -> int:
    db = _db(args)
    items = db.list_vlm_analyses(
        event_id=args.event_id,
        session_id=args.session_id,
        camera_id=args.camera_id,
        artifact_id=args.artifact_id,
        status=args.status,
        limit=args.limit,
    )
    print(json.dumps({"vlm_analyses": items, "count": len(items)}, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_smolvlm2_analyze_event(args: argparse.Namespace) -> int:
    db = _db(args)
    result = ShortClipVLMExperimentService(db).analyze_event(args.event_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    db.close()
    return 0 if result.get("status") in {"completed", "skipped"} else 3


def cmd_smolvlm2_experiments(args: argparse.Namespace) -> int:
    db = _db(args)
    items = db.list_smolvlm2_clip_experiments(
        event_id=args.event_id,
        session_id=args.session_id,
        camera_id=args.camera_id,
        clip_artifact_id=args.clip_artifact_id,
        status=args.status,
        limit=args.limit,
    )
    print(json.dumps({"smolvlm2_clip_experiments": items, "count": len(items)}, indent=2, sort_keys=True))
    db.close()
    return 0

def cmd_evidence_pack(args: argparse.Namespace) -> int:
    db = _db(args)
    result = EvidencePackBuilder(db, root=args.output_root).build_for_event(args.event_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_incident_report(args: argparse.Namespace) -> int:
    db = _db(args)
    result = IncidentReportBuilder(db, reports_root=args.reports_root, evidence_root=args.evidence_root).build(
        event_ids=args.event_id,
        title=args.title,
        severity=args.severity,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    db.close()
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    db = _db(args)
    feedback_id = TrackerTools(db).mark_event(args.event_id, label=args.label, reason=args.reason, operator=args.operator)
    print(json.dumps({"feedback_id": feedback_id, "event_id": args.event_id, "label": args.label}, indent=2, sort_keys=True))
    db.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="monitorme", description="MonitorMe Node1 AI Camera Assistant v0.4")
    parser.add_argument("--db", default="data/events/monitorme.db", help="SQLite DB path")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("camera-devices", help="List local /dev/video* camera devices")
    p.add_argument("--probe", action="store_true", help="Run v4l2-ctl --list-formats-ext when available")
    p.set_defaults(func=cmd_camera_devices)

    p = sub.add_parser("init-db", help="Apply migrations and register default model metadata")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("llm-health", help="Show local Gemma/MAX OpenAI-compatible summary configuration")
    p.add_argument("--allow-unconfigured", action="store_true", help="Return exit 0 when Gemma/MAX is not configured")
    p.add_argument("--probe", action="store_true", help="Probe the configured MAX /v1/models endpoint")
    p.set_defaults(func=cmd_llm_health)

    p = sub.add_parser("vlm-health", help="Show local Qwen VLM OpenAI-compatible keyframe-analysis configuration")
    p.add_argument("--allow-unconfigured", action="store_true", help="Return exit 0 when Qwen VLM is not configured")
    p.add_argument("--probe", action="store_true", help="Probe the configured VLM /v1/models endpoint")
    p.set_defaults(func=cmd_vlm_health)

    p = sub.add_parser("smolvlm2-health", help="Show local SmolVLM2 OpenAI-compatible short-clip experiment configuration")
    p.add_argument("--allow-unconfigured", action="store_true", help="Return exit 0 when SmolVLM2 is not configured")
    p.add_argument("--probe", action="store_true", help="Probe the configured SmolVLM2 /v1/models endpoint")
    p.set_defaults(func=cmd_smolvlm2_health)

    p = sub.add_parser("gpu-lab-health", help="Check optional native C++/CUDA non-LLM GPU workload profiler")
    p.add_argument("--enabled", action="store_true", help="Report as enabled for this health check")
    p.add_argument("--probe", action="store_true", help="Run a synthetic native smoke test if the binary exists")
    p.add_argument("--allow-unavailable", action="store_true", help="Return exit 0 when the native binary is not built yet")
    p.set_defaults(func=cmd_gpu_lab_health)

    p = sub.add_parser("gpu-lab-synthetic", help="Run native C++/CUDA sparse/mixed/dense synthetic profiler")
    p.add_argument("--scenario", default="mixed", choices=["sparse", "mixed", "dense"])
    p.set_defaults(func=cmd_gpu_lab_synthetic)


    p = sub.add_parser("gpu-lab-sparse-roi-synthetic", help="Run sparse ROI crop/resize/normalize validation; CUDA comparison is used when available")
    p.add_argument("--scenario", default="sparse", choices=["sparse", "mixed", "dense"])
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--target-width", type=int, default=16)
    p.add_argument("--target-height", type=int, default=16)
    p.add_argument("--max-rois", type=int, default=32)
    p.set_defaults(func=cmd_gpu_lab_sparse_roi_synthetic)

    p = sub.add_parser("gpu-lab-mixed-region-synthetic", help="Run mixed region connected-component grouping and grouped crop batching validation; CUDA comparison is used when available")
    p.add_argument("--scenario", default="contiguous", choices=["contiguous", "scattered", "mixed", "sparse", "dense"])
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--target-width", type=int, default=16)
    p.add_argument("--target-height", type=int, default=16)
    p.add_argument("--max-groups", type=int, default=32)
    p.set_defaults(func=cmd_gpu_lab_mixed_region_synthetic)

    p = sub.add_parser("gpu-lab-dense-full-frame-synthetic", help="Run dense full-frame diff/histogram/reduction/normalize validation; CUDA comparison is used when available")
    p.add_argument("--scenario", default="dense", choices=["dense", "mixed", "sparse"])
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.set_defaults(func=cmd_gpu_lab_dense_full_frame_synthetic)


    p = sub.add_parser("gpu-lab-overlay-heavy-synthetic", help="Run overlay-heavy alpha blend, heatmap, thumbnail, and before/after validation; CUDA comparison is used when available")
    p.add_argument("--scenario", default="mixed", choices=["mixed", "dense", "sparse"])
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--thumbnail-width", type=int, default=64)
    p.add_argument("--thumbnail-height", type=int, default=48)
    p.add_argument("--alpha", type=int, default=128)
    p.set_defaults(func=cmd_gpu_lab_overlay_heavy_synthetic)


    p = sub.add_parser("gpu-lab-audiobox-synthetic", help="Run AudioBox RMS/peak/silence/onset/cross-correlation sync-drift validation; CUDA comparison is used when available")
    p.add_argument("--audio-samples", type=int, default=32768)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--window-samples", type=int, default=1024)
    p.add_argument("--silence-threshold", type=float, default=0.02)
    p.add_argument("--onset-threshold", type=float, default=0.08)
    p.add_argument("--max-windows", type=int, default=32)
    p.add_argument("--max-lag", type=int, default=128)
    p.add_argument("--sync-drift-samples", type=int, default=64)
    p.set_defaults(func=cmd_gpu_lab_audiobox_synthetic)


    p = sub.add_parser("gpu-lab-storage-batch-synthetic", help="Run storage manifest scan, batch read planning, key moment selection, and clip timeline feature validation")
    p.add_argument("--clips", type=int, default=12)
    p.add_argument("--max-batch-bytes", type=int, default=2 * 1024 * 1024)
    p.add_argument("--max-batch-clips", type=int, default=4)
    p.add_argument("--key-moments", type=int, default=5)
    p.add_argument("--min-key-gap-ms", type=int, default=1000)
    p.set_defaults(func=cmd_gpu_lab_storage_batch_synthetic)



    p = sub.add_parser("gpu-lab-evidence-pipeline-synthetic", help="Run visual fingerprint/evidence dedup, key-moment, storage batch, latency, and safety validation")
    p.add_argument("--clips", type=int, default=12)
    p.add_argument("--max-batch-bytes", type=int, default=2 * 1024 * 1024)
    p.add_argument("--max-batch-clips", type=int, default=4)
    p.add_argument("--key-moments", type=int, default=5)
    p.add_argument("--min-key-gap-ms", type=int, default=1000)
    p.add_argument("--dedup-hamming-threshold", type=int, default=0)
    p.add_argument("--fingerprint-width", type=int, default=16)
    p.add_argument("--fingerprint-height", type=int, default=16)
    p.add_argument("--fingerprint-cycle", type=int, default=6)
    p.set_defaults(func=cmd_gpu_lab_evidence_pipeline_synthetic)

    p = sub.add_parser("gpu-lab-isp-synthetic", help="Run ISP synthetic filter validation; CUDA comparison is used when available")
    p.add_argument("--filter", default="sobel-mag", choices=["blur", "sharpen", "edge", "sobel-x", "sobel-y", "sobel-mag"])
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=48)
    p.set_defaults(func=cmd_gpu_lab_isp_synthetic)

    p = sub.add_parser("detector-health", help="Validate local YOLO ONNX detector model/runtime without opening camera")
    p.add_argument("--model-id", default="yolo11n-coco-onnx")
    p.add_argument("--model-path", default="models/object_detection/yolo11n.onnx")
    p.add_argument("--sha256", default="", help="Optional expected model SHA-256")
    p.add_argument("--skip-load", action="store_true", help="Check file/hash/runtime metadata without loading the ONNX session")
    p.add_argument("--allow-unhealthy", action="store_true", help="Return exit 0 even if health is not OK; useful for reports")
    p.set_defaults(func=cmd_detector_health)

    p = sub.add_parser("capture-run", help="Run a real bounded Node1 local camera capture from /dev/video0")
    p.add_argument("--camera-id", default="c922_node1_gate")
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--fourcc", default="MJPG")
    p.add_argument("--duration-sec", type=float, default=10.0)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--motion-threshold", type=float, default=1.5, help="Percent of changed pixels required to emit motion event")
    p.add_argument("--motion-pixel-threshold", type=int, default=30, help="Pixel-difference threshold 1..255")
    p.add_argument("--min-event-gap-sec", type=float, default=2.0)
    p.add_argument("--data-root", default="data")
    p.add_argument("--detector-enabled", action="store_true", help="Run real YOLO ONNX detection after motion gate")
    p.add_argument("--detector-model-id", default="yolo11n-coco-onnx")
    p.add_argument("--detector-model-path", default="models/object_detection/yolo11n.onnx")
    p.add_argument("--detector-conf-threshold", type=float, default=0.35)
    p.add_argument("--detector-iou-threshold", type=float, default=0.45)
    p.add_argument("--detector-max-detections", type=int, default=20)
    p.add_argument("--detector-input-size", type=int, default=640)
    p.add_argument("--no-overlays", action="store_true", help="Disable Step 17E annotated keyframe overlays")
    p.add_argument("--overlay-dir-name", default="overlays", help="Session subdirectory for annotated keyframe overlays")
    p.add_argument("--vlm-enabled", action="store_true", help="Run optional Qwen VLM analysis after trigger keyframes")
    p.add_argument("--vlm-model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--smolvlm2-enabled", action="store_true", help="Run optional SmolVLM2 short clip experiment after trigger")
    p.add_argument("--smolvlm2-model-id", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    p.add_argument("--smolvlm2-clip-frame-count", type=int, default=8)
    p.add_argument("--gpu-lab-enabled", action="store_true", help="Run non-LLM C++/CUDA workload profiler after motion trigger")
    p.add_argument("--gpu-lab-binary", default="", help="Path to native node1_non_llm_gpu_lab binary")
    p.add_argument("--gpu-lab-tile-cols", type=int, default=8)
    p.add_argument("--gpu-lab-tile-rows", type=int, default=4)
    p.add_argument("--gpu-lab-pixel-threshold", type=int, default=30)
    p.add_argument("--gpu-lab-sparse-threshold", type=int, default=8)
    p.add_argument("--gpu-lab-dense-threshold", type=int, default=24)
    p.add_argument("--gpu-lab-no-cuda", action="store_true", help="Do not request CUDA backend from native profiler")
    p.add_argument("--gpu-lab-no-python-fallback", action="store_true", help="Do not emit Python fallback workload profile if native binary is missing")
    p.add_argument("--evidence-pipeline-enabled", action="store_true", help="Run facts-only evidence pipeline after capture manifest/keyframes are written")
    p.add_argument("--evidence-pipeline-binary", default="", help="Path to native node1_non_llm_gpu_lab binary for capture-run evidence pipeline; defaults to --gpu-lab-binary/native default")
    p.add_argument("--evidence-pipeline-max-batch-bytes", type=int, default=2 * 1024 * 1024)
    p.add_argument("--evidence-pipeline-max-batch-clips", type=int, default=4)
    p.add_argument("--evidence-pipeline-key-moments", type=int, default=5)
    p.add_argument("--evidence-pipeline-min-key-gap-ms", type=int, default=1000)
    p.add_argument("--evidence-pipeline-dedup-hamming-threshold", type=int, default=0)
    p.add_argument("--evidence-pipeline-fingerprint-width", type=int, default=16)
    p.add_argument("--evidence-pipeline-fingerprint-height", type=int, default=16)
    p.add_argument("--evidence-pipeline-fingerprint-cycle", type=int, default=6)
    p.add_argument("--evidence-pipeline-no-real-fingerprints", action="store_true", help="Do not decode stored keyframes into real media fingerprints; fall back to manifest metadata fingerprints")
    p.set_defaults(func=cmd_capture_run)

    p = sub.add_parser("events", help="List normalized events")
    p.add_argument("--camera-id")
    p.add_argument("--event-type")
    p.add_argument("--label")
    p.add_argument("--session-id")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_events)


    p = sub.add_parser("artifacts", help="List capture/evidence artifacts")
    p.add_argument("--session-id")
    p.add_argument("--camera-id")
    p.add_argument("--event-id")
    p.add_argument("--artifact-type")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_artifacts)


    p = sub.add_parser("evidence-index", help="List persisted evidence pipeline profile rows")
    p.add_argument("--event-id")
    p.add_argument("--session-id")
    p.add_argument("--camera-id")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_evidence_index)

    p = sub.add_parser("evidence-fingerprints", help="List persisted evidence fingerprint rows")
    p.add_argument("--profile-id")
    p.add_argument("--event-id")
    p.add_argument("--session-id")
    p.add_argument("--camera-id")
    p.add_argument("--from-media", action="store_true")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_evidence_fingerprints)

    p = sub.add_parser("evidence-dedup-groups", help="List persisted evidence duplicate groups")
    p.add_argument("--profile-id")
    p.add_argument("--event-id")
    p.add_argument("--session-id")
    p.add_argument("--camera-id")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_evidence_dedup_groups)

    p = sub.add_parser("evidence-key-moments", help="List persisted evidence key moments")
    p.add_argument("--profile-id")
    p.add_argument("--event-id")
    p.add_argument("--session-id")
    p.add_argument("--camera-id")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_evidence_key_moments)

    p = sub.add_parser("ask", help="Ask a DB-grounded MonitorMe question")
    p.add_argument("question")
    p.add_argument("--camera-id")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--use-llm", action="store_true", help="Allow configured LLM summary if fact guard accepts it")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("assistant-summarize-event", help="Create a deterministic assistant summary/event contract for an event")
    p.add_argument("event_id")
    p.set_defaults(func=cmd_assistant_summarize_event)

    p = sub.add_parser("summaries", help="List assistant summaries")
    p.add_argument("--event-id")
    p.add_argument("--session-id")
    p.add_argument("--camera-id")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_summaries)

    p = sub.add_parser("event-contracts", help="List Node1 AI camera event contracts")
    p.add_argument("--event-id")
    p.add_argument("--session-id")
    p.add_argument("--camera-id")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_event_contracts)

    p = sub.add_parser("vlm-analyze-event", help="Run optional Qwen VLM analysis for a stored trigger keyframe")
    p.add_argument("event_id")
    p.set_defaults(func=cmd_vlm_analyze_event)

    p = sub.add_parser("vlm-analyses", help="List Qwen VLM keyframe analyses")
    p.add_argument("--event-id")
    p.add_argument("--session-id")
    p.add_argument("--camera-id")
    p.add_argument("--artifact-id")
    p.add_argument("--status", choices=["completed", "failed", "skipped"])
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_vlm_analyses)

    p = sub.add_parser("smolvlm2-analyze-event", help="Run optional SmolVLM2 short clip experiment for a stored trigger clip")
    p.add_argument("event_id")
    p.set_defaults(func=cmd_smolvlm2_analyze_event)

    p = sub.add_parser("smolvlm2-experiments", help="List SmolVLM2 short clip experiments")
    p.add_argument("--event-id")
    p.add_argument("--session-id")
    p.add_argument("--camera-id")
    p.add_argument("--clip-artifact-id")
    p.add_argument("--status", choices=["completed", "failed", "skipped"])
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_smolvlm2_experiments)

    p = sub.add_parser("evidence-pack", help="Build an evidence pack for an event")
    p.add_argument("event_id")
    p.add_argument("--output-root", default="data/evidence_packs")
    p.set_defaults(func=cmd_evidence_pack)

    p = sub.add_parser("incident-report", help="Build an incident report from one or more event ids")
    p.add_argument("--event-id", action="append", required=True)
    p.add_argument("--title")
    p.add_argument("--severity", default="info")
    p.add_argument("--reports-root", default="data/reports")
    p.add_argument("--evidence-root", default="data/evidence_packs")
    p.set_defaults(func=cmd_incident_report)

    p = sub.add_parser("feedback", help="Mark an event useful/false-positive/etc")
    p.add_argument("event_id")
    p.add_argument("--label", required=True, choices=["useful", "false_positive", "needs_review", "duplicate", "bad_bbox", "wrong_label"])
    p.add_argument("--reason", default="")
    p.add_argument("--operator", default="operator")
    p.set_defaults(func=cmd_feedback)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
