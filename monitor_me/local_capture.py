from __future__ import annotations

import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .assistant_summary import AssistantSummaryService
from .db import MonitorMeDB
from .hash_utils import sha256_file
from .keyframe_vlm import KeyframeVLMAnalysisService
from .short_clip_vlm import ShortClipVLMExperimentService
from .model_registry import register_default_models
from .non_llm_gpu_lab import (
    EVIDENCE_PIPELINE_MODEL_ID,
    EVIDENCE_PIPELINE_SCHEMA,
    GPU_LAB_MODEL_ID,
    GPU_LAB_SCHEMA,
    GpuLabConfig,
    Node1NonLLMGpuLabRunner,
)
from .overlays import OverlayBox, render_evidence_overlay
from .policy import allow_node1_local_camera
from .time_utils import now_iso
from .yolo_onnx import ObjectDetection, ObjectDetector, YoloOnnxDetector


class FrameSource(Protocol):
    """Minimal frame source protocol used by the real C922 capture runner and tests."""

    def open(self) -> None: ...
    def read(self) -> tuple[bool, Any]: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class LocalCaptureConfig:
    """Configuration for a real Node1 local camera capture run.

    Step 17C keeps the capture path evidence-first:
    - real /dev/video0 frames are read through OpenCV/V4L2
    - frame-difference motion emits a parent `motion_detected` row
    - optional YOLO ONNX runs only after motion and can emit child
      `object_detected` rows
    - if the detector is disabled, missing, or fails, MonitorMe still stores
      the parent motion event and never fabricates object labels.
    """

    camera_id: str = "c922_node1_gate"
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str = "MJPG"
    duration_sec: float = 10.0
    max_frames: int | None = None
    motion_threshold: float = 1.5
    motion_pixel_threshold: int = 30
    min_event_gap_sec: float = 2.0
    data_root: str = "data"
    write_every_motion_keyframe: bool = True

    # Step 17E evidence visualization overlays. Raw keyframes remain unchanged;
    # overlays are derived convenience artifacts for review and reports.
    overlay_enabled: bool = True
    overlay_dir_name: str = "overlays"

    # Step 17C detector settings. Disabled by default so motion capture works
    # without a model file or onnxruntime. Enable explicitly after placing the
    # real ONNX model on Node1.
    detector_enabled: bool = False
    detector_model_id: str = "yolo11n-coco-onnx"
    detector_model_path: str = "models/object_detection/yolo11n.onnx"
    detector_conf_threshold: float = 0.35
    detector_iou_threshold: float = 0.45
    detector_max_detections: int = 20
    detector_input_size: int = 640

    # Node1 Assistant v0.3 Qwen VLM settings. Disabled by default. When enabled,
    # Qwen runs only after a local trigger/keyframe has already been stored.
    vlm_enabled: bool = False
    vlm_model_id: str = "Qwen/Qwen3-VL-2B-Instruct"

    # Node1 Assistant v0.4 SmolVLM2 short clip experiment settings. Disabled by
    # default. When enabled, MonitorMe writes a small local sampled clip bundle
    # after a trigger and runs SmolVLM2 only on that stored local evidence.
    smolvlm2_enabled: bool = False
    smolvlm2_model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    smolvlm2_clip_frame_count: int = 8
    smolvlm2_clip_dir_name: str = "short_clips"

    # Node1 non-LLM C++/CUDA GPU inference lab. Disabled by default and runs
    # only after a local motion/keyframe trigger. It stores workload facts
    # such as tile masks and sparse/mixed/dense route decisions; it does not
    # identify people, infer intent, or upload frames.
    gpu_lab_enabled: bool = False
    gpu_lab_binary: str = ""
    gpu_lab_tile_cols: int = 8
    gpu_lab_tile_rows: int = 4
    gpu_lab_pixel_threshold: int = 30
    gpu_lab_sparse_threshold: int = 8
    gpu_lab_dense_threshold: int = 24
    gpu_lab_prefer_cuda: bool = True
    gpu_lab_allow_python_fallback: bool = True

    # Capture-run evidence pipeline integration. Disabled by default and runs
    # after the bounded capture manifest has collected local keyframe evidence.
    # It converts capture keyframes into a native evidence CSV manifest, runs
    # the facts-only evidence pipeline, stores the resulting JSON as an
    # artifact, and inserts a session-level evidence_pipeline_indexed event.
    evidence_pipeline_enabled: bool = False
    evidence_pipeline_binary: str = ""
    evidence_pipeline_max_batch_bytes: int = 2 * 1024 * 1024
    evidence_pipeline_max_batch_clips: int = 4
    evidence_pipeline_key_moments: int = 5
    evidence_pipeline_min_key_gap_ms: int = 1000
    evidence_pipeline_dedup_hamming_threshold: int = 0
    evidence_pipeline_fingerprint_width: int = 16
    evidence_pipeline_fingerprint_height: int = 16
    evidence_pipeline_fingerprint_cycle: int = 6


@dataclass(frozen=True)
class MotionResult:
    motion: bool
    score: float
    changed_ratio: float
    bbox: list[float] | None


@dataclass(frozen=True)
class LocalCaptureResult:
    ok: bool
    camera_id: str
    device: str
    session_id: str
    dataset_path: str
    manifest_path: str
    frames_seen: int
    frames_written: int
    motion_event_ids: list[str]
    object_event_ids: list[str]
    artifact_ids: list[str]
    artifact_paths: list[str]
    overlay_artifact_ids: list[str]
    overlay_paths: list[str]
    assistant_summary_ids: list[str]
    event_contract_ids: list[str]
    vlm_analysis_ids: list[str]
    smolvlm2_experiment_ids: list[str]
    gpu_profile_event_ids: list[str]
    evidence_pipeline_event_ids: list[str]
    evidence_pipeline_artifact_ids: list[str]
    started_at: str
    ended_at: str
    detector: dict[str, Any]
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpenCVFrameSource:
    """Real V4L2/OpenCV frame source for Node1 C922."""

    def __init__(self, config: LocalCaptureConfig):
        self.config = config
        self._cv2 = None
        self._cap = None

    def open(self) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on deployment
            raise RuntimeError("OpenCV is required for real camera capture. Install with: pip install -e '.[camera]'") from exc
        self._cv2 = cv2
        cap = cv2.VideoCapture(self.config.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"unable to open camera device: {self.config.device}")
        if self.config.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc[:4]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.config.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.config.height))
        cap.set(cv2.CAP_PROP_FPS, int(self.config.fps))
        self._cap = cap

    def read(self) -> tuple[bool, Any]:
        if self._cap is None:
            raise RuntimeError("camera source is not open")
        return self._cap.read()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = None


class IterableFrameSource:
    """Test-only frame source. It is not wired into CLI/API production paths."""

    def __init__(self, frames: Iterable[Any]):
        self._frames = iter(frames)

    def open(self) -> None:
        return None

    def read(self) -> tuple[bool, Any]:
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, None

    def close(self) -> None:
        return None


class MotionGate:
    """Simple deterministic frame-difference motion gate for local evidence."""

    def __init__(self, *, motion_threshold: float = 1.5, pixel_threshold: int = 30):
        self.motion_threshold = float(motion_threshold)
        self.pixel_threshold = int(pixel_threshold)
        self._previous_gray = None

    def evaluate(self, frame: Any) -> MotionResult:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if frame is None:
            return MotionResult(False, 0.0, 0.0, None)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self._previous_gray is None:
            self._previous_gray = gray
            return MotionResult(False, 0.0, 0.0, None)

        delta = cv2.absdiff(self._previous_gray, gray)
        self._previous_gray = gray
        thresh = cv2.threshold(delta, self.pixel_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        changed_pixels = int(np.count_nonzero(thresh))
        total_pixels = int(thresh.shape[0] * thresh.shape[1])
        changed_ratio = changed_pixels / max(total_pixels, 1)
        score = changed_ratio * 100.0
        bbox: list[float] | None = None
        if changed_pixels:
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                xs: list[int] = []
                ys: list[int] = []
                x2s: list[int] = []
                y2s: list[int] = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    xs.append(x)
                    ys.append(y)
                    x2s.append(x + w)
                    y2s.append(y + h)
                h_img, w_img = thresh.shape[:2]
                bbox = [
                    round(min(xs) / w_img, 6),
                    round(min(ys) / h_img, 6),
                    round(max(x2s) / w_img, 6),
                    round(max(y2s) / h_img, 6),
                ]
        return MotionResult(score >= self.motion_threshold, score, changed_ratio, bbox)


class LocalCameraCaptureRunner:
    """Run a bounded real local camera capture and persist MonitorMe evidence."""

    def __init__(
        self,
        db: MonitorMeDB,
        config: LocalCaptureConfig,
        *,
        frame_source: FrameSource | None = None,
        object_detector: ObjectDetector | None = None,
        keyframe_vlm_client: Any | None = None,
        short_clip_vlm_client: Any | None = None,
    ):
        self.db = db
        self.config = config
        self.frame_source = frame_source or OpenCVFrameSource(config)
        self.object_detector = object_detector
        self.keyframe_vlm_client = keyframe_vlm_client
        self.short_clip_vlm_client = short_clip_vlm_client

    def run(self) -> LocalCaptureResult:
        cfg = self.config
        started_at = now_iso()
        policy = allow_node1_local_camera(camera_id=cfg.camera_id, device=cfg.device).as_dict()
        register_default_models(self.db)
        self.db.upsert_camera(
            cfg.camera_id,
            name="Node1 Logitech C922",
            location="Node1 local USB camera",
            source_node="node1",
            source_kind="local_v4l2",
            device=cfg.device,
            enabled=True,
        )
        self.db.upsert_model(
            cfg.detector_model_id,
            role="object_detector",
            provider="onnxruntime",
            path=cfg.detector_model_path,
            metadata={
                "stage": "step17c",
                "privacy": "Runs locally on motion keyframes only; raw frames are not uploaded.",
                "enabled_for_capture": cfg.detector_enabled,
            },
            enabled=cfg.detector_enabled,
        )
        self.db.upsert_model(
            cfg.vlm_model_id,
            role="vlm_keyframe_analyzer",
            provider="qwen-openai-compatible",
            metadata={
                "stage": "node1-assistant-v0.3",
                "privacy": "Disabled by default; analyzes stored trigger keyframes only through a local OpenAI-compatible VLM endpoint.",
                "enabled_for_capture": cfg.vlm_enabled,
            },
            enabled=cfg.vlm_enabled,
        )
        self.db.upsert_model(
            cfg.smolvlm2_model_id,
            role="vlm_short_clip_experiment",
            provider="smolvlm2-openai-compatible",
            metadata={
                "stage": "node1-assistant-v0.4",
                "privacy": "Disabled by default; analyzes stored local short clip frame bundles only after trigger.",
                "enabled_for_capture": cfg.smolvlm2_enabled,
                "experimental": True,
            },
            enabled=cfg.smolvlm2_enabled,
        )
        self.db.upsert_model(
            GPU_LAB_MODEL_ID,
            role="native_gpu_workload_profiler",
            provider="cpp-cuda-sidecar",
            path=cfg.gpu_lab_binary,
            metadata={
                "stage": "node1-non-llm-gpu-inference-lab-v0.1",
                "privacy": "Runs locally after a motion/keyframe trigger; stores workload routing facts only.",
                "enabled_for_capture": cfg.gpu_lab_enabled,
                "tile_grid": [cfg.gpu_lab_tile_cols, cfg.gpu_lab_tile_rows],
                "prefer_cuda": cfg.gpu_lab_prefer_cuda,
                "allow_python_fallback": cfg.gpu_lab_allow_python_fallback,
                "facts_only": True,
            },
            enabled=cfg.gpu_lab_enabled,
        )
        self.db.upsert_model(
            EVIDENCE_PIPELINE_MODEL_ID,
            role="native_evidence_pipeline",
            provider="cpp-cpu-sidecar",
            path=cfg.evidence_pipeline_binary or cfg.gpu_lab_binary,
            metadata={
                "stage": "capture-run-evidence-pipeline-integration",
                "privacy": "Runs locally after a bounded capture run; stores facts-only evidence metadata.",
                "enabled_for_capture": cfg.evidence_pipeline_enabled,
                "max_batch_bytes": int(cfg.evidence_pipeline_max_batch_bytes),
                "max_batch_clips": int(cfg.evidence_pipeline_max_batch_clips),
                "key_moments": int(cfg.evidence_pipeline_key_moments),
                "min_key_gap_ms": int(cfg.evidence_pipeline_min_key_gap_ms),
                "dedup_hamming_threshold": int(cfg.evidence_pipeline_dedup_hamming_threshold),
                "fingerprint_shape": [int(cfg.evidence_pipeline_fingerprint_width), int(cfg.evidence_pipeline_fingerprint_height)],
                "fingerprint_cycle": int(cfg.evidence_pipeline_fingerprint_cycle),
                "facts_only": True,
                "runs_after_capture_manifest": True,
            },
            enabled=cfg.evidence_pipeline_enabled,
        )
        session_id = self.db.create_session(
            camera_id=cfg.camera_id,
            source_node="node1",
            source_kind="local_v4l2",
            device=cfg.device,
            status="running",
            started_at=started_at,
            ended_at=started_at,
            policy_decision=policy,
        )
        dataset_dir = Path(cfg.data_root) / "captures" / session_id
        frames_dir = dataset_dir / "keyframes"
        overlays_dir = dataset_dir / cfg.overlay_dir_name
        short_clips_dir = dataset_dir / cfg.smolvlm2_clip_dir_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        if cfg.overlay_enabled:
            overlays_dir.mkdir(parents=True, exist_ok=True)
        if cfg.smolvlm2_enabled:
            short_clips_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = dataset_dir / "manifest.json"

        gate = MotionGate(motion_threshold=cfg.motion_threshold, pixel_threshold=cfg.motion_pixel_threshold)
        detector_status: dict[str, Any] = {
            "enabled": bool(cfg.detector_enabled),
            "loaded": False,
            "model_id": cfg.detector_model_id,
            "model_path": cfg.detector_model_path,
            "conf_threshold": cfg.detector_conf_threshold,
            "iou_threshold": cfg.detector_iou_threshold,
            "object_events": 0,
            "error": None,
        }
        detector = self._load_detector(detector_status, session_id)

        frame_id = 0
        frames_seen = 0
        frames_written = 0
        bytes_written = 0
        motion_event_ids: list[str] = []
        object_event_ids: list[str] = []
        artifact_ids: list[str] = []
        artifact_paths: list[str] = []
        overlay_artifact_ids: list[str] = []
        overlay_paths: list[str] = []
        assistant_summary_ids: list[str] = []
        event_contract_ids: list[str] = []
        vlm_analysis_ids: list[str] = []
        smolvlm2_experiment_ids: list[str] = []
        gpu_profile_event_ids: list[str] = []
        evidence_pipeline_event_ids: list[str] = []
        evidence_pipeline_artifact_ids: list[str] = []
        summary_service = AssistantSummaryService(self.db)
        vlm_service = KeyframeVLMAnalysisService(self.db, vlm=self.keyframe_vlm_client) if cfg.vlm_enabled else None
        smolvlm2_service = ShortClipVLMExperimentService(self.db, vlm=self.short_clip_vlm_client) if cfg.smolvlm2_enabled else None
        gpu_lab_runner = (
            Node1NonLLMGpuLabRunner(
                GpuLabConfig(
                    enabled=cfg.gpu_lab_enabled,
                    binary_path=cfg.gpu_lab_binary,
                    tile_cols=cfg.gpu_lab_tile_cols,
                    tile_rows=cfg.gpu_lab_tile_rows,
                    pixel_threshold=cfg.gpu_lab_pixel_threshold,
                    sparse_threshold=cfg.gpu_lab_sparse_threshold,
                    dense_threshold=cfg.gpu_lab_dense_threshold,
                    prefer_cuda=cfg.gpu_lab_prefer_cuda,
                    allow_python_fallback=cfg.gpu_lab_allow_python_fallback,
                )
            )
            if cfg.gpu_lab_enabled
            else None
        )
        recent_frames: deque[tuple[int, Any]] = deque(maxlen=max(1, int(cfg.smolvlm2_clip_frame_count)))
        manifest_frames: list[dict[str, Any]] = []
        previous_frame_for_gpu_lab: Any | None = None
        last_event_time = 0.0
        error: str | None = None
        ok = False

        try:
            self.frame_source.open()
            deadline = time.monotonic() + max(float(cfg.duration_sec), 0.0)
            while time.monotonic() < deadline:
                if cfg.max_frames is not None and frames_seen >= cfg.max_frames:
                    break
                ret, frame = self.frame_source.read()
                if not ret:
                    break
                frame_id += 1
                frames_seen += 1
                if cfg.smolvlm2_enabled:
                    recent_frames.append((frame_id, frame.copy() if hasattr(frame, "copy") else frame))
                motion = gate.evaluate(frame)
                gpu_lab_previous_frame = previous_frame_for_gpu_lab
                previous_frame_for_gpu_lab = frame.copy() if hasattr(frame, "copy") else frame
                now_mono = time.monotonic()
                should_emit = motion.motion and (now_mono - last_event_time >= cfg.min_event_gap_sec)
                if should_emit:
                    last_event_time = now_mono
                    event_ts = now_iso()
                    keyframe_path = frames_dir / f"frame_{frame_id:06d}.jpg"
                    self._write_frame(keyframe_path, frame)
                    size = keyframe_path.stat().st_size
                    digest = sha256_file(keyframe_path)
                    frames_written += 1
                    bytes_written += size
                    artifact_id = self.db.add_artifact(
                        session_id=session_id,
                        camera_id=cfg.camera_id,
                        artifact_type="keyframe",
                        path=str(keyframe_path),
                        media_type="image/jpeg",
                        size_bytes=size,
                        sha256=digest,
                    )
                    motion_id = self.db.insert_event(
                        camera_id=cfg.camera_id,
                        session_id=session_id,
                        frame_id=frame_id,
                        ts=event_ts,
                        event_type="motion_detected",
                        severity="info",
                        label="motion",
                        confidence=round(motion.score, 6),
                        bbox=motion.bbox,
                        source_node="node1",
                        source_kind="local_v4l2",
                        artifact_id=artifact_id,
                        attrs={
                            "motion_score": round(motion.score, 6),
                            "changed_ratio": round(motion.changed_ratio, 8),
                            "device": cfg.device,
                            "width": cfg.width,
                            "height": cfg.height,
                            "fps": cfg.fps,
                            "fourcc": cfg.fourcc,
                            "detector_enabled": bool(cfg.detector_enabled),
                            "note": "Real local frame-difference motion event. Object labels are emitted only by a real enabled detector.",
                        },
                    )
                    motion_event_ids.append(motion_id)
                    artifact_ids.append(artifact_id)
                    artifact_paths.append(str(keyframe_path))

                    detections = self._detect_objects(detector, frame, detector_status, session_id, motion_id)
                    child_ids = self._insert_object_events(
                        detections=detections,
                        cfg=cfg,
                        session_id=session_id,
                        parent_event_id=motion_id,
                        frame_id=frame_id,
                        event_ts=event_ts,
                        artifact_id=artifact_id,
                        motion=motion,
                    )
                    object_event_ids.extend(child_ids)
                    detector_status["object_events"] = int(detector_status.get("object_events", 0)) + len(child_ids)

                    gpu_profile: dict[str, Any] | None = None
                    gpu_profile_event_id: str | None = None
                    if gpu_lab_runner is not None and gpu_lab_previous_frame is not None:
                        gpu_profile = self._profile_gpu_workload(
                            runner=gpu_lab_runner,
                            previous_frame=gpu_lab_previous_frame,
                            current_frame=frame,
                            cfg=cfg,
                            session_id=session_id,
                            parent_event_id=motion_id,
                            frame_id=frame_id,
                            event_ts=event_ts,
                            artifact_id=artifact_id,
                        )
                        gpu_profile_event_id = gpu_profile.get("event_id") if gpu_profile else None
                        if gpu_profile_event_id:
                            gpu_profile_event_ids.append(str(gpu_profile_event_id))

                    overlay_result: dict[str, Any] | None = None
                    overlay_artifact_id: str | None = None
                    if cfg.overlay_enabled and detections and child_ids:
                        overlay_path = overlays_dir / f"frame_{frame_id:06d}_overlay.jpg"
                        overlay_result = self._write_overlay(
                            frame=frame,
                            output_path=overlay_path,
                            detections=detections,
                            child_event_ids=child_ids,
                            parent_event_id=motion_id,
                            session_id=session_id,
                            frame_id=frame_id,
                        )
                        overlay_size = overlay_path.stat().st_size
                        overlay_digest = sha256_file(overlay_path)
                        overlay_artifact_id = self.db.add_artifact(
                            session_id=session_id,
                            camera_id=cfg.camera_id,
                            artifact_type="annotated_keyframe",
                            path=str(overlay_path),
                            media_type="image/jpeg",
                            size_bytes=overlay_size,
                            sha256=overlay_digest,
                        )
                        artifact_ids.append(overlay_artifact_id)
                        artifact_paths.append(str(overlay_path))
                        overlay_artifact_ids.append(overlay_artifact_id)
                        overlay_paths.append(str(overlay_path))
                        bytes_written += overlay_size
                        self.db.audit(
                            "overlay.create",
                            camera_id=cfg.camera_id,
                            event_id=motion_id,
                            session_id=session_id,
                            details={
                                "overlay_artifact_id": overlay_artifact_id,
                                "overlay_path": str(overlay_path),
                                "raw_keyframe_artifact_id": artifact_id,
                                "object_event_ids": child_ids,
                                "model_id": cfg.detector_model_id,
                                "frame_id": frame_id,
                            },
                        )

                    try:
                        summary_results = summary_service.summarize_event_group(motion_id, child_ids)
                        for item in summary_results:
                            if item.get("summary_id"):
                                assistant_summary_ids.append(str(item["summary_id"]))
                            # Pick up contracts created for this event from the DB.
                            latest_contract = self.db.latest_event_contract(str((item.get("event_contract") or {}).get("event_id") or ""))
                            if latest_contract and latest_contract.get("contract_id"):
                                event_contract_ids.append(str(latest_contract["contract_id"]))
                    except Exception as exc:
                        self.db.audit(
                            "assistant.summary.auto_create_failed",
                            outcome="warning",
                            camera_id=cfg.camera_id,
                            event_id=motion_id,
                            session_id=session_id,
                            details={"error": str(exc), "object_event_ids": child_ids},
                        )

                    vlm_result: dict[str, Any] | None = None
                    if vlm_service is not None:
                        vlm_result = vlm_service.analyze_event(motion_id)
                        if vlm_result.get("analysis_id"):
                            vlm_analysis_ids.append(str(vlm_result["analysis_id"]))

                    smolvlm2_result: dict[str, Any] | None = None
                    short_clip_artifact_id: str | None = None
                    short_clip_path: str | None = None
                    if smolvlm2_service is not None:
                        clip = self._write_short_clip_bundle(
                            output_root=short_clips_dir,
                            session_id=session_id,
                            camera_id=cfg.camera_id,
                            trigger_event_id=motion_id,
                            trigger_frame_id=frame_id,
                            event_ts=event_ts,
                            frames=list(recent_frames),
                        )
                        short_clip_artifact_id = clip["clip_artifact_id"]
                        short_clip_path = clip["manifest_path"]
                        artifact_ids.extend(clip["artifact_ids"])
                        artifact_paths.extend(clip["artifact_paths"])
                        bytes_written += int(clip["bytes_written"])
                        smolvlm2_result = smolvlm2_service.analyze_event(motion_id)
                        if smolvlm2_result.get("experiment_id"):
                            smolvlm2_experiment_ids.append(str(smolvlm2_result["experiment_id"]))

                    manifest_frames.append(
                        {
                            "frame_id": frame_id,
                            "ts": event_ts,
                            "motion_event_id": motion_id,
                            "object_event_ids": child_ids,
                            "artifact_id": artifact_id,
                            "path": str(keyframe_path),
                            "sha256": digest,
                            "motion_score": round(motion.score, 6),
                            "bbox": motion.bbox,
                            "detections": [d.as_dict() for d in detections],
                            "overlay_enabled": bool(cfg.overlay_enabled),
                            "overlay_artifact_id": overlay_artifact_id,
                            "overlay_path": (overlay_result or {}).get("overlay_path"),
                            "overlay_boxes": (overlay_result or {}).get("overlay_boxes", []),
                            "vlm_enabled": bool(cfg.vlm_enabled),
                            "vlm_analysis_id": (vlm_result or {}).get("analysis_id") if 'vlm_result' in locals() else None,
                            "vlm_status": (vlm_result or {}).get("status") if 'vlm_result' in locals() else None,
                            "smolvlm2_enabled": bool(cfg.smolvlm2_enabled),
                            "smolvlm2_experiment_id": (smolvlm2_result or {}).get("experiment_id") if 'smolvlm2_result' in locals() else None,
                            "smolvlm2_status": (smolvlm2_result or {}).get("status") if 'smolvlm2_result' in locals() else None,
                            "short_clip_artifact_id": short_clip_artifact_id,
                            "short_clip_path": short_clip_path,
                            "gpu_lab_enabled": bool(cfg.gpu_lab_enabled),
                            "gpu_profile_event_id": gpu_profile_event_id,
                            "gpu_profile": gpu_profile,
                        }
                    )
            ok = True
        except Exception as exc:
            error = str(exc)
            ok = False
        finally:
            self.frame_source.close()

        ended_at = now_iso()
        manifest = {
            "schema": "monitorme.local_capture_manifest.v2",
            "session_id": session_id,
            "camera_id": cfg.camera_id,
            "device": cfg.device,
            "source_node": "node1",
            "source_kind": "local_v4l2",
            "started_at": started_at,
            "ended_at": ended_at,
            "frames_seen": frames_seen,
            "frames_written": frames_written,
            "bytes_written": bytes_written,
            "motion_event_ids": motion_event_ids,
            "object_event_ids": object_event_ids,
            "artifact_ids": artifact_ids,
            "overlay_artifact_ids": overlay_artifact_ids,
            "overlay_paths": overlay_paths,
            "assistant_summary_ids": assistant_summary_ids,
            "event_contract_ids": event_contract_ids,
            "vlm_analysis_ids": vlm_analysis_ids,
            "smolvlm2_experiment_ids": smolvlm2_experiment_ids,
            "gpu_profile_event_ids": gpu_profile_event_ids,
            "evidence_pipeline_event_ids": evidence_pipeline_event_ids,
            "evidence_pipeline_artifact_ids": evidence_pipeline_artifact_ids,
            "evidence_pipeline": {
                "enabled": bool(cfg.evidence_pipeline_enabled),
                "model_id": EVIDENCE_PIPELINE_MODEL_ID,
                "event_count": len(evidence_pipeline_event_ids),
                "artifact_count": len(evidence_pipeline_artifact_ids),
                "max_batch_bytes": int(cfg.evidence_pipeline_max_batch_bytes),
                "max_batch_clips": int(cfg.evidence_pipeline_max_batch_clips),
                "key_moments": int(cfg.evidence_pipeline_key_moments),
                "min_key_gap_ms": int(cfg.evidence_pipeline_min_key_gap_ms),
                "dedup_hamming_threshold": int(cfg.evidence_pipeline_dedup_hamming_threshold),
                "fingerprint_shape": [int(cfg.evidence_pipeline_fingerprint_width), int(cfg.evidence_pipeline_fingerprint_height)],
                "fingerprint_cycle": int(cfg.evidence_pipeline_fingerprint_cycle),
                "facts_only": True,
                "runs_after_capture_manifest": True,
            },
            "gpu_lab": {
                "enabled": bool(cfg.gpu_lab_enabled),
                "model_id": GPU_LAB_MODEL_ID,
                "profile_count": len(gpu_profile_event_ids),
                "tile_grid": [int(cfg.gpu_lab_tile_cols), int(cfg.gpu_lab_tile_rows)],
                "pixel_threshold": int(cfg.gpu_lab_pixel_threshold),
                "sparse_threshold": int(cfg.gpu_lab_sparse_threshold),
                "dense_threshold": int(cfg.gpu_lab_dense_threshold),
                "prefer_cuda": bool(cfg.gpu_lab_prefer_cuda),
                "allow_python_fallback": bool(cfg.gpu_lab_allow_python_fallback),
                "facts_only": True,
            },
            "smolvlm2": {
                "enabled": bool(cfg.smolvlm2_enabled),
                "model_id": cfg.smolvlm2_model_id,
                "experiment_count": len(smolvlm2_experiment_ids),
                "clip_frame_count": int(cfg.smolvlm2_clip_frame_count),
                "runs_after_trigger_only": True,
                "raw_frame_upload": False,
                "experimental": True,
            },
            "vlm": {
                "enabled": bool(cfg.vlm_enabled),
                "model_id": cfg.vlm_model_id,
                "analysis_count": len(vlm_analysis_ids),
                "runs_after_trigger_only": True,
                "raw_frame_upload": False,
            },
            "detector": detector_status,
            "policy_decision": policy,
            "privacy": {"external_upload": False, "face_recognition": False, "raw_frame_upload": False},
            "frames": manifest_frames,
            "error": error,
        }
        evidence_pipeline_result: dict[str, Any] | None = None
        if cfg.evidence_pipeline_enabled:
            evidence_pipeline_result = self._run_capture_evidence_pipeline(
                cfg=cfg,
                session_id=session_id,
                dataset_dir=dataset_dir,
                manifest_frames=manifest_frames,
                started_at=started_at,
                ended_at=ended_at,
            )
            if evidence_pipeline_result.get("event_id"):
                evidence_pipeline_event_ids.append(str(evidence_pipeline_result["event_id"]))
            for aid in evidence_pipeline_result.get("artifact_ids", []):
                evidence_pipeline_artifact_ids.append(str(aid))
                artifact_ids.append(str(aid))
            for path in evidence_pipeline_result.get("artifact_paths", []):
                artifact_paths.append(str(path))
            bytes_written += int(evidence_pipeline_result.get("bytes_written", 0) or 0)
            manifest["evidence_pipeline_event_ids"] = evidence_pipeline_event_ids
            manifest["evidence_pipeline_artifact_ids"] = evidence_pipeline_artifact_ids
            manifest["evidence_pipeline"].update({
                "event_count": len(evidence_pipeline_event_ids),
                "artifact_count": len(evidence_pipeline_artifact_ids),
                "last_result": {
                    "ok": bool(evidence_pipeline_result.get("ok")),
                    "event_id": evidence_pipeline_result.get("event_id"),
                    "manifest_artifact_id": evidence_pipeline_result.get("manifest_artifact_id"),
                    "profile_artifact_id": evidence_pipeline_result.get("profile_artifact_id"),
                    "manifest_csv_path": evidence_pipeline_result.get("manifest_csv_path"),
                    "profile_path": evidence_pipeline_result.get("profile_path"),
                    "error": evidence_pipeline_result.get("error"),
                },
            })

        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_size = manifest_path.stat().st_size
        bytes_written += manifest_size
        manifest_artifact_id = self.db.add_artifact(
            session_id=session_id,
            camera_id=cfg.camera_id,
            artifact_type="capture_manifest",
            path=str(manifest_path),
            media_type="application/json",
            size_bytes=manifest_size,
            sha256=sha256_file(manifest_path),
        )
        artifact_ids.append(manifest_artifact_id)
        artifact_paths.append(str(manifest_path))
        self.db.update_session(
            session_id,
            status="completed" if ok else "failed",
            ended_at=ended_at,
            manifest_path=str(manifest_path),
            dataset_path=str(dataset_dir),
            frames_seen=frames_seen,
            frames_written=frames_written,
            bytes_written=bytes_written,
            error=error,
        )
        if not ok:
            self.db.audit("camera.capture.failed", outcome="failed", camera_id=cfg.camera_id, session_id=session_id, details={"error": error})
        else:
            self.db.audit(
                "camera.capture.completed",
                camera_id=cfg.camera_id,
                session_id=session_id,
                details={"frames_seen": frames_seen, "motion_events": len(motion_event_ids), "object_events": len(object_event_ids), "assistant_summaries": len(assistant_summary_ids), "vlm_analyses": len(vlm_analysis_ids), "smolvlm2_experiments": len(smolvlm2_experiment_ids), "gpu_profiles": len(gpu_profile_event_ids), "evidence_pipeline_events": len(evidence_pipeline_event_ids)},
            )
        return LocalCaptureResult(
            ok=ok,
            camera_id=cfg.camera_id,
            device=cfg.device,
            session_id=session_id,
            dataset_path=str(dataset_dir),
            manifest_path=str(manifest_path),
            frames_seen=frames_seen,
            frames_written=frames_written,
            motion_event_ids=motion_event_ids,
            object_event_ids=object_event_ids,
            artifact_ids=artifact_ids,
            artifact_paths=artifact_paths,
            overlay_artifact_ids=overlay_artifact_ids,
            overlay_paths=overlay_paths,
            assistant_summary_ids=assistant_summary_ids,
            event_contract_ids=event_contract_ids,
            vlm_analysis_ids=vlm_analysis_ids,
            smolvlm2_experiment_ids=smolvlm2_experiment_ids,
            gpu_profile_event_ids=gpu_profile_event_ids,
            evidence_pipeline_event_ids=evidence_pipeline_event_ids,
            evidence_pipeline_artifact_ids=evidence_pipeline_artifact_ids,
            started_at=started_at,
            ended_at=ended_at,
            detector=detector_status,
            error=error,
        )


    def _run_capture_evidence_pipeline(
        self,
        *,
        cfg: LocalCaptureConfig,
        session_id: str,
        dataset_dir: Path,
        manifest_frames: list[dict[str, Any]],
        started_at: str,
        ended_at: str,
    ) -> dict[str, Any]:
        if not manifest_frames:
            self.db.audit(
                "evidence_pipeline.capture_run.skipped",
                outcome="warning",
                camera_id=cfg.camera_id,
                session_id=session_id,
                details={"reason": "no_motion_keyframes", "facts_only": True},
            )
            return {"ok": False, "error": "no motion keyframes available", "artifact_ids": [], "artifact_paths": [], "bytes_written": 0}

        evidence_dir = dataset_dir / "evidence_pipeline"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        csv_path = evidence_dir / "capture_evidence_manifest.csv"
        profile_path = evidence_dir / "evidence_pipeline_profile.json"

        rows = self._capture_frames_to_evidence_rows(manifest_frames=manifest_frames, fps=cfg.fps)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "clip_id",
                    "path",
                    "start_ms",
                    "duration_ms",
                    "bytes",
                    "motion_score",
                    "audio_score",
                    "lighting_delta",
                    "changed_pixels",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        csv_size = csv_path.stat().st_size
        csv_artifact_id = self.db.add_artifact(
            session_id=session_id,
            camera_id=cfg.camera_id,
            artifact_type="evidence_pipeline_manifest_csv",
            path=str(csv_path),
            media_type="text/csv",
            size_bytes=csv_size,
            sha256=sha256_file(csv_path),
        )

        runner = Node1NonLLMGpuLabRunner(
            GpuLabConfig(
                enabled=True,
                binary_path=cfg.evidence_pipeline_binary or cfg.gpu_lab_binary,
                prefer_cuda=False,
                allow_python_fallback=False,
            )
        )
        result = runner.run_evidence_pipeline_manifest(
            manifest_path=csv_path,
            max_batch_bytes=cfg.evidence_pipeline_max_batch_bytes,
            max_batch_clips=cfg.evidence_pipeline_max_batch_clips,
            key_moments=cfg.evidence_pipeline_key_moments,
            min_key_gap_ms=cfg.evidence_pipeline_min_key_gap_ms,
            dedup_hamming_threshold=cfg.evidence_pipeline_dedup_hamming_threshold,
            fingerprint_width=cfg.evidence_pipeline_fingerprint_width,
            fingerprint_height=cfg.evidence_pipeline_fingerprint_height,
            fingerprint_cycle=cfg.evidence_pipeline_fingerprint_cycle,
        )
        profile_payload = {
            "schema": EVIDENCE_PIPELINE_SCHEMA,
            "session_id": session_id,
            "camera_id": cfg.camera_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "capture_manifest_schema": "monitorme.local_capture_manifest.v2",
            "evidence_manifest_csv": str(csv_path),
            "result": result,
            "facts_only": True,
            "note": "Capture-run evidence pipeline integration metadata only; no visual, audio, identity, behavior, or intent claim is emitted.",
        }
        profile_path.write_text(json.dumps(profile_payload, indent=2, sort_keys=True), encoding="utf-8")
        profile_size = profile_path.stat().st_size
        profile_artifact_id = self.db.add_artifact(
            session_id=session_id,
            camera_id=cfg.camera_id,
            artifact_type="evidence_pipeline_profile",
            path=str(profile_path),
            media_type="application/json",
            size_bytes=profile_size,
            sha256=sha256_file(profile_path),
        )

        evidence = result.get("evidence_pipeline") or {}
        safety = evidence.get("safety") or {}
        event_id = self.db.insert_event(
            camera_id=cfg.camera_id,
            session_id=session_id,
            event_type="evidence_pipeline_indexed",
            severity="info" if result.get("ok") and safety.get("ok", False) else "warning",
            label="facts_only_evidence_pipeline",
            confidence=1.0 if result.get("ok") and safety.get("ok", False) else 0.0,
            source_node="node1",
            source_kind="local_v4l2",
            model_id=EVIDENCE_PIPELINE_MODEL_ID,
            artifact_id=profile_artifact_id,
            attrs={
                "schema": EVIDENCE_PIPELINE_SCHEMA,
                "native_schema": evidence.get("schema"),
                "capture_session_id": session_id,
                "capture_manifest_rows": len(rows),
                "manifest_artifact_id": csv_artifact_id,
                "profile_artifact_id": profile_artifact_id,
                "manifest_csv_path": str(csv_path),
                "profile_path": str(profile_path),
                "source": result.get("source"),
                "binary_path": result.get("binary_path"),
                "ok": bool(result.get("ok")),
                "fingerprint_count": evidence.get("fingerprint_count"),
                "duplicate_group_count": evidence.get("duplicate_group_count"),
                "duplicate_clip_count": evidence.get("duplicate_clip_count"),
                "key_moment_count": evidence.get("key_moment_count"),
                "planned_read_bytes": evidence.get("planned_read_bytes"),
                "safety": safety,
                "latency": evidence.get("latency"),
                "facts_only": True,
                "privacy": {"external_upload": False, "identity": False, "intent": False, "media_decode": False},
                "note": "Facts-only capture-run evidence indexing event. It stores storage/fingerprint/dedup/timeline/safety metadata only.",
            },
        )
        self.db.audit(
            "evidence_pipeline.capture_run.create",
            camera_id=cfg.camera_id,
            event_id=event_id,
            session_id=session_id,
            details={
                "manifest_artifact_id": csv_artifact_id,
                "profile_artifact_id": profile_artifact_id,
                "fingerprint_count": evidence.get("fingerprint_count"),
                "duplicate_group_count": evidence.get("duplicate_group_count"),
                "key_moment_count": evidence.get("key_moment_count"),
                "safety_ok": safety.get("ok"),
                "facts_only": True,
            },
        )
        return {
            "ok": bool(result.get("ok")),
            "event_id": event_id,
            "manifest_artifact_id": csv_artifact_id,
            "profile_artifact_id": profile_artifact_id,
            "manifest_csv_path": str(csv_path),
            "profile_path": str(profile_path),
            "artifact_ids": [csv_artifact_id, profile_artifact_id],
            "artifact_paths": [str(csv_path), str(profile_path)],
            "bytes_written": csv_size + profile_size,
            "error": result.get("error"),
        }

    def _capture_frames_to_evidence_rows(self, *, manifest_frames: list[dict[str, Any]], fps: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        first_frame_id = int(manifest_frames[0].get("frame_id") or 0)
        frame_duration_ms = max(1, int(round(1000.0 / max(int(fps), 1))))
        for index, item in enumerate(manifest_frames):
            path = Path(str(item.get("path") or ""))
            frame_id = int(item.get("frame_id") or (first_frame_id + index))
            start_ms = max(0, int(round((frame_id - first_frame_id) * 1000.0 / max(int(fps), 1))))
            motion_score = float(item.get("motion_score") or 0.0)
            bbox = item.get("bbox") or []
            bbox_area = 0.0
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    bbox_area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
                except (TypeError, ValueError):
                    bbox_area = 0.0
            changed_pixels = int(round(float(item.get("motion_score") or 0.0) / 100.0 * float(max(self.config.width * self.config.height, 1))))
            gpu_profile = item.get("gpu_profile") or {}
            frame_profile = gpu_profile.get("frame") if isinstance(gpu_profile, dict) else {}
            if isinstance(frame_profile, dict) and frame_profile.get("changed_pixels") is not None:
                try:
                    changed_pixels = int(frame_profile.get("changed_pixels") or changed_pixels)
                except (TypeError, ValueError):
                    pass
            lighting_delta = round(min(255.0, motion_score + bbox_area * 100.0), 6)
            clip_id = str(item.get("motion_event_id") or item.get("artifact_id") or f"frame_{frame_id}")
            rows.append(
                {
                    "clip_id": clip_id,
                    "path": str(path),
                    "start_ms": start_ms,
                    "duration_ms": frame_duration_ms,
                    "bytes": path.stat().st_size if path.exists() else 0,
                    "motion_score": round(motion_score / 100.0, 6),
                    "audio_score": 0.0,
                    "lighting_delta": lighting_delta,
                    "changed_pixels": changed_pixels,
                }
            )
        return rows

    def _profile_gpu_workload(
        self,
        *,
        runner: Node1NonLLMGpuLabRunner,
        previous_frame: Any,
        current_frame: Any,
        cfg: LocalCaptureConfig,
        session_id: str,
        parent_event_id: str,
        frame_id: int,
        event_ts: str,
        artifact_id: str,
    ) -> dict[str, Any]:
        try:
            result = runner.analyze_frames(previous_frame=previous_frame, current_frame=current_frame)
            frame_result = result.get("frame") or {}
            label = str(frame_result.get("path") or "unavailable")
            confidence = float(frame_result.get("changed_ratio") or 0.0)
            profile_event_id = self.db.insert_event(
                parent_event_id=parent_event_id,
                camera_id=cfg.camera_id,
                session_id=session_id,
                frame_id=frame_id,
                ts=event_ts,
                event_type="gpu_workload_profiled",
                severity="info" if result.get("ok") else "warning",
                label=label,
                confidence=confidence,
                source_node="node1",
                source_kind="local_v4l2",
                model_id=GPU_LAB_MODEL_ID,
                artifact_id=artifact_id,
                attrs={
                    "schema": GPU_LAB_SCHEMA,
                    "parent_motion_event_id": parent_event_id,
                    "keyframe_artifact_id": artifact_id,
                    "native_binary_available": result.get("native_binary_available", result.get("source") == "native_binary"),
                    "source": result.get("source"),
                    "binary_path": result.get("binary_path"),
                    "cuda_compiled": result.get("cuda_compiled"),
                    "frame": frame_result,
                    "frame_cuda": result.get("frame_cuda"),
                    "routing_decision": label,
                    "tile_mask_hex": frame_result.get("tile_mask_hex"),
                    "active_tiles": frame_result.get("active_tiles"),
                    "changed_ratio": frame_result.get("changed_ratio"),
                    "privacy": {"external_upload": False, "identity": False, "intent": False},
                    "note": "Non-LLM CPU/GPU workload profile emitted after local motion trigger. This is routing metadata, not an object or identity claim.",
                },
            )
            result["event_id"] = profile_event_id
            self.db.audit(
                "gpu_lab.profile.create",
                camera_id=cfg.camera_id,
                event_id=profile_event_id,
                session_id=session_id,
                details={
                    "parent_event_id": parent_event_id,
                    "path": label,
                    "active_tiles": frame_result.get("active_tiles"),
                    "tile_mask_hex": frame_result.get("tile_mask_hex"),
                    "source": result.get("source"),
                },
            )
            return result
        except Exception as exc:
            self.db.audit(
                "gpu_lab.profile.failed",
                outcome="warning",
                camera_id=cfg.camera_id,
                event_id=parent_event_id,
                session_id=session_id,
                details={"error": str(exc)},
            )
            return {"ok": False, "error": str(exc)}

    def _load_detector(self, detector_status: dict[str, Any], session_id: str) -> ObjectDetector | None:
        cfg = self.config
        if not cfg.detector_enabled:
            return None
        if self.object_detector is not None:
            detector_status["loaded"] = True
            detector_status["source"] = "injected"
            self.db.audit("detector.loaded", camera_id=cfg.camera_id, session_id=session_id, details={"model_id": cfg.detector_model_id, "source": "injected"})
            return self.object_detector
        try:
            detector = YoloOnnxDetector(
                cfg.detector_model_path,
                model_id=cfg.detector_model_id,
                conf_threshold=cfg.detector_conf_threshold,
                iou_threshold=cfg.detector_iou_threshold,
                max_detections=cfg.detector_max_detections,
                input_size=cfg.detector_input_size,
            )
            detector_status["loaded"] = True
            detector_status["source"] = "onnxruntime"
            self.db.audit("detector.loaded", camera_id=cfg.camera_id, session_id=session_id, details={"model_id": cfg.detector_model_id, "path": cfg.detector_model_path})
            return detector
        except Exception as exc:
            detector_status["loaded"] = False
            detector_status["error"] = str(exc)
            self.db.audit(
                "detector.unavailable",
                outcome="warning",
                camera_id=cfg.camera_id,
                session_id=session_id,
                details={"model_id": cfg.detector_model_id, "path": cfg.detector_model_path, "error": str(exc)},
            )
            return None

    def _detect_objects(
        self,
        detector: ObjectDetector | None,
        frame: Any,
        detector_status: dict[str, Any],
        session_id: str,
        motion_event_id: str,
    ) -> list[ObjectDetection]:
        if detector is None:
            return []
        try:
            detections = detector.detect(frame)
            self.db.audit(
                "detector.run",
                camera_id=self.config.camera_id,
                event_id=motion_event_id,
                session_id=session_id,
                details={"model_id": self.config.detector_model_id, "detections": len(detections)},
            )
            return detections[: self.config.detector_max_detections]
        except Exception as exc:
            detector_status["error"] = str(exc)
            self.db.audit(
                "detector.run.failed",
                outcome="warning",
                camera_id=self.config.camera_id,
                event_id=motion_event_id,
                session_id=session_id,
                details={"model_id": self.config.detector_model_id, "error": str(exc)},
            )
            return []

    def _insert_object_events(
        self,
        *,
        detections: list[ObjectDetection],
        cfg: LocalCaptureConfig,
        session_id: str,
        parent_event_id: str,
        frame_id: int,
        event_ts: str,
        artifact_id: str,
        motion: MotionResult,
    ) -> list[str]:
        object_event_ids: list[str] = []
        for det in detections:
            object_id = self.db.insert_event(
                parent_event_id=parent_event_id,
                camera_id=cfg.camera_id,
                session_id=session_id,
                frame_id=frame_id,
                ts=event_ts,
                event_type="object_detected",
                severity="info",
                label=det.label,
                confidence=det.confidence,
                bbox=det.bbox,
                source_node="node1",
                source_kind="local_v4l2",
                model_id=det.model_id or cfg.detector_model_id,
                artifact_id=artifact_id,
                attrs={
                    **det.as_event_attrs(),
                    "parent_motion_event_id": parent_event_id,
                    "parent_motion_score": round(motion.score, 6),
                    "keyframe_artifact_id": artifact_id,
                    "note": "Real object_detected child row emitted by enabled YOLO ONNX detector after motion gate.",
                },
            )
            object_event_ids.append(object_id)
        return object_event_ids

    def _write_short_clip_bundle(
        self,
        *,
        output_root: Path,
        session_id: str,
        camera_id: str,
        trigger_event_id: str,
        trigger_frame_id: int,
        event_ts: str,
        frames: list[tuple[int, Any]],
    ) -> dict[str, Any]:
        clip_dir = output_root / f"clip_{trigger_frame_id:06d}"
        clip_dir.mkdir(parents=True, exist_ok=True)
        frame_items: list[dict[str, Any]] = []
        artifact_ids: list[str] = []
        artifact_paths: list[str] = []
        bytes_written = 0
        for fid, frame in frames[-max(1, int(self.config.smolvlm2_clip_frame_count)):]:
            frame_path = clip_dir / f"frame_{fid:06d}.jpg"
            self._write_frame(frame_path, frame)
            size = frame_path.stat().st_size
            digest = sha256_file(frame_path)
            frame_artifact_id = self.db.add_artifact(
                session_id=session_id,
                camera_id=camera_id,
                artifact_type="short_clip_frame",
                path=str(frame_path),
                media_type="image/jpeg",
                size_bytes=size,
                sha256=digest,
            )
            bytes_written += size
            artifact_ids.append(frame_artifact_id)
            artifact_paths.append(str(frame_path))
            frame_items.append(
                {
                    "frame_id": fid,
                    "path": str(frame_path),
                    "artifact_id": frame_artifact_id,
                    "sha256": digest,
                    "role": "trigger" if fid == trigger_frame_id else "context",
                }
            )
        manifest = {
            "schema": "monitorme.short_clip_manifest.v0.4",
            "session_id": session_id,
            "camera_id": camera_id,
            "trigger_event_id": trigger_event_id,
            "trigger_frame_id": trigger_frame_id,
            "created_at": now_iso(),
            "event_ts": event_ts,
            "frame_count": len(frame_items),
            "frames": frame_items,
            "privacy": {"external_upload": False, "raw_frame_upload": False, "local_experiment_only": True},
        }
        manifest_path = clip_dir / "clip_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_size = manifest_path.stat().st_size
        manifest_digest = sha256_file(manifest_path)
        clip_artifact_id = self.db.add_artifact(
            session_id=session_id,
            camera_id=camera_id,
            artifact_type="short_clip_manifest",
            path=str(manifest_path),
            media_type="application/json",
            size_bytes=manifest_size,
            sha256=manifest_digest,
        )
        bytes_written += manifest_size
        artifact_ids.append(clip_artifact_id)
        artifact_paths.append(str(manifest_path))
        self.db.audit(
            "smolvlm2.short_clip_bundle.create",
            camera_id=camera_id,
            event_id=trigger_event_id,
            session_id=session_id,
            details={"clip_artifact_id": clip_artifact_id, "manifest_path": str(manifest_path), "frame_count": len(frame_items)},
        )
        return {
            "clip_artifact_id": clip_artifact_id,
            "manifest_path": str(manifest_path),
            "artifact_ids": artifact_ids,
            "artifact_paths": artifact_paths,
            "bytes_written": bytes_written,
        }

    def _write_overlay(
        self,
        *,
        frame: Any,
        output_path: Path,
        detections: list[ObjectDetection],
        child_event_ids: list[str],
        parent_event_id: str,
        session_id: str,
        frame_id: int,
    ) -> dict[str, Any]:
        boxes = [
            OverlayBox(
                bbox=det.bbox,
                label=det.label,
                confidence=det.confidence,
                event_id=event_id,
                parent_event_id=parent_event_id,
                model_id=det.model_id or self.config.detector_model_id,
            )
            for det, event_id in zip(detections, child_event_ids)
        ]
        return render_evidence_overlay(
            frame=frame,
            boxes=boxes,
            output_path=output_path,
            title=f"MonitorMe {self.config.camera_id} frame={frame_id} session={session_id}",
        )

    @staticmethod
    def _write_frame(path: Path, frame: Any) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("OpenCV is required to write camera frames") from exc
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise RuntimeError(f"failed to write keyframe: {path}")


def run_local_capture(db: MonitorMeDB, **kwargs: Any) -> dict[str, Any]:
    config = LocalCaptureConfig(**kwargs)
    return LocalCameraCaptureRunner(db, config).run().as_dict()
