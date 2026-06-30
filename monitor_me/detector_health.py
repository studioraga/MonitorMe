from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_detector_health(
    *,
    model_path: str | Path = "models/object_detection/yolo11n.onnx",
    model_id: str = "yolo11n-coco-onnx",
    expected_sha256: str | None = None,
    load_model: bool = True,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Return a local-only YOLO ONNX detector health report.

    This check never opens the camera and never uploads frames. It validates the
    deployment state needed before live Step 17C/17D inference:
    model path, file size, optional SHA-256, ONNX Runtime availability,
    providers, and optional model-session load metadata.
    """

    path = Path(model_path)
    exists = path.is_file()
    report: dict[str, Any] = {
        "ok": False,
        "model_id": model_id,
        "model_path": str(path),
        "model_path_abs": str(path.resolve()) if path.exists() else str(path.absolute()),
        "exists": exists,
        "size_bytes": None,
        "sha256": None,
        "expected_sha256": expected_sha256 or "",
        "sha256_matches": None,
        "onnxruntime": {
            "available": False,
            "version": None,
            "available_providers": [],
            "selected_providers": [],
            "error": None,
        },
        "load": {
            "requested": bool(load_model),
            "ok": False,
            "error": None,
            "inputs": [],
            "outputs": [],
        },
        "privacy": {
            "external_upload": False,
            "camera_opened": False,
            "raw_frame_upload": False,
            "note": "Detector health validates local model/runtime state only; it does not read CCTV frames.",
        },
        "next_steps": [],
    }

    if exists:
        report["size_bytes"] = path.stat().st_size
        digest = _sha256(path)
        report["sha256"] = digest
        if expected_sha256:
            report["sha256_matches"] = digest.lower() == expected_sha256.lower()
        else:
            report["sha256_matches"] = None
    else:
        report["next_steps"].append("Run scripts/models/download_yolo_onnx.sh or set MONITORME_DETECTOR_MODEL_PATH to a real ONNX file.")

    try:
        import onnxruntime as ort  # type: ignore

        try:
            version = importlib.metadata.version("onnxruntime")
        except Exception:
            version = getattr(ort, "__version__", None)
        available = list(ort.get_available_providers())
        selected = providers or (["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
        selected = [p for p in selected if p in available] or available[:1]
        report["onnxruntime"] = {
            "available": True,
            "version": version,
            "available_providers": available,
            "selected_providers": selected,
            "error": None,
        }
    except Exception as exc:
        report["onnxruntime"]["error"] = str(exc)
        report["next_steps"].append("Install detector support: python -m pip install -e '.[api,camera,detector,test]'.")
        ort = None  # type: ignore

    if load_model:
        if not exists:
            report["load"]["error"] = "model file does not exist"
        elif not report["onnxruntime"]["available"]:
            report["load"]["error"] = "onnxruntime is not available"
        else:
            try:
                session = ort.InferenceSession(str(path), providers=report["onnxruntime"]["selected_providers"])  # type: ignore[name-defined]
                report["load"]["ok"] = True
                report["load"]["inputs"] = [
                    {"name": i.name, "shape": list(i.shape), "type": i.type} for i in session.get_inputs()
                ]
                report["load"]["outputs"] = [
                    {"name": o.name, "shape": list(o.shape), "type": o.type} for o in session.get_outputs()
                ]
            except Exception as exc:
                report["load"]["error"] = str(exc)
                report["next_steps"].append("Verify the ONNX file is a valid YOLO export for ONNX Runtime.")

    file_ok = exists and bool(report["size_bytes"])
    sha_ok = report["sha256_matches"] is not False
    runtime_ok = bool(report["onnxruntime"]["available"])
    load_ok = (not load_model) or bool(report["load"]["ok"])
    report["ok"] = bool(file_ok and sha_ok and runtime_ok and load_ok)
    if report["ok"]:
        report["next_steps"].append("Detector health is OK. Run scripts/validate_node1_c922_yolo_live.sh for live camera validation.")
    return report
