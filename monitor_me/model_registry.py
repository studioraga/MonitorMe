from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .db import MonitorMeDB


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    role: str
    provider: str = "local"
    version: str = ""
    path: str = ""
    sha256: str = ""
    metadata: dict[str, Any] | None = None
    enabled: bool = True


DEFAULT_MODELS = [
    ModelRecord(
        model_id="yolo11n-coco-onnx",
        role="object_detector",
        provider="onnxruntime",
        path="models/object_detection/yolo11n.onnx",
        metadata={"notes": "Detector metadata placeholder. Model file is configured by deployment."},
    ),
    ModelRecord(
        model_id="google/gemma-3-1b-it",
        role="text_model",
        provider="max-or-local-http",
        metadata={"privacy": "Receives local structured event facts only, never raw private frames."},
    ),
    ModelRecord(
        model_id="deterministic-null-llm",
        role="text_model_fallback",
        provider="deterministic",
        metadata={"privacy": "No model call; deterministic local answer composer."},
    ),
    ModelRecord(
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        role="vlm_keyframe_analyzer",
        provider="qwen-openai-compatible",
        enabled=False,
        metadata={
            "stage": "node1-assistant-v0.3",
            "privacy": "Disabled by default; analyzes stored trigger keyframes only through a local OpenAI-compatible VLM endpoint.",
        },
    ),
    ModelRecord(
        model_id="sam2-small",
        role="segmentation_model_future",
        provider="disabled",
        enabled=False,
        metadata={"stage": "future-step-20"},
    ),
    ModelRecord(
        model_id="grounding-dino-edge",
        role="open_vocab_detector_future",
        provider="disabled",
        enabled=False,
        metadata={"stage": "future-step-21"},
    ),
    ModelRecord(
        model_id="clip-siglip-local",
        role="embedding_model_future",
        provider="disabled",
        enabled=False,
        metadata={"stage": "future-step-22"},
    ),
]


def register_default_models(db: MonitorMeDB) -> None:
    text_model = os.getenv("MONITORME_TEXT_MODEL_ID", "google/gemma-3-1b-it")
    for record in DEFAULT_MODELS:
        model_id = text_model if record.role == "text_model" else record.model_id
        db.upsert_model(
            model_id,
            role=record.role,
            provider=record.provider,
            version=record.version,
            path=record.path,
            sha256=record.sha256,
            metadata=record.metadata or {},
            enabled=record.enabled,
        )
