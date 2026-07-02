from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .yolo_onnx import ObjectDetection, ObjectDetector, YoloOnnxDetector


class YoloClient(Protocol):
    """Small interface for fast visual facts on Node1.

    YOLO is responsible only for visible object facts: label, confidence, bbox,
    frame/model metadata. Policy decisions and natural-language explanations
    live in separate layers.
    """

    model_id: str

    def detect(self, frame: Any) -> list[ObjectDetection]:
        ...

    def health(self) -> dict[str, Any]:
        ...


@dataclass
class LocalYoloOnnxClient:
    """Thin client wrapper around the ONNX YOLO detector."""

    model_path: str
    model_id: str = "yolo11n-coco-onnx"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    max_detections: int = 20
    input_size: int = 640

    def __post_init__(self) -> None:
        self._detector = YoloOnnxDetector(
            self.model_path,
            model_id=self.model_id,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections,
            input_size=self.input_size,
        )

    def detect(self, frame: Any) -> list[ObjectDetection]:
        return self._detector.detect(frame)

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "client": "local_yolo_onnx",
            "model_id": self.model_id,
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
            "max_detections": self.max_detections,
            "input_size": self.input_size,
            "role": "fast_visual_facts_only",
        }
