from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import math


COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

VEHICLE_RAW_LABELS = {"bicycle", "car", "motorcycle", "bus", "train", "truck", "boat"}


def canonical_label(raw_label: str) -> str:
    label = raw_label.strip().lower()
    if label in VEHICLE_RAW_LABELS:
        return "vehicle"
    return label


@dataclass(frozen=True)
class ObjectDetection:
    """A normalized object detection ready to become a MonitorMe event row.

    bbox is normalized [x1, y1, x2, y2] in the original frame coordinate system.
    label is canonicalized for CCTV queryability, so COCO car/truck/bus/etc.
    become label="vehicle" while raw_label stores the exact model class.
    """

    label: str
    confidence: float
    bbox: list[float]
    model_id: str
    class_id: int | None = None
    raw_label: str | None = None
    attrs: dict[str, Any] | None = None

    def as_event_attrs(self) -> dict[str, Any]:
        return {
            "detector": "yolo_onnx",
            "class_id": self.class_id,
            "raw_label": self.raw_label or self.label,
            **(self.attrs or {}),
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObjectDetector(Protocol):
    model_id: str

    def detect(self, frame: Any) -> list[ObjectDetection]: ...


@dataclass(frozen=True)
class LetterboxInfo:
    input_w: int
    input_h: int
    orig_w: int
    orig_h: int
    gain: float
    pad_x: float
    pad_y: float


class YoloOnnxDetector:
    """Small ONNX Runtime wrapper for Ultralytics-style YOLOv8/YOLO11 ONNX models.

    It supports common output layouts:
    - [1, 84, N] or [1, N, 84] for YOLOv8/YOLO11 COCO: xywh + class scores
    - [1, N, 85] for YOLOv5-style: xywh + objectness + class scores
    - [1, N, 6] for already-decoded: x1,y1,x2,y2,confidence,class_id

    This class is deliberately optional. If onnxruntime or the ONNX file is not
    available, MonitorMe can still store parent motion events and audit that the
    object detector was unavailable, without fabricating child object rows.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str = "yolo11n-coco-onnx",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 20,
        input_size: int = 640,
        providers: list[str] | None = None,
    ):
        self.model_id = model_id
        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.input_size = int(input_size)
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on deployment
            raise RuntimeError("onnxruntime is required for YOLO ONNX detection. Install with: pip install -e '.[detector]'") from exc
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"YOLO ONNX model not found: {self.model_path}")
        chosen = providers or self._default_providers(ort)
        self.session = ort.InferenceSession(self.model_path, providers=chosen)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.model_input_w, self.model_input_h = self._resolve_input_size(self.input_shape, fallback=self.input_size)

    @staticmethod
    def _default_providers(ort: Any) -> list[str]:
        available = set(ort.get_available_providers())
        # Prefer CUDA when onnxruntime-gpu is installed, otherwise CPU.
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _resolve_input_size(shape: list[Any], *, fallback: int) -> tuple[int, int]:
        # Common NCHW: [1, 3, 640, 640]. Dynamic dimensions may be strings/None.
        try:
            h = int(shape[2]) if len(shape) >= 4 and isinstance(shape[2], int) else fallback
            w = int(shape[3]) if len(shape) >= 4 and isinstance(shape[3], int) else fallback
            return max(w, 1), max(h, 1)
        except Exception:
            return fallback, fallback

    def detect(self, frame: Any) -> list[ObjectDetection]:
        import numpy as np  # type: ignore

        tensor, info = self._preprocess(frame)
        outputs = self.session.run(None, {self.input_name: tensor})
        if not outputs:
            return []
        rows = self._as_rows(outputs[0])
        candidates = self._decode_rows(rows, info)
        keep = self._nms(candidates)
        return keep[: self.max_detections]

    def _preprocess(self, frame: Any) -> tuple[Any, LetterboxInfo]:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        if frame is None:
            raise ValueError("frame is None")
        orig_h, orig_w = frame.shape[:2]
        input_w, input_h = self.model_input_w, self.model_input_h
        gain = min(input_w / max(orig_w, 1), input_h / max(orig_h, 1))
        new_w, new_h = int(round(orig_w * gain)), int(round(orig_h * gain))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
        pad_x = (input_w - new_w) / 2.0
        pad_y = (input_h - new_h) / 2.0
        x0, y0 = int(round(pad_x)), int(round(pad_y))
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, :, :, :]
        return tensor, LetterboxInfo(input_w, input_h, orig_w, orig_h, gain, pad_x, pad_y)

    @staticmethod
    def _as_rows(output: Any) -> Any:
        import numpy as np  # type: ignore

        arr = np.asarray(output)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            arr = arr.reshape(-1, arr.shape[-1])
        # Ultralytics commonly exports [84, 8400]; transpose to [8400, 84].
        if arr.shape[0] < arr.shape[1] and arr.shape[0] in (6, 7, 84, 85, 116):
            arr = arr.T
        return arr

    def _decode_rows(self, rows: Any, info: LetterboxInfo) -> list[ObjectDetection]:
        import numpy as np  # type: ignore

        candidates: list[ObjectDetection] = []
        for row in rows:
            row = np.asarray(row, dtype=np.float32)
            if row.size < 6:
                continue
            if row.size == 6:
                x1, y1, x2, y2, conf, cls_id = [float(v) for v in row[:6]]
                class_id = int(cls_id)
                score = float(conf)
                box = self._xyxy_to_normalized(x1, y1, x2, y2, info)
            else:
                x, y, w, h = [float(v) for v in row[:4]]
                # YOLOv5: xywh + objectness + class scores. YOLOv8/11: xywh + class scores.
                if row.size >= 85:
                    obj = float(row[4])
                    class_scores = row[5:]
                    class_id = int(np.argmax(class_scores))
                    score = obj * float(class_scores[class_id])
                else:
                    class_scores = row[4:]
                    class_id = int(np.argmax(class_scores))
                    score = float(class_scores[class_id])
                x1, y1, x2, y2 = x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0
                box = self._xyxy_to_normalized(x1, y1, x2, y2, info)
            if not math.isfinite(score) or score < self.conf_threshold:
                continue
            raw_label = COCO80[class_id] if 0 <= class_id < len(COCO80) else f"class_{class_id}"
            label = canonical_label(raw_label)
            if not box:
                continue
            candidates.append(
                ObjectDetection(
                    label=label,
                    confidence=round(float(score), 6),
                    bbox=box,
                    model_id=self.model_id,
                    class_id=class_id,
                    raw_label=raw_label,
                    attrs={"canonical_label": label},
                )
            )
        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return candidates

    @staticmethod
    def _xyxy_to_normalized(x1: float, y1: float, x2: float, y2: float, info: LetterboxInfo) -> list[float] | None:
        # Undo letterbox padding/scaling.
        x1 = (x1 - info.pad_x) / max(info.gain, 1e-9)
        x2 = (x2 - info.pad_x) / max(info.gain, 1e-9)
        y1 = (y1 - info.pad_y) / max(info.gain, 1e-9)
        y2 = (y2 - info.pad_y) / max(info.gain, 1e-9)
        x1 = max(0.0, min(float(info.orig_w), x1))
        x2 = max(0.0, min(float(info.orig_w), x2))
        y1 = max(0.0, min(float(info.orig_h), y1))
        y2 = max(0.0, min(float(info.orig_h), y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return [
            round(x1 / max(info.orig_w, 1), 6),
            round(y1 / max(info.orig_h, 1), 6),
            round(x2 / max(info.orig_w, 1), 6),
            round(y2 / max(info.orig_h, 1), 6),
        ]

    def _nms(self, candidates: list[ObjectDetection]) -> list[ObjectDetection]:
        kept: list[ObjectDetection] = []
        for det in candidates:
            if len(kept) >= self.max_detections:
                break
            if all(self._iou(det.bbox, prev.bbox) <= self.iou_threshold or det.label != prev.label for prev in kept):
                kept.append(det)
        return kept

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0.0
