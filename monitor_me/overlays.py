from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .time_utils import now_iso


@dataclass(frozen=True)
class OverlayBox:
    """One evidence annotation rendered on an overlay image."""

    bbox: list[float]
    label: str
    confidence: float | None
    event_id: str
    parent_event_id: str | None
    model_id: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox,
            "label": self.label,
            "confidence": self.confidence,
            "event_id": self.event_id,
            "parent_event_id": self.parent_event_id,
            "model_id": self.model_id,
        }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bbox_to_pixels(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        return (0, 0, max(width - 1, 0), max(height - 1, 0))
    x1 = int(round(_clip01(bbox[0]) * width))
    y1 = int(round(_clip01(bbox[1]) * height))
    x2 = int(round(_clip01(bbox[2]) * width))
    y2 = int(round(_clip01(bbox[3]) * height))
    x1, x2 = sorted((max(0, min(x1, width - 1)), max(0, min(x2, width - 1))))
    y1, y2 = sorted((max(0, min(y1, height - 1)), max(0, min(y2, height - 1))))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


def render_evidence_overlay(
    *,
    frame: Any,
    boxes: Iterable[OverlayBox],
    output_path: str | Path,
    title: str = "MonitorMe evidence overlay",
) -> dict[str, Any]:
    """Write an annotated keyframe while leaving the raw keyframe untouched.

    The overlay is an investigator/operator convenience artifact. The original
    JPEG keyframe remains the primary evidence artifact and is never modified.
    Each object box is annotated with the normalized detection facts that the
    assistant is allowed to cite: label, confidence, event_id, parent_event_id,
    and model_id.
    """

    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - optional deployment dependency
        raise RuntimeError("OpenCV is required to render MonitorMe overlays. Install with: pip install -e '.[camera]'") from exc

    if frame is None:
        raise ValueError("frame is required to render evidence overlay")

    image = frame.copy()
    height, width = image.shape[:2]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Draw a small title banner. Colors are fixed OpenCV BGR values and do not
    # change evidence facts; they are only visual presentation.
    cv2.rectangle(image, (0, 0), (width, 34), (0, 0, 0), thickness=-1)
    cv2.putText(image, title[:120], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    rendered: list[dict[str, Any]] = []
    for idx, box in enumerate(boxes, start=1):
        x1, y1, x2, y2 = _bbox_to_pixels(box.bbox, width, height)
        color = (0, 255, 255) if box.label == "person" else (0, 200, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=2)
        conf_text = "" if box.confidence is None else f" {box.confidence:.2f}"
        lines = [
            f"{idx}. {box.label}{conf_text}",
            f"event_id={box.event_id}",
            f"parent_event_id={box.parent_event_id or ''}",
            f"model_id={box.model_id or ''}",
        ]
        y_text = max(52, y1 - 58)
        text_width = max(cv2.getTextSize(line[:96], cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0] for line in lines)
        box_h = 15 * len(lines) + 8
        cv2.rectangle(image, (x1, y_text - 14), (min(width - 1, x1 + text_width + 10), min(height - 1, y_text + box_h)), (0, 0, 0), thickness=-1)
        for offset, line in enumerate(lines):
            cv2.putText(image, line[:96], (x1 + 5, y_text + offset * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        rendered.append(box.as_dict())

    ok = cv2.imwrite(str(output), image)
    if not ok:
        raise RuntimeError(f"failed to write evidence overlay: {output}")
    return {
        "overlay_path": str(output),
        "overlay_box_count": len(rendered),
        "overlay_boxes": rendered,
        "created_at": now_iso(),
        "privacy": {
            "raw_frame_modified": False,
            "external_upload": False,
            "note": "Annotated overlay is a derived local artifact; the original raw keyframe remains unchanged.",
        },
    }
