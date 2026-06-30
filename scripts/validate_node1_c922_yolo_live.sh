#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
DB="${MONITORME_DB:-data/events/monitorme.db}"
DEVICE="${MONITORME_CAMERA_DEVICE:-/dev/video0}"
CAMERA_ID="${MONITORME_CAMERA_ID:-c922_node1_gate}"
DURATION="${MONITORME_CAPTURE_DURATION_SEC:-10}"
WIDTH="${MONITORME_CAMERA_WIDTH:-1280}"
HEIGHT="${MONITORME_CAMERA_HEIGHT:-720}"
FPS="${MONITORME_CAMERA_FPS:-30}"
THRESHOLD="${MONITORME_MOTION_THRESHOLD:-1.5}"
MODEL_PATH="${MONITORME_DETECTOR_MODEL_PATH:-models/object_detection/yolo11n.onnx}"
MODEL_ID="${MONITORME_DETECTOR_MODEL_ID:-yolo11n-coco-onnx}"
CONF="${MONITORME_DETECTOR_CONF_THRESHOLD:-0.35}"
IOU="${MONITORME_DETECTOR_IOU_THRESHOLD:-0.45}"
MAX_DET="${MONITORME_DETECTOR_MAX_DETECTIONS:-20}"
INPUT_SIZE="${MONITORME_DETECTOR_INPUT_SIZE:-640}"

echo "=== MonitorMe Node1 C922 + YOLO ONNX live validation ==="
echo "repo=$REPO_ROOT"
echo "db=$DB"
echo "device=$DEVICE camera_id=$CAMERA_ID duration=${DURATION}s profile=${WIDTH}x${HEIGHT}@${FPS}"
echo "model_id=$MODEL_ID model_path=$MODEL_PATH conf=$CONF iou=$IOU"

test -e "$DEVICE" || { echo "ERROR: $DEVICE does not exist" >&2; exit 2; }
test -f "$MODEL_PATH" || { echo "ERROR: real YOLO ONNX model not found: $MODEL_PATH" >&2; echo "Place your model there or set MONITORME_DETECTOR_MODEL_PATH." >&2; exit 3; }
python - <<'PY'
try:
    import onnxruntime  # noqa: F401
except Exception as exc:
    raise SystemExit("ERROR: onnxruntime is not installed. Run: python -m pip install -e '.[api,camera,detector,test]' ") from exc
PY
echo "--- Detector health ---"
python -m monitor_me.cli detector-health \
  --model-id "$MODEL_ID" \
  --model-path "$MODEL_PATH"
python -m monitor_me.cli --db "$DB" init-db >/tmp/monitorme_yolo_live_init.json
python -m monitor_me.cli camera-devices --probe || true
CAPTURE_JSON="$(mktemp /tmp/monitorme_yolo_capture_XXXXXX.json)"
python -m monitor_me.cli --db "$DB" capture-run \
  --camera-id "$CAMERA_ID" \
  --device "$DEVICE" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --fps "$FPS" \
  --fourcc MJPG \
  --duration-sec "$DURATION" \
  --motion-threshold "$THRESHOLD" \
  --detector-enabled \
  --detector-model-id "$MODEL_ID" \
  --detector-model-path "$MODEL_PATH" \
  --detector-conf-threshold "$CONF" \
  --detector-iou-threshold "$IOU" \
  --detector-max-detections "$MAX_DET" \
  --detector-input-size "$INPUT_SIZE" | tee "$CAPTURE_JSON"
SESSION_ID="$(python - <<'PY' "$CAPTURE_JSON"
import json, sys
print(json.load(open(sys.argv[1], encoding='utf-8')).get('session_id', ''))
PY
)"

echo "--- Recent object detections ---"
python -m monitor_me.cli --db "$DB" events --event-type object_detected --limit 20

echo "--- Grounded assistant answer ---"
python -m monitor_me.cli --db "$DB" ask "What person events happened today?"
echo "--- Grounded assistant union answer, person/vehicle if available ---"
python -m monitor_me.cli --db "$DB" ask "What person and vehicle events happened today?"



echo "--- Recent annotated overlays ---"
python -m monitor_me.cli --db "$DB" artifacts --session-id "$SESSION_ID" --artifact-type annotated_keyframe --limit 20 || true

echo "=== MonitorMe Node1 C922 + YOLO ONNX live validation COMPLETE ==="
echo "If zero object_detected rows were emitted, move in front of the camera, lower thresholds, or verify the ONNX model/classes."
