#!/usr/bin/env bash
set -Eeuo pipefail

# Validate MonitorMe's v0.2 Gemma/MAX integration against a running local MAX server.
# Run this in TERM2 after scripts/max/term1_start_max_gemma3_1b.sh is serving.
# This uses a synthetic in-process validation frame/detector to create local DB evidence;
# it does not open /dev/video0 and does not upload frames externally.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DB="${MONITORME_DB:-data/events/validate_node1_ai_camera_assistant_v02_max_live.db}"
BASE_URL="${MONITORME_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_ID="${MONITORME_LLM_MODEL_ID:-google/gemma-3-1b-it}"
RESULTS_DIR="${MONITORME_RESULTS_DIR:-results/max_gemma_monitorme_v02_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$DB")" "$RESULTS_DIR"

echo "=== MonitorMe v0.2 live Gemma/MAX validation ==="
echo "repo=$REPO_ROOT"
echo "db=$DB"
echo "base_url=$BASE_URL"
echo "model_id=$MODEL_ID"
echo "results=$RESULTS_DIR"
echo

echo "=== Waiting for MAX API server ==="
for i in $(seq 1 120); do
  if curl -fsS "$BASE_URL/models" >/dev/null 2>&1; then
    echo "PASS: API server reachable at $BASE_URL"
    break
  fi
  if [ "$i" -eq 120 ]; then
    echo "ERROR: API server not reachable. Start TERM1 first."
    exit 1
  fi
  sleep 1
done

export MONITORME_DB="$DB"
export MONITORME_LLM_PROVIDER=max-openai
export MONITORME_ASSISTANT_USE_GEMMA=1
export MONITORME_LLM_BASE_URL="$BASE_URL"
export MONITORME_LLM_MODEL_ID="$MODEL_ID"
export MONITORME_LLM_API_KEY="${MONITORME_LLM_API_KEY:-EMPTY}"
export MONITORME_LLM_TEMPERATURE="${MONITORME_LLM_TEMPERATURE:-0.0}"
export MONITORME_LLM_TIMEOUT_SEC="${MONITORME_LLM_TIMEOUT_SEC:-120}"
export MONITORME_LLM_MAX_TOKENS="${MONITORME_LLM_MAX_TOKENS:-192}"

echo
echo "=== MonitorMe llm-health with API probe ==="
python -m monitor_me.cli --db "$DB" llm-health --probe | tee "$RESULTS_DIR/llm_health.json"

echo
echo "=== Creating local validation event and asking Gemma for strict JSON summary ==="
python - <<'PY' | tee "$RESULTS_DIR/monitorme_gemma_summary.json"
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from monitor_me.assistant_summary import AssistantSummaryService
from monitor_me.db import MonitorMeDB
from monitor_me.local_capture import IterableFrameSource, LocalCameraCaptureRunner, LocalCaptureConfig
from monitor_me.yolo_client import Detection

class PersonGuitarDetector:
    model_id = "yolo11n-coco-onnx"
    source = "test-validation-detector"
    def detect(self, frame):
        return [
            Detection(label="person", confidence=0.91, bbox=[0.20, 0.10, 0.70, 0.95], class_id=0, raw_label="person"),
            Detection(label="guitar", confidence=0.88, bbox=[0.30, 0.45, 0.75, 0.82], class_id=999, raw_label="guitar"),
        ]

black = np.zeros((240, 320, 3), dtype=np.uint8)
white = np.full((240, 320, 3), 255, dtype=np.uint8)
frames = [black, black, white, white]

db_path = Path(__import__("os").environ.get("MONITORME_DB", "data/events/validate_node1_ai_camera_assistant_v02_max_live.db"))
db = MonitorMeDB(db_path)
config = LocalCaptureConfig(
    camera_id="c922_node1_gate",
    device="/dev/video0",
    width=320,
    height=240,
    fps=30,
    duration_sec=1,
    motion_threshold=1.0,
    data_root=str(db_path.parent.parent if db_path.parent.name == "events" else Path("data")),
    detector_enabled=True,
    overlay_enabled=True,
)
result = LocalCameraCaptureRunner(db, config, frame_source=IterableFrameSource(frames), detector=PersonGuitarDetector()).run()
summary = AssistantSummaryService(db).summarize_event(result.motion_event_ids[0])
print(json.dumps({"capture": result.as_dict(), "summary": summary}, indent=2, sort_keys=True))
if summary.get("summary_source") != "gemma_max":
    raise SystemExit(f"Gemma/MAX summary was not accepted; fallback_reason={summary.get('fallback_reason')}")
PY

echo
echo "=== Recent summaries ==="
python -m monitor_me.cli --db "$DB" summaries --limit 5 | tee "$RESULTS_DIR/recent_summaries.json"

echo
echo "=== Validation summary ==="
cat > "$RESULTS_DIR/VALIDATION_SUMMARY.txt" <<EOF
MonitorMe Node1 AI Camera Assistant v0.2 live Gemma/MAX validation

Status:
- MAX API reachable: PASS
- MonitorMe llm-health probe: PASS
- Synthetic local evidence event created: PASS
- Gemma/MAX strict JSON summary accepted: PASS
- Raw frames uploaded externally: NO
- Camera opened: NO

Model:
- $MODEL_ID

Base URL:
- $BASE_URL
EOF
cat "$RESULTS_DIR/VALIDATION_SUMMARY.txt"
echo
echo "Artifacts saved under: $RESULTS_DIR"
