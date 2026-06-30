#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
DB="${MONITORME_VALIDATE_DB:-data/events/validate_step17d.db}"
rm -f "$DB" "$DB-shm" "$DB-wal"
mkdir -p data/events data/captures data/evidence_packs data/reports models/object_detection results/models

echo "=== MonitorMe Step 17D validation ==="
echo "repo=$REPO_ROOT"
echo "db=$DB"

python -m monitor_me.cli --db "$DB" init-db >/tmp/monitorme_step17d_init_db.json
python -m monitor_me.cli camera-devices >/tmp/monitorme_step17d_camera_devices.json || true
python -m monitor_me.cli --db "$DB" detector-health --model-path results/models/missing.onnx --skip-load --allow-unhealthy >/tmp/monitorme_step17d_missing_detector_health.json

# Validate the repo-local model download helper without network access by
# pointing it at an already-present non-empty placeholder file.
MODEL_SCRIPT="scripts/models/download_yolo_onnx.sh"
test -x "$MODEL_SCRIPT" || { echo "ERROR: missing executable $MODEL_SCRIPT" >&2; exit 4; }
"$MODEL_SCRIPT" --help >/tmp/monitorme_download_yolo_help.txt
MODEL_TEST_PATH="results/models/test_yolo11n_existing.onnx"
MODEL_TEST_ENV="results/models/test-download.env"
printf 'not-a-real-onnx-for-script-validation\n' > "$MODEL_TEST_PATH"
rm -f "$MODEL_TEST_ENV"
MONITORME_DETECTOR_MODEL_PATH="$MODEL_TEST_PATH" \
MONITORME_ENV_FILE="$MODEL_TEST_ENV" \
  "$MODEL_SCRIPT" >/tmp/monitorme_download_yolo_existing.json
grep -q '^MONITORME_DETECTOR_MODEL_PATH=results/models/test_yolo11n_existing.onnx$' "$MODEL_TEST_ENV"
python -m monitor_me.cli --db "$DB" detector-health --model-path "$MODEL_TEST_PATH" --skip-load --allow-unhealthy >/tmp/monitorme_step17d_placeholder_detector_health.json

pytest -q

echo "=== MonitorMe Step 17D validation PASSED ==="
echo "This validation proves detector health reporting, query-planner union/correlation handling, and normalized object_detected child rows with a fake injected detector."
echo "It does not fabricate demo CCTV events and does not require a physical camera or ONNX model."
echo "Run scripts/models/download_yolo_onnx.sh, then scripts/validate_node1_c922_yolo_live.sh on Node1 for the real model path."
