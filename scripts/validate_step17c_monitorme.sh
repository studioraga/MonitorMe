#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
DB="${MONITORME_VALIDATE_DB:-data/events/validate_step17c.db}"
rm -f "$DB" "$DB-shm" "$DB-wal"
mkdir -p data/events data/captures data/evidence_packs data/reports models/object_detection results/models

echo "=== MonitorMe Step 17C validation ==="
echo "repo=$REPO_ROOT"
echo "db=$DB"

python -m monitor_me.cli --db "$DB" init-db >/tmp/monitorme_step17c_init_db.json
python -m monitor_me.cli camera-devices >/tmp/monitorme_step17c_camera_devices.json || true

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

pytest -q

echo "=== MonitorMe Step 17C validation PASSED ==="
echo "This validation proves the normalized object_detected child-row path with a fake injected detector."
echo "It does not fabricate demo CCTV events and does not require a physical camera or ONNX model."
echo "Run scripts/models/download_yolo_onnx.sh, then scripts/validate_node1_c922_yolo_live.sh on Node1 for the real model path."
