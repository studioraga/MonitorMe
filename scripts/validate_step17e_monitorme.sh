#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DB="${MONITORME_VALIDATE_DB:-data/events/validate_step17e.db}"
rm -f "$DB" "$DB-shm" "$DB-wal"
mkdir -p "$(dirname "$DB")" results/validation

LOG="results/validation/validate_step17e_$(date +%Y%m%d_%H%M%S).log"
{
  echo "=== MonitorMe Step 17E validation ==="
  echo "repo=$REPO"
  echo "db=$DB"
  MONITORME_VALIDATE_DB="$DB" python -m pytest \
    tests/test_evidence_overlays.py \
    tests/test_yolo_detection_pipeline.py \
    tests/test_evidence_pack_and_report.py \
    tests/test_api_routes.py
  echo "=== MonitorMe Step 17E validation PASSED ==="
  echo "This validation proves annotated overlay artifacts are generated from detected object rows while raw keyframes remain unchanged."
  echo "It uses fake injected frames/detector and does not require a physical camera or ONNX model."
} | tee "$LOG"
