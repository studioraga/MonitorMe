#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DB="${MONITORME_VALIDATE_DB:-data/events/validate_node1_assistant_v01.db}"
rm -f "$DB" "$DB-shm" "$DB-wal"
mkdir -p "$(dirname "$DB")" results

echo "=== MonitorMe Node1 AI Camera Assistant v0.1 validation ==="
echo "repo=$REPO_ROOT"
echo "db=$DB"

python -m pytest -q \
  tests/test_node1_ai_camera_assistant_v01.py \
  tests/test_assistant_grounding.py \
  tests/test_evidence_pack_and_report.py

echo "=== MonitorMe Node1 AI Camera Assistant v0.1 validation PASSED ==="
echo "This validation proves event contracts, deterministic policy, automatic summaries, DB-grounded assistant answers, incident-report summary inclusion, and non-invention behavior."
echo "It uses fake injected frames/detectors and does not require a physical camera, ONNX model, Gemma/MAX, or external services."
