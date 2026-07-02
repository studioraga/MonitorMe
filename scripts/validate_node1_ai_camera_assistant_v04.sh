#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DB="data/events/validate_node1_assistant_v04.db"
mkdir -p "$(dirname "$DB")"
rm -f "$DB" "$DB-shm" "$DB-wal"

export MONITORME_DB="$DB"

echo "=== MonitorMe Node1 AI Camera Assistant v0.4/v0.4.1 validation ==="
echo "repo=$REPO_ROOT"
echo "db=$DB"
python -m pytest tests/test_node1_ai_camera_assistant_v04.py -q

echo "=== MonitorMe Node1 AI Camera Assistant v0.4/v0.4.1 validation PASSED ==="
echo "This validates optional SmolVLM2 short clip experiments after local triggers, including v0.4.1 constrained structured_outputs JSON schema."
echo "It does not require SmolVLM2 to be running and does not upload CCTV frames externally."
