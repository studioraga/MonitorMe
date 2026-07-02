#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
DB="data/events/validate_node1_assistant_v03.db"
mkdir -p data/events
rm -f "$DB" "$DB-shm" "$DB-wal"

echo "=== MonitorMe Node1 AI Camera Assistant v0.3 validation ==="
echo "repo=$REPO_ROOT"
echo "db=$DB"
python -m pytest -q tests/test_node1_ai_camera_assistant_v03.py
python -m monitor_me.cli --db "$DB" vlm-health --allow-unconfigured
cat <<'MSG'
=== MonitorMe Node1 AI Camera Assistant v0.3 validation PASSED ===
This validates optional local Qwen VLM keyframe analysis after trigger, strict
JSON validation, local-only endpoint guardrails, failed-analysis storage, and
no-analysis behavior when Qwen VLM is disabled.
It uses fake injected frames/detectors/VLM and does not require a physical
camera, Qwen server, MAX/Gemma, or external services.
MSG
