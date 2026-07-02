#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DB="${MONITORME_DB:-data/events/validate_node1_assistant_v02.db}"
rm -f "$DB" "$DB-shm" "$DB-wal"
mkdir -p "$(dirname "$DB")"

printf '=== MonitorMe Node1 AI Camera Assistant v0.2 validation ===\n'
printf 'repo=%s\n' "$REPO_ROOT"
printf 'db=%s\n' "$DB"

python -m monitor_me.cli --db "$DB" init-db >/dev/null
python -m monitor_me.cli --db "$DB" llm-health --allow-unconfigured >/dev/null
python -m pytest tests/test_node1_ai_camera_assistant_v01.py tests/test_node1_ai_camera_assistant_v02.py tests/test_api_routes.py -q

printf '=== MonitorMe Node1 AI Camera Assistant v0.2 validation PASSED ===\n'
printf 'This validates strict Gemma/MAX JSON summary parsing, validation, and deterministic fallback.\n'
printf 'It does not require MAX/Gemma to be running and does not upload CCTV frames externally.\n'
