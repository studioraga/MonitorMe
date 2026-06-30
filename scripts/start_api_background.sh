#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p run logs data/events
PID_FILE="${MONITORME_PID_FILE:-run/monitorme_api.pid}"
LOG_FILE="${MONITORME_LOG_FILE:-logs/monitorme_api.log}"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "MonitorMe API already running pid=$(cat "$PID_FILE")"
  exit 0
fi
export PYTHONPATH="$REPO_ROOT"
export MONITORME_CREATE_APP_AT_IMPORT=1
export MONITORME_DB="${MONITORME_DB:-data/events/monitorme.db}"
nohup python -m uvicorn monitor_me.routes:app --host "${MONITORME_HOST:-127.0.0.1}" --port "${MONITORME_PORT:-8088}" >"$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
echo "MonitorMe API started pid=$(cat "$PID_FILE") log=$LOG_FILE"
