#!/usr/bin/env bash
set -euo pipefail
PID_FILE="${MONITORME_PID_FILE:-run/monitorme_api.pid}"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "MonitorMe API running pid=$(cat "$PID_FILE")"
else
  echo "MonitorMe API not running"
fi
ss -ltnp | grep ':8088' || true
