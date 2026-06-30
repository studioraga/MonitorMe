#!/usr/bin/env bash
set -euo pipefail
PID_FILE="${MONITORME_PID_FILE:-run/monitorme_api.pid}"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  kill "$(cat "$PID_FILE")"
  rm -f "$PID_FILE"
  echo "MonitorMe API stopped"
else
  echo "MonitorMe API was not running"
fi
