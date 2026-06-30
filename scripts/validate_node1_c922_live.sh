#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
DB="${MONITORME_DB:-data/events/monitorme.db}"
DEVICE="${MONITORME_CAMERA_DEVICE:-/dev/video0}"
CAMERA_ID="${MONITORME_CAMERA_ID:-c922_node1_gate}"
DURATION="${MONITORME_CAPTURE_DURATION_SEC:-10}"
WIDTH="${MONITORME_CAMERA_WIDTH:-1280}"
HEIGHT="${MONITORME_CAMERA_HEIGHT:-720}"
FPS="${MONITORME_CAMERA_FPS:-30}"
THRESHOLD="${MONITORME_MOTION_THRESHOLD:-1.5}"

echo "=== MonitorMe Node1 C922 live validation ==="
echo "repo=$REPO_ROOT"
echo "db=$DB"
echo "device=$DEVICE camera_id=$CAMERA_ID duration=${DURATION}s profile=${WIDTH}x${HEIGHT}@${FPS}"

test -e "$DEVICE" || { echo "ERROR: $DEVICE does not exist" >&2; exit 2; }
python -m monitor_me.cli --db "$DB" init-db >/tmp/monitorme_live_init.json
python -m monitor_me.cli camera-devices --probe || true
python -m monitor_me.cli --db "$DB" capture-run \
  --camera-id "$CAMERA_ID" \
  --device "$DEVICE" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --fps "$FPS" \
  --fourcc MJPG \
  --duration-sec "$DURATION" \
  --motion-threshold "$THRESHOLD"

echo "--- Recent motion events ---"
python -m monitor_me.cli --db "$DB" events --event-type motion_detected --limit 10

echo "--- Grounded assistant answer ---"
python -m monitor_me.cli --db "$DB" ask "What motion events happened today?"

echo "=== MonitorMe Node1 C922 live validation COMPLETE ==="
echo "If zero events were emitted, move in front of the camera or lower MONITORME_MOTION_THRESHOLD, e.g. 0.5."
