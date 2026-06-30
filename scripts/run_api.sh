#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
export MONITORME_CREATE_APP_AT_IMPORT=1
export MONITORME_DB="${MONITORME_DB:-data/events/monitorme.db}"
python -m uvicorn monitor_me.routes:app --host "${MONITORME_HOST:-127.0.0.1}" --port "${MONITORME_PORT:-8088}"
