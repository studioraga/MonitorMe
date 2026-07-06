#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export MONITORME_DB="${MONITORME_DB:-data/events/monitorme.db}"

python -m monitor_me.cli --db "${MONITORME_DB}" evidence-retention-schedule-run "$@"
