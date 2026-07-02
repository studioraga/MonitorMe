#!/usr/bin/env bash
set -Eeuo pipefail

# Create a stable-channel Python 3.12 MAX/Gemma workspace and explicitly add
# the standalone mojo package. Use this when both nightly and stable modular-only
# solves fail with serving-time Mojo errors such as:
#   unable to locate module 'std'
#   unable to locate module 'nn'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MONITORME_MAX_NEW_PROJECT_DIR="${MONITORME_MAX_NEW_PROJECT_DIR:-$HOME/dev/modular/project/quickstart_py312_stable_mojo}"
export MONITORME_MAX_CHANNEL="${MONITORME_MAX_CHANNEL:-https://conda.modular.com/max}"
export MONITORME_CONDA_FORGE_CHANNEL="${MONITORME_CONDA_FORGE_CHANNEL:-conda-forge}"
export MONITORME_MAX_ADD_EXPLICIT_MOJO="${MONITORME_MAX_ADD_EXPLICIT_MOJO:-1}"

cat <<MSG
=== MonitorMe stable MAX/Gemma + explicit Mojo recovery workspace ===
Project: $MONITORME_MAX_NEW_PROJECT_DIR
Channel: $MONITORME_MAX_CHANNEL
Python:  python=3.12
Extra:   explicit mojo package

This creates a new workspace separate from failed quickstart_py312 and
quickstart_py312_stable attempts.
MSG

exec "$SCRIPT_DIR/create_max_gemma3_1b_py312_env.sh"
