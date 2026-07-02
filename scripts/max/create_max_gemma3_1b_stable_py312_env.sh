#!/usr/bin/env bash
set -Eeuo pipefail

# Create a clean, isolated MAX + Gemma 3 1B pixi workspace using the stable
# Modular channel instead of max-nightly. Use this when the nightly solve fails
# to compile serving-time Mojo modules such as max._kv_cache_ops with:
#   unable to locate module 'std'
#   unable to locate module 'nn'
#
# This script delegates to create_max_gemma3_1b_py312_env.sh, but changes the
# project directory and Modular channel so the user gets a separate, stable
# workspace and keeps the failing nightly workspace available for diagnostics.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MONITORME_MAX_NEW_PROJECT_DIR="${MONITORME_MAX_NEW_PROJECT_DIR:-$HOME/dev/modular/project/quickstart_py312_stable}"
export MONITORME_MAX_CHANNEL="${MONITORME_MAX_CHANNEL:-https://conda.modular.com/max}"
export MONITORME_CONDA_FORGE_CHANNEL="${MONITORME_CONDA_FORGE_CHANNEL:-conda-forge}"

cat <<MSG
=== MonitorMe stable MAX/Gemma recovery workspace ===
Project: $MONITORME_MAX_NEW_PROJECT_DIR
Channel: $MONITORME_MAX_CHANNEL
Python:  python=3.12

This creates a stable-channel workspace separate from the failing max-nightly one.
MSG

exec "$SCRIPT_DIR/create_max_gemma3_1b_py312_env.sh"
