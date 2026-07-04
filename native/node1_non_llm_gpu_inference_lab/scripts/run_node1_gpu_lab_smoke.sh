#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="${MONITORME_GPU_LAB_BIN:-$LAB_DIR/build/node1_non_llm_gpu_lab}"

for scenario in sparse mixed dense; do
  echo "=== scenario=$scenario ==="
  "$BIN" --mode synthetic --scenario "$scenario" --gpu | python3 -m json.tool
 done
