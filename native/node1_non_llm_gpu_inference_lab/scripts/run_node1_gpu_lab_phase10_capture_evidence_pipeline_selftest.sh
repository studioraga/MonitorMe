#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$LAB_DIR/../.." && pwd)"

cmake -S "$LAB_DIR" -B "$LAB_DIR/build-cpu" -DCMAKE_BUILD_TYPE=Release
cmake --build "$LAB_DIR/build-cpu" -j"$(nproc)"
"$LAB_DIR/build-cpu/node1_non_llm_gpu_lab_selftest"

cd "$REPO_ROOT"
python -m compileall -q monitor_me tests
python -m pytest -q tests/test_node1_capture_evidence_pipeline_phase10.py

echo "node1_non_llm_gpu_lab Phase 10 Capture-run Evidence Pipeline selftest PASS"
