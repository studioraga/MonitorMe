#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="native/node1_non_llm_gpu_inference_lab/build-cpu"
rm -rf "$BUILD_DIR"
cmake -S native/node1_non_llm_gpu_inference_lab -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNODE1_NON_LLM_GPU_LAB_WITH_CUDA=OFF
cmake --build "$BUILD_DIR" -j"$(nproc)"
"$BUILD_DIR/node1_non_llm_gpu_lab_selftest"

python -m pytest -q tests/test_node1_operator_dashboard_grafana_phase19.py

echo "node1_non_llm_gpu_lab Phase 19 Grafana Dashboard selftest PASS"
