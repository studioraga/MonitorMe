#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${LAB_DIR}/build-cpu"

cmake -S "${LAB_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"
"${BUILD_DIR}/node1_non_llm_gpu_lab_selftest"
"${BUILD_DIR}/node1_non_llm_gpu_lab" --mode synthetic --scenario sparse | python3 -m json.tool >/dev/null
"${BUILD_DIR}/node1_non_llm_gpu_lab" --mode synthetic --scenario mixed | python3 -m json.tool >/dev/null
"${BUILD_DIR}/node1_non_llm_gpu_lab" --mode synthetic --scenario dense | python3 -m json.tool >/dev/null

echo "node1_non_llm_gpu_lab Phase 0 CPU selftest PASS"
