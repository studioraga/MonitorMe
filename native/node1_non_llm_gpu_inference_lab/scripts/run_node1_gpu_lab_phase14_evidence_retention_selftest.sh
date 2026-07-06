#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LAB_DIR}/../.." && pwd)"

cd "${LAB_DIR}"
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DNODE1_NON_LLM_GPU_LAB_WITH_CUDA=OFF
cmake --build build-cpu -j"$(nproc)"
./build-cpu/node1_non_llm_gpu_lab_selftest

cd "${REPO_ROOT}"
python -m pytest -q tests/test_node1_evidence_retention_phase14.py

echo "node1_non_llm_gpu_lab Phase 14 Evidence Index Retention selftest PASS"
