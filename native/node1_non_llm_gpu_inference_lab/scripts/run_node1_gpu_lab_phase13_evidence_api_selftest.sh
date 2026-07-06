#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LAB_DIR}/../.." && pwd)"

cd "${LAB_DIR}"
rm -rf build-cpu
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
./build-cpu/node1_non_llm_gpu_lab_selftest

cd "${REPO_ROOT}"
python -m pytest -q tests/test_node1_evidence_api_phase13.py

echo "node1_non_llm_gpu_lab Phase 13 Evidence API selftest PASS"
