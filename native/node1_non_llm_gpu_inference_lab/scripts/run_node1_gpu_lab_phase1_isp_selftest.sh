#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${LAB_DIR}/build-cpu"

cmake -S "${LAB_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

"${BUILD_DIR}/node1_non_llm_gpu_lab_selftest"

for filter in blur sharpen edge sobel-x sobel-y sobel-mag; do
  "${BUILD_DIR}/node1_non_llm_gpu_lab" \
    --mode isp-synthetic \
    --isp-filter "${filter}" \
    --width 64 \
    --height 48 \
    > /tmp/node1_gpu_lab_phase1_isp_${filter}.json
  python3 - "${filter}" "/tmp/node1_gpu_lab_phase1_isp_${filter}.json" <<'PY'
import json
import sys
filter_name = sys.argv[1]
path = sys.argv[2]
data = json.load(open(path, "r", encoding="utf-8"))
assert data["ok"] is True, data
assert data["isp"]["ok"] is True, data
assert data["isp"]["filter"] == filter_name, data
assert data["isp"]["schema"] == "node1_non_llm_isp_filters.v0.1", data
assert data["isp"]["facts_only"] is True, data
assert data["isp"]["pixels_processed"] == 64 * 48, data
print("PASS ISP", filter_name, "edge_energy=", data["isp"]["edge_energy"])
PY
done

echo "node1_non_llm_gpu_lab Phase 1 ISP CPU selftest PASS"
