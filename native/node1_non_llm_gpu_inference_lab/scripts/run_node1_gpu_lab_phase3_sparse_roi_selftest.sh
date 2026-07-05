#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${LAB_DIR}/build-cpu"

cmake -S "${LAB_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"

"${BUILD_DIR}/node1_non_llm_gpu_lab_selftest"

for scenario in sparse mixed dense; do
  echo "===== sparse ROI CPU ${scenario} ====="
  payload_file="$(mktemp "/tmp/node1_sparse_roi_cpu_${scenario}.XXXXXX.json")"
  trap 'rm -f "${payload_file}"' RETURN

  "${BUILD_DIR}/node1_non_llm_gpu_lab" \
    --mode sparse-roi-synthetic \
    --scenario "${scenario}" \
    --width 320 \
    --height 240 \
    --target-width 16 \
    --target-height 16 \
    --include-output \
    > "${payload_file}"

  python3 - "${scenario}" "${payload_file}" <<'PY'
import json
import sys
from pathlib import Path
scenario = sys.argv[1]
payload_path = Path(sys.argv[2])
p = json.loads(payload_path.read_text())
assert p["ok"] is True, p
assert p["mode"] == "sparse-roi-synthetic", p
assert p["frame"]["ok"] is True, p
roi = p["sparse_roi"]
assert roi["ok"] is True, roi
assert roi["backend"] == "cpu", roi
assert roi["facts_only"] is True, roi
assert roi["roi_count"] == roi["active_tiles"], roi
assert roi["roi_count"] > 0, roi
assert roi["target_width"] == 16 and roi["target_height"] == 16, roi
assert roi["output_elements"] == roi["roi_count"] * 16 * 16, roi
assert len(roi["rois"]) == roi["roi_count"], roi
assert len(roi["normalized"]) == roi["output_elements"], roi
assert all(0.0 <= float(v) <= 1.0 for v in roi["normalized"]), roi
print("PASS sparse ROI CPU", scenario, "roi_count=", roi["roi_count"], "tile_mask=", roi["tile_mask_hex"])
PY

  rm -f "${payload_file}"
  trap - RETURN
done

echo "node1_non_llm_gpu_lab Phase 3 Sparse ROI CPU selftest PASS"
