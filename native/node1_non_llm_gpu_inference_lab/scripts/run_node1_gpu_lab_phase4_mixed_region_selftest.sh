#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${LAB_DIR}/build-cpu"

cmake -S "${LAB_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"
"${BUILD_DIR}/node1_non_llm_gpu_lab_selftest"

for scenario in contiguous scattered dense; do
  echo "===== mixed region CPU ${scenario} ====="
  tmp_json=$(mktemp)
  "${BUILD_DIR}/node1_non_llm_gpu_lab" \
    --mode mixed-region-synthetic \
    --scenario "${scenario}" \
    --width 320 \
    --height 240 \
    --target-width 16 \
    --target-height 16 \
    --include-output > "${tmp_json}"
  python3 - "${scenario}" "${tmp_json}" <<'PY'
import json
import sys
from pathlib import Path

scenario = sys.argv[1]
p = json.loads(Path(sys.argv[2]).read_text())
mr = p["mixed_region"]
assert p["ok"] is True
assert p["mode"] == "mixed-region-synthetic"
assert p["frame"]["ok"] is True
assert mr["ok"] is True
assert mr["backend"] == "cpu"
assert mr["schema"] == "node1_non_llm_mixed_region.v0.1"
assert mr["facts_only"] is True
assert "identity" in mr["note"]
assert mr["group_count"] == len(mr["groups"])
assert mr["output_elements"] == mr["group_count"] * mr["target_width"] * mr["target_height"]
assert len(mr["normalized"]) == mr["output_elements"]
if scenario == "contiguous":
    assert p["frame"]["path"] == "mixed"
    assert mr["classification"] == "contiguous"
    assert mr["group_count"] == 1
    assert mr["groups"][0]["tile_count"] == 16
elif scenario == "scattered":
    assert p["frame"]["path"] == "mixed"
    assert mr["classification"] == "scattered"
    assert mr["group_count"] == 16
elif scenario == "dense":
    assert p["frame"]["path"] == "dense"
    assert mr["classification"] == "contiguous"
    assert mr["group_count"] == 1
    assert mr["groups"][0]["tile_count"] == 32
print("PASS mixed region CPU", scenario, "group_count=", mr["group_count"], "classification=", mr["classification"])
PY
  rm -f "${tmp_json}"
done

echo "node1_non_llm_gpu_lab Phase 4 Mixed Region CPU selftest PASS"
