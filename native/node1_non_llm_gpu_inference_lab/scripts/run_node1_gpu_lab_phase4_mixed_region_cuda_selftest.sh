#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LAB_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${LAB_DIR}/build"
BIN="${BUILD_DIR}/node1_non_llm_gpu_lab"

if [[ ! -x "${BIN}" ]]; then
  echo "CUDA binary not found at ${BIN}; run ./scripts/build_node1_gpu_lab.sh first" >&2
  exit 2
fi

for scenario in contiguous scattered dense; do
  echo "===== mixed region CUDA ${scenario} ====="
  tmp_json=$(mktemp)
  "${BIN}" \
    --mode mixed-region-synthetic \
    --scenario "${scenario}" \
    --width 320 \
    --height 240 \
    --target-width 16 \
    --target-height 16 \
    --gpu \
    --include-output > "${tmp_json}"
  python3 - "${scenario}" "${tmp_json}" <<'PY'
import json
import sys
from pathlib import Path

scenario = sys.argv[1]
p = json.loads(Path(sys.argv[2]).read_text())
cpu = p["mixed_region"]
gpu = p["mixed_region_cuda"]
cmp = p["mixed_region_cpu_cuda_comparison"]
assert p["ok"] is True
assert p["cuda_compiled"] is True
assert cpu["ok"] is True and gpu["ok"] is True
assert cpu["backend"] == "cpu"
assert gpu["backend"] == "cuda"
assert cpu["facts_only"] is True and gpu["facts_only"] is True and cmp["facts_only"] is True
assert cpu["groups"] == gpu["groups"]
assert cpu["normalized"] == gpu["normalized"]
assert cmp["ok"] is True
assert cmp["groups_equal"] is True
assert cmp["output_close"] is True
assert cmp["metrics_close"] is True
assert cmp["mismatch_count"] == 0
assert float(cmp["max_abs_diff"]) <= 1e-7
if scenario == "contiguous":
    assert p["frame"]["path"] == "mixed"
    assert cpu["classification"] == "contiguous"
    assert cpu["group_count"] == 1
elif scenario == "scattered":
    assert p["frame"]["path"] == "mixed"
    assert cpu["classification"] == "scattered"
    assert cpu["group_count"] == 16
elif scenario == "dense":
    assert p["frame"]["path"] == "dense"
    assert cpu["classification"] == "contiguous"
    assert cpu["group_count"] == 1
print("PASS mixed region CUDA", scenario, "group_count=", cpu["group_count"], "kernel_ms=", gpu["timing"]["kernel_ms"])
PY
  rm -f "${tmp_json}"
done

echo "node1_non_llm_gpu_lab Phase 4 Mixed Region CUDA selftest PASS"
