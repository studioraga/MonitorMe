#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${LAB_DIR}/build/node1_non_llm_gpu_lab"

if [[ ! -x "${BIN}" ]]; then
  echo "CUDA binary not found: ${BIN}" >&2
  echo "Run ./scripts/build_node1_gpu_lab.sh first." >&2
  exit 2
fi

for scenario in sparse mixed dense; do
  echo "===== sparse ROI CUDA ${scenario} ====="
  payload_file="$(mktemp "/tmp/node1_sparse_roi_cuda_${scenario}.XXXXXX.json")"
  trap 'rm -f "${payload_file}"' RETURN

  "${BIN}" \
    --mode sparse-roi-synthetic \
    --scenario "${scenario}" \
    --width 320 \
    --height 240 \
    --target-width 16 \
    --target-height 16 \
    --gpu \
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
assert p["cuda_compiled"] is True, p
assert p["mode"] == "sparse-roi-synthetic", p
assert p["frame"]["ok"] is True, p
assert p["frame_cuda"]["ok"] is True, p
cpu = p["sparse_roi"]
gpu = p["sparse_roi_cuda"]
cmp = p["sparse_roi_cpu_cuda_comparison"]
assert cpu["ok"] is True and cpu["backend"] == "cpu", cpu
assert gpu["ok"] is True and gpu["backend"] == "cuda", gpu
assert cpu["facts_only"] is True and gpu["facts_only"] is True, (cpu, gpu)
assert cmp["ok"] is True, cmp
assert cmp["rois_equal"] is True, cmp
assert cmp["output_close"] is True, cmp
assert cmp["metrics_close"] is True, cmp
assert cmp["mismatch_count"] == 0, cmp
assert float(cmp["max_abs_diff"]) <= 1e-7, cmp
assert cmp["facts_only"] is True, cmp
assert cpu["roi_count"] == gpu["roi_count"] == cmp["roi_count"], (cpu, gpu, cmp)
print("PASS sparse ROI CUDA", scenario, "roi_count=", cpu["roi_count"], "kernel_ms=", gpu["timing"]["kernel_ms"])
PY

  rm -f "${payload_file}"
  trap - RETURN
done

echo "node1_non_llm_gpu_lab Phase 3 Sparse ROI CUDA selftest PASS"
