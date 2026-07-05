#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${LAB_DIR}"

BUILD_LOG="${BUILD_LOG:-/tmp/node1_gpu_lab_phase2_cuda_build.log}"
if ! ./scripts/build_node1_gpu_lab.sh >"${BUILD_LOG}" 2>&1; then
  cat "${BUILD_LOG}" >&2
  exit 1
fi

for filter in blur sharpen edge sobel-x sobel-y sobel-mag; do
  json="$(./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter "${filter}" --width 64 --height 48 --gpu --include-output)"
  tmp_json="$(mktemp)"
  printf '%s\n' "${json}" > "${tmp_json}"
  python3 - "${filter}" "${tmp_json}" <<'PY'
import json
import sys

expected_filter = sys.argv[1]
with open(sys.argv[2], "r", encoding="utf-8") as f:
    payload = json.load(f)
assert payload["ok"] is True, payload
assert payload["mode"] == "isp-synthetic", payload
assert payload["cuda_compiled"] is True, payload
assert payload["isp"]["ok"] is True, payload
assert payload["isp_cuda"]["ok"] is True, payload
assert payload["isp"]["backend"] == "cpu", payload
assert payload["isp_cuda"]["backend"] == "cuda", payload
assert payload["isp"]["filter"] == expected_filter, payload
assert payload["isp_cuda"]["filter"] == expected_filter, payload
assert payload["isp"]["facts_only"] is True, payload
assert payload["isp_cuda"]["facts_only"] is True, payload
comparison = payload["isp_cpu_cuda_comparison"]
assert comparison["ok"] is True, comparison
assert comparison["schema"] == "node1_non_llm_isp_cpu_cuda_compare.v0.1", comparison
assert comparison["output_equal"] is True, comparison
assert comparison["metrics_close"] is True, comparison
assert comparison["mismatch_count"] == 0, comparison
assert comparison["max_abs_diff"] == 0, comparison
assert comparison["facts_only"] is True, comparison
print(f"PASS ISP CUDA {expected_filter} kernel_ms={payload['isp_cuda']['timing']['kernel_ms']}")
PY
  rm -f "${tmp_json}"
done

echo "node1_non_llm_gpu_lab Phase 2 CUDA ISP selftest PASS"
