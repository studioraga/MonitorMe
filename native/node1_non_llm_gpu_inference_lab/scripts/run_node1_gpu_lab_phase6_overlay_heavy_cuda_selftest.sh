#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ ! -x ./build/node1_non_llm_gpu_lab ]]; then
  ./scripts/build_node1_gpu_lab.sh >/tmp/node1_gpu_lab_phase6_cuda_build.log 2>&1 || { cat /tmp/node1_gpu_lab_phase6_cuda_build.log >&2; exit 1; }
fi

for scenario in mixed dense sparse; do
  echo "===== overlay-heavy CUDA ${scenario} ====="
  tmp_json="$(mktemp)"
  ./build/node1_non_llm_gpu_lab \
    --mode overlay-heavy-synthetic \
    --scenario "${scenario}" \
    --width 320 \
    --height 240 \
    --thumbnail-width 64 \
    --thumbnail-height 48 \
    --gpu \
    --include-output \
    > "${tmp_json}"
  python3 - "${scenario}" "${tmp_json}" <<'PY'
import json
import sys
from pathlib import Path
scenario = sys.argv[1]
p = json.loads(Path(sys.argv[2]).read_text())
cpu = p["overlay_heavy"]
gpu = p["overlay_heavy_cuda"]
cmp = p["overlay_heavy_cpu_cuda_comparison"]
assert p["ok"] is True
assert p["cuda_compiled"] is True
assert cpu["backend"] == "cpu"
assert gpu["backend"] == "cuda"
assert cpu["ok"] is True
assert gpu["ok"] is True
assert cpu["facts_only"] is True
assert gpu["facts_only"] is True
assert cpu["heatmap"] == gpu["heatmap"]
assert cpu["overlay_rgb"] == gpu["overlay_rgb"]
assert cpu["thumbnail_rgb"] == gpu["thumbnail_rgb"]
assert cmp["ok"] is True
assert cmp["heatmap_equal"] is True
assert cmp["overlay_equal"] is True
assert cmp["thumbnail_equal"] is True
assert cmp["mismatch_count"] == 0
assert cmp["max_abs_diff"] == 0
assert cmp["metrics_close"] is True
assert cmp["facts_only"] is True
if scenario == "dense":
    assert p["frame"]["path"] == "dense"
    assert cpu["changed_pixels"] == cpu["pixels_processed"]
elif scenario == "mixed":
    assert p["frame"]["path"] == "mixed"
elif scenario == "sparse":
    assert p["frame"]["path"] == "sparse"
print("PASS overlay-heavy CUDA", scenario, "changed_pixels=", cpu["changed_pixels"], "kernel_ms=", gpu["timing"]["kernel_ms"])
PY
  rm -f "${tmp_json}"
done

echo "node1_non_llm_gpu_lab Phase 6 Overlay-heavy CUDA selftest PASS"
