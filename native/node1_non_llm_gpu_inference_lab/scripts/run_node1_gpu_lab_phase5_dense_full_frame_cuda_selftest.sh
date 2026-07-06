#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$LAB_DIR/build"
BIN="$BUILD_DIR/node1_non_llm_gpu_lab"

if [[ ! -x "$BIN" ]]; then
  "$SCRIPT_DIR/build_node1_gpu_lab.sh" >/dev/null
fi

for scenario in dense mixed sparse; do
  echo "===== dense full-frame CUDA $scenario ====="
  tmp_json="$(mktemp)"
  trap 'rm -f "$tmp_json"' RETURN
  "$BIN" \
    --mode dense-full-frame-synthetic \
    --scenario "$scenario" \
    --width 320 \
    --height 240 \
    --gpu \
    --include-output > "$tmp_json"
  python3 - "$scenario" "$tmp_json" <<'PY'
import json
import sys
from pathlib import Path
scenario = sys.argv[1]
p = json.loads(Path(sys.argv[2]).read_text())
cpu = p["dense_full_frame"]
gpu = p["dense_full_frame_cuda"]
cmp = p["dense_full_frame_cpu_cuda_comparison"]
assert p["ok"] is True
assert p["cuda_compiled"] is True
assert p["mode"] == "dense-full-frame-synthetic"
assert cpu["ok"] is True and cpu["backend"] == "cpu"
assert gpu["ok"] is True and gpu["backend"] == "cuda"
assert cpu["facts_only"] is True and gpu["facts_only"] is True and cmp["facts_only"] is True
assert cpu["diff_histogram"] == gpu["diff_histogram"]
assert cpu["normalized"] == gpu["normalized"]
assert cmp["ok"] is True
assert cmp["histogram_equal"] is True
assert cmp["normalized_close"] is True
assert cmp["reductions_close"] is True
assert cmp["mismatch_count"] == 0
assert float(cmp["max_abs_diff"]) <= 1e-7
if scenario == "dense":
    assert p["frame"]["path"] == "dense"
    assert cpu["changed_pixels"] == cpu["pixels_processed"]
    assert cpu["diff_histogram"][200] == cpu["pixels_processed"]
elif scenario == "mixed":
    assert p["frame"]["path"] == "mixed"
elif scenario == "sparse":
    assert p["frame"]["path"] == "sparse"
print("PASS dense full-frame CUDA", scenario, "changed_pixels=", cpu["changed_pixels"], "kernel_ms=", gpu["timing"]["kernel_ms"])
PY
  rm -f "$tmp_json"
  trap - RETURN
done

echo "node1_non_llm_gpu_lab Phase 5 Dense Full-Frame CUDA selftest PASS"
