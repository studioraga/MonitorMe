#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$LAB_DIR/build-cpu"

cmake -S "$LAB_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DNODE1_NON_LLM_ENABLE_CUDA=OFF
cmake --build "$BUILD_DIR" -j"$(nproc)"

"$BUILD_DIR/node1_non_llm_gpu_lab_selftest"

for scenario in dense mixed sparse; do
  echo "===== dense full-frame CPU $scenario ====="
  tmp_json="$(mktemp)"
  trap 'rm -f "$tmp_json"' RETURN
  "$BUILD_DIR/node1_non_llm_gpu_lab" \
    --mode dense-full-frame-synthetic \
    --scenario "$scenario" \
    --width 320 \
    --height 240 \
    --include-output > "$tmp_json"
  python3 - "$scenario" "$tmp_json" <<'PY'
import json
import sys
from pathlib import Path
scenario = sys.argv[1]
p = json.loads(Path(sys.argv[2]).read_text())
dense = p["dense_full_frame"]
assert p["ok"] is True
assert p["mode"] == "dense-full-frame-synthetic"
assert dense["ok"] is True
assert dense["backend"] == "cpu"
assert dense["facts_only"] is True
assert dense["histogram_total"] == dense["pixels_processed"]
assert len(dense["diff_histogram"]) == 256
assert len(dense["normalized"]) == dense["pixels_processed"]
if scenario == "dense":
    assert p["frame"]["path"] == "dense"
    assert dense["changed_pixels"] == dense["pixels_processed"]
    assert dense["diff_histogram"][200] == dense["pixels_processed"]
    assert abs(dense["lighting_delta"] - 200.0) <= 1e-9
elif scenario == "mixed":
    assert p["frame"]["path"] == "mixed"
    assert dense["changed_pixels"] == p["frame"]["changed_pixels"]
elif scenario == "sparse":
    assert p["frame"]["path"] == "sparse"
    assert dense["changed_pixels"] == p["frame"]["changed_pixels"]
print("PASS dense full-frame CPU", scenario, "changed_pixels=", dense["changed_pixels"], "lighting_delta=", dense["lighting_delta"])
PY
  rm -f "$tmp_json"
  trap - RETURN
done

echo "node1_non_llm_gpu_lab Phase 5 Dense Full-Frame CPU selftest PASS"
