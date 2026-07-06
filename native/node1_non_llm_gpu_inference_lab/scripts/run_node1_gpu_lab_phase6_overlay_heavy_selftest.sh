#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

rm -rf build-cpu
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
./build-cpu/node1_non_llm_gpu_lab_selftest

for scenario in mixed dense sparse; do
  echo "===== overlay-heavy CPU ${scenario} ====="
  tmp_json="$(mktemp)"
  ./build-cpu/node1_non_llm_gpu_lab \
    --mode overlay-heavy-synthetic \
    --scenario "${scenario}" \
    --width 320 \
    --height 240 \
    --thumbnail-width 64 \
    --thumbnail-height 48 \
    --include-output \
    > "${tmp_json}"
  python3 - "${scenario}" "${tmp_json}" <<'PY'
import json
import sys
from pathlib import Path
scenario = sys.argv[1]
p = json.loads(Path(sys.argv[2]).read_text())
o = p["overlay_heavy"]
assert p["ok"] is True
assert p["mode"] == "overlay-heavy-synthetic"
assert o["ok"] is True
assert o["backend"] == "cpu"
assert o["schema"] == "node1_non_llm_overlay_heavy.v0.1"
assert o["facts_only"] is True
assert o["pixels_processed"] == 320 * 240
assert o["heatmap_elements"] == o["pixels_processed"]
assert o["overlay_rgb_elements"] == o["pixels_processed"] * 3
assert o["thumbnail_rgb_elements"] == 64 * 48 * 3
assert len(o["heatmap"]) == o["pixels_processed"]
assert len(o["overlay_rgb"]) == o["pixels_processed"] * 3
assert len(o["thumbnail_rgb"]) == 64 * 48 * 3
assert o["before_after_max_diff"] == o["heatmap_max"]
assert o["changed_pixels"] == p["frame"]["changed_pixels"]
assert o["changed_ratio"] == p["frame"]["changed_ratio"]
if scenario == "dense":
    assert p["frame"]["path"] == "dense"
    assert o["changed_pixels"] == o["pixels_processed"]
elif scenario == "mixed":
    assert p["frame"]["path"] == "mixed"
elif scenario == "sparse":
    assert p["frame"]["path"] == "sparse"
print("PASS overlay-heavy CPU", scenario, "changed_pixels=", o["changed_pixels"], "thumbnail_elements=", o["thumbnail_rgb_elements"])
PY
  rm -f "${tmp_json}"
done

echo "node1_non_llm_gpu_lab Phase 6 Overlay-heavy CPU selftest PASS"
