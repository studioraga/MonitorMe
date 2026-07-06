#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

BUILD_DIR="native/node1_non_llm_gpu_inference_lab/build-cpu"
rm -rf "$BUILD_DIR"
cmake -S native/node1_non_llm_gpu_inference_lab -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNODE1_NON_LLM_GPU_LAB_WITH_CUDA=OFF
cmake --build "$BUILD_DIR" -j"$(nproc)"
"$BUILD_DIR/node1_non_llm_gpu_lab_selftest"

TMP_PLAN="$(mktemp)"
trap 'rm -f "$TMP_PLAN"' EXIT
native/node1_non_llm_gpu_inference_lab/scripts/profile_node1_gpu_lab_nsight_compute.sh \
  --dry-run \
  --workload dense_full_frame \
  --output-dir results/node1_gpu_lab/nsight_compute/phase20_selftest_dry_run > "$TMP_PLAN"

python3 - "$TMP_PLAN" <<'PY'
import json
import sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
assert plan["ok"] is True
assert plan["schema"] == "monitorme.node1_nsight_compute_profile_run.v0.1"
assert plan["mode"] == "dry_run"
assert plan["phase"] == 20
assert plan["workloads"] == ["dense_full_frame"]
assert plan["source_scope"]["synthetic_inputs_only"] is True
assert plan["source_scope"]["media_decode"] is False
assert plan["privacy"]["external_upload"] is False
cmds = plan["commands"]["workload_commands"]
assert len(cmds) == 1
cmd = cmds[0]["ncu_command"]
assert "ncu" in cmd[0]
assert "--target-processes" in cmd
assert "all" in cmd
assert "--export" in cmd
assert "dense-full-frame-synthetic" in cmds[0]["ncu_command_string"]
assert "--gpu" in cmd
assert "http://" not in json.dumps(plan)
assert "https://" not in json.dumps(plan)
print("Phase 20 dry-run Nsight Compute plan PASS")
PY

echo "node1_non_llm_gpu_lab Phase 20 Nsight Compute profiling selftest PASS"
