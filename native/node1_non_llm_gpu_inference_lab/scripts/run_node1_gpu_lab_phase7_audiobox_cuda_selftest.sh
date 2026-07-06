#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$LAB_DIR/build/node1_non_llm_gpu_lab"

if [[ ! -x "$BIN" ]]; then
  echo "missing CUDA build binary: $BIN" >&2
  echo "run ./scripts/build_node1_gpu_lab.sh first" >&2
  exit 2
fi

for drift in 0 64 -64; do
  echo "===== AudioBox CUDA drift $drift ====="
  tmp_json="$(mktemp)"
  trap 'rm -f "$tmp_json"' EXIT
  "$BIN" \
    --mode audiobox-synthetic \
    --audio-samples 32768 \
    --sample-rate 48000 \
    --audio-window-samples 1024 \
    --audio-max-windows 32 \
    --silence-threshold 0.02 \
    --onset-threshold 0.08 \
    --max-lag 128 \
    --sync-drift-samples "$drift" \
    --gpu \
    --include-output > "$tmp_json"

  python3 - "$drift" "$tmp_json" <<'PY'
import json
import sys
from pathlib import Path

drift = int(sys.argv[1])
p = json.loads(Path(sys.argv[2]).read_text())
cpu = p["audiobox"]
gpu = p["audiobox_cuda"]
cmp = p["audiobox_cpu_cuda_comparison"]
assert p["ok"] is True
assert p["cuda_compiled"] is True
assert cpu["backend"] == "cpu"
assert gpu["backend"] == "cuda"
assert cpu["ok"] is True
assert gpu["ok"] is True
assert cpu["facts_only"] is True
assert gpu["facts_only"] is True
assert cmp["facts_only"] is True
assert cpu["sync_drift_samples"] == drift
assert gpu["sync_drift_samples"] == drift
assert cmp["ok"] is True
assert cmp["rms_close"] is True
assert cmp["peaks_close"] is True
assert cmp["correlation_close"] is True
assert cmp["masks_equal"] is True
assert cmp["drift_equal"] is True
assert cmp["metrics_close"] is True
assert cmp["mismatch_count"] == 0
assert float(cmp["max_abs_diff"]) <= 1e-4
print("PASS AudioBox CUDA drift=", drift, "kernel_ms=", gpu["timing"]["kernel_ms"])
PY
  rm -f "$tmp_json"
  trap - EXIT
done

echo "node1_non_llm_gpu_lab Phase 7 AudioBox CUDA selftest PASS"
