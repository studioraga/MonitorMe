#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$LAB_DIR/build-cpu"
BIN="$BUILD_DIR/node1_non_llm_gpu_lab"
SELFTEST="$BUILD_DIR/node1_non_llm_gpu_lab_selftest"

cmake -S "$LAB_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"
"$SELFTEST"

for drift in 0 64 -64; do
  echo "===== AudioBox CPU drift $drift ====="
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
    --include-output > "$tmp_json"

  python3 - "$drift" "$tmp_json" <<'PY'
import json
import sys
from pathlib import Path

drift = int(sys.argv[1])
p = json.loads(Path(sys.argv[2]).read_text())
a = p["audiobox"]
assert p["ok"] is True
assert p["mode"] == "audiobox-synthetic"
assert p["cuda_compiled"] is False
assert a["ok"] is True
assert a["backend"] == "cpu"
assert a["facts_only"] is True
assert a["windows"] == 32
assert len(a["rms"]) == 32
assert len(a["peaks"]) == 32
assert len(a["correlation_scores"]) == 257
assert a["active_windows"] > 0
assert a["silent_windows"] > 0
assert a["onset_count"] >= 1
assert a["max_peak"] > 0.35
assert a["sync_drift_samples"] == drift
assert abs(a["sync_drift_ms"] - (1000.0 * drift / 48000.0)) < 1e-4
assert a["sync_correlation_abs"] > 0.99
print("PASS AudioBox CPU drift=", drift, "active_windows=", a["active_windows"], "onsets=", a["onset_count"])
PY
  rm -f "$tmp_json"
  trap - EXIT
done

echo "node1_non_llm_gpu_lab Phase 7 AudioBox CPU selftest PASS"
