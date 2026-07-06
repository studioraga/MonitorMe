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

for clips in 8 12 20; do
  echo "===== storage batch CPU clips $clips ====="
  tmp_json="$(mktemp)"
  trap 'rm -f "$tmp_json"' EXIT
  "$BIN" \
    --mode storage-batch-synthetic \
    --clips "$clips" \
    --max-batch-bytes 1600000 \
    --max-batch-clips 3 \
    --key-moments 4 \
    --min-key-gap-ms 1000 \
    --include-output > "$tmp_json"

  python3 - "$clips" "$tmp_json" <<'PY'
import json
import sys
from pathlib import Path

clips = int(sys.argv[1])
p = json.loads(Path(sys.argv[2]).read_text())
s = p["storage_batch"]
assert p["ok"] is True
assert p["mode"] == "storage-batch-synthetic"
assert p["cuda_compiled"] is False
assert s["ok"] is True
assert s["backend"] == "cpu"
assert s["facts_only"] is True
assert s["manifest_entries"] == clips
assert s["clip_count"] == clips
assert s["batch_count"] >= 1
assert s["key_moment_count"] == min(4, clips)
assert s["planned_read_bytes"] == s["total_manifest_bytes"]
assert len(s["batches"]) == s["batch_count"]
assert len(s["key_moments"]) == s["key_moment_count"]
assert len(s["manifest"]) == clips
assert s["timeline"]["clip_count"] == clips
assert s["timeline"]["timeline_span_ms"] > 0
assert s["timeline"]["total_bytes"] == s["total_manifest_bytes"]
for b in s["batches"]:
    assert b["clip_count"] >= 1
    assert b["total_bytes"] > 0
    assert len(b["clip_indices"]) == b["clip_count"]
for k in s["key_moments"]:
    assert k["priority_score"] > 0
    assert k["reason"]
print("PASS storage batch CPU clips=", clips, "batches=", s["batch_count"], "key_moments=", s["key_moment_count"])
PY
  rm -f "$tmp_json"
  trap - EXIT
done

manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
cat > "$manifest" <<'CSV'
clip_id,path,start_ms,duration_ms,bytes,motion_score,audio_score,lighting_delta,changed_pixels
a,clips/a.mkv,0,1000,500000,0.1,0.0,2,100
b,clips/b.mkv,1200,1000,600000,0.9,0.2,30,30000
c,clips/c.mkv,2600,1000,700000,0.2,0.8,12,12000
d,clips/d.mkv,4000,1000,800000,0.7,0.7,50,50000
CSV

echo "===== storage batch manifest scan ====="
tmp_json="$(mktemp)"
trap 'rm -f "$manifest" "$tmp_json"' EXIT
"$BIN" \
  --mode storage-batch-manifest \
  --manifest "$manifest" \
  --max-batch-bytes 1200000 \
  --max-batch-clips 2 \
  --key-moments 2 \
  --min-key-gap-ms 1000 \
  --include-output > "$tmp_json"
python3 - "$tmp_json" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
s = p["storage_batch"]
assert p["ok"] is True
assert p["mode"] == "storage-batch-manifest"
assert s["ok"] is True
assert s["manifest_entries"] == 4
assert s["batch_count"] == 3
assert s["key_moment_count"] == 2
assert s["key_moments"][0]["clip_id"] == "d"
assert s["planned_read_bytes"] == 2600000
print("PASS storage batch manifest scan batches=", s["batch_count"], "top_key=", s["key_moments"][0]["clip_id"])
PY
rm -f "$manifest" "$tmp_json"
trap - EXIT

echo "node1_non_llm_gpu_lab Phase 8 Storage Batch selftest PASS"
