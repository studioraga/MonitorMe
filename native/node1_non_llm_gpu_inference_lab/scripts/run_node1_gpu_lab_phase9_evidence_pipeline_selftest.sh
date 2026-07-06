#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${LAB_DIR}/build-cpu"
BIN="${BUILD_DIR}/node1_non_llm_gpu_lab"
SELFTEST="${BUILD_DIR}/node1_non_llm_gpu_lab_selftest"
PYTHON_BIN="${PYTHON:-python3}"

cmake -S "${LAB_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" -j"$(nproc)"
"${SELFTEST}"

for clips in 8 12 20; do
  echo "===== evidence pipeline CPU clips ${clips} ====="
  payload_file="$(mktemp "/tmp/node1_evidence_pipeline_${clips}.XXXXXX.json")"
  "${BIN}" \
    --mode evidence-pipeline-synthetic \
    --clips "${clips}" \
    --max-batch-bytes 1600000 \
    --max-batch-clips 3 \
    --key-moments 4 \
    --min-key-gap-ms 1000 \
    --dedup-hamming-threshold 0 \
    --fingerprint-width 16 \
    --fingerprint-height 16 \
    --fingerprint-cycle 6 \
    --include-output > "${payload_file}"
  "${PYTHON_BIN}" - "${clips}" "${payload_file}" <<'PY'
import json
import sys
from pathlib import Path

clips = int(sys.argv[1])
p = json.loads(Path(sys.argv[2]).read_text())
e = p["evidence_pipeline"]
storage = p["storage_batch"]
assert p["ok"] is True
assert p["mode"] == "evidence-pipeline-synthetic"
assert e["ok"] is True
assert e["backend"] == "cpu"
assert e["schema"] == "node1_non_llm_evidence_pipeline.v0.1"
assert e["facts_only"] is True
assert e["manifest_entries"] == clips
assert e["fingerprint_count"] == clips
assert e["storage_batch"]["clip_count"] == clips
assert e["storage_batch"]["planned_read_bytes"] == e["storage_batch"]["total_manifest_bytes"]
assert e["planned_read_bytes"] == e["total_manifest_bytes"]
assert e["batch_count"] == storage["batch_count"]
assert e["key_moment_count"] <= 4
assert e["unique_clip_count"] + e["duplicate_clip_count"] == e["fingerprint_count"]
assert e["duplicate_group_count"] >= 1
assert len(e["fingerprints"]) == clips
assert len(e["duplicate_groups"]) == e["duplicate_group_count"]
assert e["timeline"]["clip_count"] == clips
assert e["latency"]["total_ms"] >= 0
assert e["latency"]["planned_read_mb"] > 0
assert e["safety"]["ok"] is True
assert e["safety"]["violation_count"] == 0
assert e["safety"]["batch_plan_ok"] is True
assert e["safety"]["fingerprint_ok"] is True
assert e["safety"]["dedup_ok"] is True
print("PASS evidence pipeline CPU clips=", clips, "duplicates=", e["duplicate_clip_count"])
PY
  rm -f "${payload_file}"
done

manifest_file="$(mktemp /tmp/node1_evidence_manifest.XXXXXX.csv)"
cat > "${manifest_file}" <<'CSV'
clip_id,path,start_ms,duration_ms,bytes,motion_score,audio_score,lighting_delta,changed_pixels
a,clips/a.mkv,0,1000,500000,0.1,0.0,2,100
b,clips/b.mkv,1200,1000,600000,0.9,0.2,30,30000
c,clips/c.mkv,2600,1000,700000,0.2,0.8,12,12000
d,clips/d.mkv,4000,1000,800000,0.7,0.7,50,50000
CSV
payload_file="$(mktemp /tmp/node1_evidence_manifest.XXXXXX.json)"
"${BIN}" \
  --mode evidence-pipeline-manifest \
  --manifest "${manifest_file}" \
  --max-batch-bytes 1200000 \
  --max-batch-clips 2 \
  --key-moments 2 \
  --min-key-gap-ms 1000 \
  --dedup-hamming-threshold 0 \
  --fingerprint-cycle 2 \
  --include-output > "${payload_file}"
"${PYTHON_BIN}" - "${payload_file}" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text())
e = p["evidence_pipeline"]
assert p["ok"] is True
assert p["mode"] == "evidence-pipeline-manifest"
assert e["ok"] is True
assert e["manifest_entries"] == 4
assert e["storage_batch"]["batch_count"] == 3
assert e["key_moment_count"] == 2
assert e["safety"]["ok"] is True
assert e["safety"]["violation_count"] == 0
assert e["storage_batch"]["planned_read_bytes"] == 2600000
print("PASS evidence pipeline manifest scan")
PY
rm -f "${manifest_file}" "${payload_file}"

echo "node1_non_llm_gpu_lab Phase 9 Evidence Pipeline selftest PASS"
