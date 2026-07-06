#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${LAB_DIR}/../.." && pwd)"

cmake -S "${LAB_DIR}" -B "${LAB_DIR}/build-cpu" -DCMAKE_BUILD_TYPE=Release
cmake --build "${LAB_DIR}/build-cpu" -j"$(nproc)"
"${LAB_DIR}/build-cpu/node1_non_llm_gpu_lab_selftest"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT
MANIFEST="${TMP_DIR}/real_media_fingerprints.csv"
cat > "${MANIFEST}" <<'CSV'
clip_id,path,start_ms,duration_ms,bytes,motion_score,audio_score,lighting_delta,changed_pixels,fingerprint_source,decoded_width,decoded_height,ahash64,dhash64,fingerprint64,histogram16
clip_a,/tmp/keyframes/a.jpg,0,33,1000,0.9,0.0,12.0,1200,decoded_keyframe,160,120,12295233980590535660,12372613817689354991,2291424859679777924,16|16|16|16|16|16|16|16|16|16|16|16|16|16|16|16
clip_b,/tmp/keyframes/b.jpg,33,33,1100,0.8,0.0,10.0,1100,decoded_keyframe,160,120,12295233980590535660,12372613817689354991,2291424859679777924,16|16|16|16|16|16|16|16|16|16|16|16|16|16|16|16
clip_c,/tmp/keyframes/c.jpg,1000,33,900,0.2,0.0,5.0,300,decoded_keyframe,160,120,9041747396760141171,3844866112420011387,1715255923150493636,15|17|16|16|16|16|16|16|16|16|16|16|16|16|16|16
CSV

"${LAB_DIR}/build-cpu/node1_non_llm_gpu_lab" \
  --mode evidence-pipeline-manifest \
  --manifest "${MANIFEST}" \
  --max-batch-bytes 1600000 \
  --max-batch-clips 3 \
  --key-moments 2 \
  --min-key-gap-ms 1 \
  --dedup-hamming-threshold 0 \
  --fingerprint-width 16 \
  --fingerprint-height 16 \
  --fingerprint-cycle 6 \
  --include-output \
  > "${TMP_DIR}/evidence.json"

python3 - "${TMP_DIR}/evidence.json" <<'PY'
import json
import sys
from pathlib import Path

p = json.loads(Path(sys.argv[1]).read_text())
e = p["evidence_pipeline"]
assert p["ok"] is True
assert e["ok"] is True
assert e["real_media_ingestion"] is True
assert e["media_fingerprint_count"] == 3
assert e["synthetic_fingerprint_count"] == 0
assert e["fingerprint_count"] == 3
assert e["duplicate_group_count"] == 1
assert e["duplicate_clip_count"] == 1
assert e["safety"]["ok"] is True
assert e["safety"]["violation_count"] == 0
assert all(f["from_media"] is True for f in e["fingerprints"])
assert all(f["fingerprint_source"] == "decoded_keyframe" for f in e["fingerprints"])
assert all(f["decoded_width"] == 160 and f["decoded_height"] == 120 for f in e["fingerprints"])
assert e["storage_batch"]["manifest"][0]["has_media_fingerprint"] is True
print("PASS real-media evidence manifest media_fingerprint_count=", e["media_fingerprint_count"], "duplicates=", e["duplicate_clip_count"])
PY

cd "${REPO_ROOT}"
python -m pytest -q tests/test_node1_capture_real_fingerprint_phase11.py

echo "node1_non_llm_gpu_lab Phase 11 Real Media Fingerprint selftest PASS"
