# Node1 Non-LLM GPU Inference Lab

This native sidecar module is the first C++/CUDA implementation path for MonitorMe's deterministic, non-LLM CPU/GPU workload routing.

It reuses the sparse/dense/mixed idea from the earlier workload optimization labs and applies it to real camera/audio artifacts:

```text
previous frame + current frame
        |
        v
32-tile frame-difference mask
        |
        v
active tile count + low/high half occupancy
        |
        v
sparse / mixed / dense route decision
        |
        +--> CPU fallback path
        +--> CUDA tile-mask kernel when built with CUDA
```

The module deliberately does **not** identify people, infer intent, or make safety claims. It only produces workload facts useful for routing and profiling.

## Build: CPU-only smoke build

Use this when developing on a host that may not have CUDA installed:

```bash
cd native/node1_non_llm_gpu_inference_lab
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
./build-cpu/node1_non_llm_gpu_lab --mode synthetic --scenario mixed
```

## Build: Node1 CUDA build

On Node1 with CUDA 13.3 and RTX 5060 Ti / `sm_120`:

```bash
cd native/node1_non_llm_gpu_inference_lab
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNODE1_NON_LLM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j"$(nproc)"
./build/node1_non_llm_gpu_lab --mode synthetic --scenario sparse --gpu
./build/node1_non_llm_gpu_lab --mode synthetic --scenario mixed --gpu
./build/node1_non_llm_gpu_lab --mode synthetic --scenario dense --gpu
```

## Analyze raw gray frames

The Python MonitorMe bridge writes temporary raw grayscale frames and calls this binary like this:

```bash
./build/node1_non_llm_gpu_lab \
  --mode analyze-raw-gray \
  --prev /tmp/prev.gray \
  --curr /tmp/curr.gray \
  --width 1280 \
  --height 720 \
  --tile-cols 8 \
  --tile-rows 4 \
  --pixel-threshold 30 \
  --gpu
```

## Analyze raw float32 audio windows

```bash
./build/node1_non_llm_gpu_lab \
  --mode audio-raw-f32 \
  --audio /tmp/audio.f32 \
  --audio-window-samples 1024 \
  --audio-threshold 0.05 \
  --gpu
```

## JSON output contract

The executable prints one JSON object containing:

```text
schema
mode
cuda_compiled
frame.tile_mask_hex
frame.active_tiles
frame.path = sparse|mixed|dense
frame.low_half_active_tiles
frame.high_half_active_tiles
audio.event_mask_hex
audio.active_windows
```

MonitorMe stores these as child `gpu_workload_profiled` evidence rows when `--gpu-lab-enabled` is passed to `capture-run`.

## Phase 0 hardening layout

Phase 0 keeps the v0.1 feature contract unchanged while splitting the native lab
into reusable pieces for the next ISP, sparse ROI, dense frame, mixed region,
overlay, audio, storage, and latency phases.

New reusable pieces:

```text
include/node1_non_llm/gpu_lab_types.hpp       shared structs, route helpers, validation
include/node1_non_llm/gpu_lab_timing.hpp      host timing structs/helpers
include/node1_non_llm/gpu_lab_json.hpp        JSON serialization helpers
include/node1_non_llm/gpu_lab_cuda_utils.cuh  CUDA error/event timing helpers
src/gpu_lab_types.cpp                         shared helper implementation
src/gpu_lab_timing.cpp                        timing implementation
src/gpu_lab_json.cpp                          JSON implementation
src/gpu_lab_cuda_utils.cu                     CUDA helper implementation
src/gpu_lab_selftest.cpp                      stricter native CPU self-test
```

The existing CLI still supports:

```bash
./build/node1_non_llm_gpu_lab --mode synthetic --scenario sparse --gpu
./build/node1_non_llm_gpu_lab --mode synthetic --scenario mixed --gpu
./build/node1_non_llm_gpu_lab --mode synthetic --scenario dense --gpu
./build/node1_non_llm_gpu_lab --mode analyze-raw-gray --prev prev.gray --curr curr.gray --width 1280 --height 720 --gpu
./build/node1_non_llm_gpu_lab --mode audio-raw-f32 --audio samples.f32 --gpu
```

Every frame/audio result now includes a `timing` object:

```json
{"h2d_ms":0.0,"kernel_ms":0.0,"d2h_ms":0.0,"total_ms":0.0}
```

CPU paths populate `kernel_ms` and `total_ms`. CUDA paths populate H2D,
kernel, D2H, and total timing when CUDA is requested.

Run the Phase 0 CPU self-test:

```bash
./scripts/run_node1_gpu_lab_phase0_selftest.sh
```

For CUDA validation on Node1:

```bash
./scripts/build_node1_gpu_lab.sh
compute-sanitizer --tool memcheck ./build/node1_non_llm_gpu_lab --mode synthetic --scenario sparse --gpu
compute-sanitizer --tool memcheck ./build/node1_non_llm_gpu_lab --mode synthetic --scenario mixed --gpu
compute-sanitizer --tool memcheck ./build/node1_non_llm_gpu_lab --mode synthetic --scenario dense --gpu
```

## Phase 1: CPU ISP rolling line-buffer filters

Phase 1 adds a CPU-only ISP memory-locality lab. It is intentionally separate
from camera semantics and does not emit object, person, identity, behavior, or
intent claims. It emits only deterministic image-processing metrics.

Supported filters:

- `blur` — 3x3 box blur with clamped borders
- `sharpen` — 3x3 cross sharpen kernel
- `edge` / `conv-edge` — 3x3 Laplacian-style edge response
- `sobel-x` — horizontal Sobel response
- `sobel-y` — vertical Sobel response
- `sobel-mag` — Sobel magnitude

The CPU implementation uses a true rolling 3-row line buffer:

```text
initial load: y-1, y, y+1
for each output row:
  compute with line0/line1/line2
  rotate line0 <- line1, line1 <- line2
  load only the new y+2 row into line2
```

This is the Phase 1 foundation used by the Phase 2 CUDA shared-memory tiled
ISP kernels.

### Synthetic ISP smoke

```bash
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-mag --width 64 --height 48
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter blur --width 64 --height 48 --include-output
```

### PGM/PPM file mode

The native binary supports binary P5 PGM and P6 PPM input. PPM is converted to
grayscale before filtering.

```bash
./build-cpu/node1_non_llm_gpu_lab \
  --mode isp-pgm \
  --input input.pgm \
  --output output.pgm \
  --isp-filter sobel-mag
```

If `--output` ends in `.ppm`, the grayscale output is expanded to RGB PPM;
otherwise output is written as P5 PGM.

### ISP output contract

The top-level JSON includes an `isp` object with the schema
`node1_non_llm_isp_filters.v0.1`. Example fields:

```json
{
  "backend": "cpu",
  "filter": "sobel-mag",
  "pixels_processed": 3072,
  "bytes_read": 3072,
  "bytes_written": 3072,
  "edge_energy": 146.97,
  "focus_score": 4279.25,
  "noise_score": 96.69,
  "lighting_delta": 3.64,
  "saturation_ratio": 0.22,
  "facts_only": true
}
```

### Phase 1 validation

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
./build-cpu/node1_non_llm_gpu_lab_selftest
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-mag --width 64 --height 48
```

The self-test validates rolling-buffer output against a naive 3x3 reference for
all filters, known-value kernels, PGM/PPM roundtrip I/O, and ISP JSON safety
fields.

## Phase 2: CUDA ISP shared-memory tiled filters

Phase 2 adds CUDA versions of the Phase 1 ISP filters. The CUDA path uses a
shared-memory 16x16 output tile plus a 1-pixel halo for 3x3 stencil access. The
same native command emits the CPU baseline in `isp` and the CUDA result in
`isp_cuda` when built with CUDA and invoked with `--gpu`.

```bash
./scripts/build_node1_gpu_lab.sh
./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-mag --width 64 --height 48 --gpu --include-output
```

Run the Phase 2 CUDA parity self-test:

```bash
./scripts/run_node1_gpu_lab_phase2_isp_cuda_selftest.sh
```

The self-test requires all filters to satisfy:

```text
isp_cpu_cuda_comparison.ok == true
output_equal == true
metrics_close == true
mismatch_count == 0
max_abs_diff == 0
```

The CUDA ISP path also emits edge/focus/noise/lighting/saturation metrics using
CUDA reduction counters. It remains facts-only workload metadata and does not
emit object, identity, behavior, or intent claims.

## Phase 3: Sparse ROI crop/resize/normalize

Phase 3 adds a facts-only sparse ROI path for active-tile workloads. The native lab walks the active bits in the frame tile mask, converts each active tile into a rectangular ROI, crops from the current grayscale frame, resizes each ROI to a fixed target size with deterministic nearest-neighbor sampling, and normalizes each output element to float32 `[0, 1]`.

The CPU path emits `sparse_roi`. A CUDA-built binary invoked with `--gpu` also emits `sparse_roi_cuda` and `sparse_roi_cpu_cuda_comparison`.

Example:

```bash
./build/node1_non_llm_gpu_lab \
  --mode sparse-roi-synthetic \
  --scenario sparse \
  --width 320 \
  --height 240 \
  --target-width 16 \
  --target-height 16 \
  --gpu \
  --include-output
```

Expected facts include `roi_count`, active tile rectangles, output element count, bytes read/written, normalized output range, timing, and CPU-vs-CUDA parity. The sparse ROI path does not emit object, identity, behavior, intent, weapon, or suspiciousness claims.

Validation:

```bash
./scripts/run_node1_gpu_lab_phase3_sparse_roi_selftest.sh
./scripts/build_node1_gpu_lab.sh
./scripts/run_node1_gpu_lab_phase3_sparse_roi_cuda_selftest.sh
compute-sanitizer --tool memcheck ./build/node1_non_llm_gpu_lab --mode sparse-roi-synthetic --scenario sparse --gpu --include-output
```

## Phase 4: Mixed region connected-component grouping

Phase 4 adds the mixed-region path for the non-LLM GPU lab. It takes the
existing frame tile mask, walks active tiles with 4-neighbor connected-component
logic, classifies the result as `contiguous` or `scattered`, and batches one
crop/resize/normalize operation per connected component.

Native CPU validation:

```bash
./scripts/run_node1_gpu_lab_phase4_mixed_region_selftest.sh
```

Native CUDA validation after `./scripts/build_node1_gpu_lab.sh`:

```bash
./scripts/run_node1_gpu_lab_phase4_mixed_region_cuda_selftest.sh
```

Manual example:

```bash
./build/node1_non_llm_gpu_lab \
  --mode mixed-region-synthetic \
  --scenario scattered \
  --width 320 \
  --height 240 \
  --target-width 16 \
  --target-height 16 \
  --gpu \
  --include-output
```

The JSON emits `mixed_region` for the CPU reference path, `mixed_region_cuda`
for the CUDA path, and `mixed_region_cpu_cuda_comparison` for parity facts. The
module remains facts-only workload metadata; it does not claim object identity,
behavior, intent, weapons, or suspicious activity.

## Phase 5: Dense full-frame diff/histogram/reduction/normalize

Phase 5 adds the dense full-frame path for high-activity frames. The path runs
absolute frame diff, changed-pixel reduction, a 256-bin diff histogram,
previous/current/diff mean reductions, lighting delta, and dense current-frame
normalization to float32 `[0, 1]`.

Native CPU validation:

```bash
./scripts/run_node1_gpu_lab_phase5_dense_full_frame_selftest.sh
```

Native CUDA validation after `./scripts/build_node1_gpu_lab.sh`:

```bash
./scripts/run_node1_gpu_lab_phase5_dense_full_frame_cuda_selftest.sh
```

Manual example:

```bash
./build/node1_non_llm_gpu_lab \
  --mode dense-full-frame-synthetic \
  --scenario dense \
  --width 320 \
  --height 240 \
  --gpu \
  --include-output
```

The JSON emits `dense_full_frame` for the CPU reference path,
`dense_full_frame_cuda` for the CUDA path, and
`dense_full_frame_cpu_cuda_comparison` for parity facts. The module remains
facts-only workload metadata; it does not claim object identity, behavior,
intent, weapons, or suspicious activity.

## Phase 6: Overlay-heavy alpha blend / heatmap / thumbnail path

Phase 6 adds the overlay-heavy path for visual artifact generation. The path
runs full-frame absolute diff, deterministic motion heatmap generation,
alpha-blended RGB overlay generation, thumbnail generation, and before/after
comparison facts.

Native CPU validation:

```bash
./scripts/run_node1_gpu_lab_phase6_overlay_heavy_selftest.sh
```

Native CUDA validation after `./scripts/build_node1_gpu_lab.sh`:

```bash
./scripts/run_node1_gpu_lab_phase6_overlay_heavy_cuda_selftest.sh
```

Manual example:

```bash
./build/node1_non_llm_gpu_lab \
  --mode overlay-heavy-synthetic \
  --scenario mixed \
  --width 320 \
  --height 240 \
  --thumbnail-width 64 \
  --thumbnail-height 48 \
  --gpu \
  --include-output
```

The JSON emits `overlay_heavy` for the CPU reference path,
`overlay_heavy_cuda` for the CUDA path, and
`overlay_heavy_cpu_cuda_comparison` for parity facts. The module remains
facts-only workload metadata; it does not claim object identity, behavior,
intent, weapons, or suspicious activity.

## Phase 7 — AudioBox path

The Phase 7 AudioBox path adds facts-only audio signal analysis:

```text
RMS -> peak -> silence mask -> onset mask -> cross-correlation sync drift
```

CPU validation:

```bash
./scripts/run_node1_gpu_lab_phase7_audiobox_selftest.sh
```

CUDA validation after `./scripts/build_node1_gpu_lab.sh`:

```bash
./scripts/run_node1_gpu_lab_phase7_audiobox_cuda_selftest.sh
```

Manual run:

```bash
./build/node1_non_llm_gpu_lab \
  --mode audiobox-synthetic \
  --audio-samples 32768 \
  --sample-rate 48000 \
  --audio-window-samples 1024 \
  --max-lag 128 \
  --sync-drift-samples 64 \
  --gpu \
  --include-output
```

## Phase 8 — Storage batch planner and clip sampler

Phase 8 adds a facts-only storage planning path for manifest-driven clip work. It does not read or decode media contents; it plans deterministic storage batches and key timeline moments from manifest metadata only.

Implemented workload pieces:

- manifest scan from synthetic metadata or a simple CSV manifest
- batch read plan constrained by max bytes and max clips per batch
- key moment selection by deterministic priority score with minimum timeline gap
- clip timeline features including total bytes, span, covered duration, gaps, and score summaries

Native synthetic validation:

```bash
./build-cpu/node1_non_llm_gpu_lab \
  --mode storage-batch-synthetic \
  --clips 12 \
  --max-batch-bytes 1600000 \
  --max-batch-clips 3 \
  --key-moments 4 \
  --min-key-gap-ms 1000 \
  --include-output
```

Manifest CSV format:

```text
clip_id,path,start_ms,duration_ms,bytes,motion_score,audio_score,lighting_delta,changed_pixels
```

The output emits `storage_batch` with manifest counts, batch plans, selected key moments, timeline features, bytes planned, timing, and `facts_only=true`. It does not emit visual, audio, identity, behavior, or intent claims.

## Phase 9 — Evidence pipeline expansion

Phase 9 adds the facts-only evidence pipeline:

```text
manifest scan / storage plan reuse
visual fingerprint workload vectors
evidence dedup groups
key-moment selector with dedup representatives
latency and throughput monitor
evidence safety validator
```

Run CPU validation:

```bash
./scripts/run_node1_gpu_lab_phase9_evidence_pipeline_selftest.sh
```

Run a native synthetic example:

```bash
./build-cpu/node1_non_llm_gpu_lab \
  --mode evidence-pipeline-synthetic \
  --clips 12 \
  --max-batch-bytes 1600000 \
  --max-batch-clips 3 \
  --key-moments 4 \
  --min-key-gap-ms 1000 \
  --dedup-hamming-threshold 0 \
  --fingerprint-cycle 6 \
  --include-output
```

The output is `evidence_pipeline` with `storage_batch`, `fingerprints`, `duplicate_groups`, `key_moments`, `timeline`, `latency`, and `safety`. It does not decode media and does not emit object, person, identity, speech content, behavior, intent, or suspiciousness claims.

## Phase 10 — Capture-run evidence pipeline integration

Phase 10 connects the native evidence pipeline to MonitorMe `capture-run` sessions through the Python bridge. The native module still runs in `evidence-pipeline-manifest` mode; the capture runner is responsible for converting local keyframe evidence into the CSV manifest contract.

Validation:

```bash
./scripts/run_node1_gpu_lab_phase10_capture_evidence_pipeline_selftest.sh
```

The selftest builds the CPU native binary, runs the native selftest binary, then runs the Python capture-run integration test. Expected result:

```text
node1_non_llm_gpu_lab Phase 10 Capture-run Evidence Pipeline selftest PASS
```

## Phase 11 — real media fingerprint ingestion

The native evidence pipeline can now consume manifest rows with precomputed decoded-keyframe fingerprints. The Python capture runner writes these columns after locally decoding stored JPEG keyframes:

```text
fingerprint_source,decoded_width,decoded_height,ahash64,dhash64,fingerprint64,histogram16
```

Rows with `fingerprint_source=decoded_keyframe`, positive decoded dimensions, and a 16-bin histogram are treated as real media fingerprints. Native JSON reports `real_media_ingestion`, `media_fingerprint_count`, `synthetic_fingerprint_count`, and per-fingerprint `from_media` facts. Rows without valid media fingerprint columns continue to use deterministic metadata-synthetic fingerprints.

Selftest:

```bash
./scripts/run_node1_gpu_lab_phase11_real_media_fingerprint_selftest.sh
```

### Phase 12: evidence index persistence / DB migration

Phase 12 persists the capture-run evidence pipeline profile into normalized SQLite tables instead of relying only on the large JSON artifact and `events.attrs_json`. The migration `005_evidence_index_persistence.sql` creates queryable rows for evidence profiles, per-keyframe fingerprints, duplicate groups, and key moments.

After `--evidence-pipeline-enabled`, capture-run still writes:

```text
capture_evidence_manifest.csv
evidence_pipeline_profile.json
evidence_pipeline_indexed event
```

It now also persists:

```text
evidence_pipeline_profiles
evidence_fingerprints
evidence_dedup_groups
evidence_key_moments
```

Read back persisted evidence index rows:

```bash
python -m monitor_me.cli evidence-index --session-id <session_id>
python -m monitor_me.cli evidence-fingerprints --session-id <session_id> --from-media --limit 20
python -m monitor_me.cli evidence-dedup-groups --session-id <session_id>
python -m monitor_me.cli evidence-key-moments --session-id <session_id>
```

The persisted index remains facts-only. It stores storage paths, hashes, fingerprint integers, histogram bins, duplicate accounting, key-moment metadata, safety validator output, and latency/throughput counters. It does not store identity, behavior, intent, suspiciousness, object claims, or speech-content claims.

Validate Phase 12:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase12_evidence_index_selftest.sh
python -m pytest -q tests/test_node1_evidence_index_phase12.py
```

### Phase 13 — Evidence summary API exposure

Phase 13 exposes the persisted evidence index through MonitorMe FastAPI routes. It is a Python/API layer on top of the Phase 12 SQLite evidence index and does not add a new CUDA workload.

Validation:

```bash
./scripts/run_node1_gpu_lab_phase13_evidence_api_selftest.sh
```

The API routes return facts-only evidence summaries, fingerprints, duplicate groups, and key moments. They do not decode media in the API request path and do not emit semantic identity, speech, behavior, intent, weapon, danger, or suspiciousness claims.

### Phase 14 — Evidence index retention / compaction policies

Phase 14 is a MonitorMe DB/API policy layer on top of the Phase 12 evidence index. It does not add a CUDA kernel. It adds deterministic retention planning, dry-run/apply execution, WAL checkpoint compaction, optional SQLite `VACUUM`, and retention run history.

The policy deletes only normalized evidence index rows and keeps the original capture/event/artifact evidence intact. This means the lightweight query index can be compacted while preserving the ability to rebuild it from retained evidence artifacts later.

Validate:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase14_evidence_retention_selftest.sh
```

Expected:

```text
node1_non_llm_gpu_lab_selftest PASS
node1_non_llm_gpu_lab Phase 14 Evidence Index Retention selftest PASS
```

---

## Phase 15 — Operator dashboard / UI integration

Phase 15 adds a local operator dashboard over the persisted evidence pipeline index.

```text
GET /operator/dashboard       -> read-only HTML UI
GET /operator/dashboard/data  -> JSON model used by the UI/tests
```

The dashboard is local-only and facts-only. It reads SQLite evidence-index rows and retention-run audit rows. It does not decode media, upload frames, fetch external web assets, run VLM/LLM analysis, infer identity/intent, or execute destructive retention actions.

Selftest:

```bash
./scripts/run_node1_gpu_lab_phase15_operator_dashboard_selftest.sh
```
