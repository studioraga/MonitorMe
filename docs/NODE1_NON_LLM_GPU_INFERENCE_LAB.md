# Node1 Non-LLM GPU Inference Lab v0.1

## Purpose

This module starts the MonitorMe path for C++/CUDA CPU/GPU optimization without using an LLM as the inference core.

The design follows the Node1 tutorial rule:

```text
Do not send every frame through one path.
Measure activity.
Build masks.
Classify workload shape.
Route sparse, mixed, and dense cases differently.
```

The tutorial frames this as the non-LLM equivalent of multiple specialized processing heads: motion tile occupancy, object/ROI occupancy, overlay/pixel-change region, audio transient/energy window, storage/batch-read planner, and latency/throughput monitor. The attached tutorial specifically defines the pipeline as `CPU Workload Classifier -> Sparse Path / Mixed Path / Dense Path -> GPU Feature / Inference Kernels -> CPU Event Builder / Evidence Index`. fileciteturn0file0L32-L58

## What was added

```text
native/node1_non_llm_gpu_inference_lab/
  CMakeLists.txt
  include/node1_non_llm/gpu_lab.hpp
  src/gpu_lab_cpu.cpp
  src/gpu_lab_cuda.cu
  src/gpu_lab_main.cpp
  scripts/build_node1_gpu_lab.sh
  scripts/run_node1_gpu_lab_smoke.sh

monitor_me/non_llm_gpu_lab.py
  Safe Python bridge around the native C++/CUDA sidecar.

monitor_me/local_capture.py
  Optional after-trigger integration.
  Emits gpu_workload_profiled child events only when --gpu-lab-enabled is used.
```

## Runtime architecture

```text
C922 /dev/video0
  -> OpenCV bounded capture
  -> deterministic frame-difference motion gate
  -> keyframe artifact
  -> motion_detected parent event
  -> optional YOLO object_detected children
  -> optional node1_non_llm_gpu_lab profile
       previous frame + current frame
         -> grayscale raw pair
         -> C++ CPU tile-mask path
         -> optional CUDA tile-mask kernel
         -> sparse/mixed/dense route decision
         -> gpu_workload_profiled child event
  -> deterministic summaries / VLM / SmolVLM2 experiments
  -> manifest + SQLite evidence rows
```

## Evidence contract

The GPU lab does not claim object identity, behavior, intent, weapon status, or person identity.

It stores workload facts only:

```text
event_type = gpu_workload_profiled
label      = sparse | mixed | dense | unavailable
model_id   = node1-non-llm-gpu-lab-v0.1
attrs:
  tile_mask_hex
  active_tiles
  low_half_active_tiles
  high_half_active_tiles
  changed_pixels
  changed_ratio
  routing_decision
  native_binary_available
  cuda_compiled
  frame_cuda
  privacy.external_upload = false
  privacy.identity = false
  privacy.intent = false
```

## Build the native module on Node1

```bash
cd MonitorMe/native/node1_non_llm_gpu_inference_lab
./scripts/build_node1_gpu_lab.sh
```

Or manually:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNODE1_NON_LLM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j"$(nproc)"
```

## CPU-only build for development

```bash
cd MonitorMe/native/node1_non_llm_gpu_inference_lab
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
./build-cpu/node1_non_llm_gpu_lab --mode synthetic --scenario mixed
```

## Validate synthetic sparse/mixed/dense behavior

```bash
export MONITORME_GPU_LAB_BIN=$PWD/build/node1_non_llm_gpu_lab
python -m monitor_me.cli gpu-lab-health --probe --allow-unavailable
python -m monitor_me.cli gpu-lab-synthetic --scenario sparse
python -m monitor_me.cli gpu-lab-synthetic --scenario mixed
python -m monitor_me.cli gpu-lab-synthetic --scenario dense
```

## Enable during a real MonitorMe capture

```bash
python -m monitor_me.cli --db data/events/monitorme.db capture-run \
  --camera-id c922_node1_gate \
  --device /dev/video0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --fourcc MJPG \
  --duration-sec 10 \
  --motion-threshold 1.5 \
  --gpu-lab-enabled \
  --gpu-lab-binary native/node1_non_llm_gpu_inference_lab/build/node1_non_llm_gpu_lab
```

List GPU workload profile events:

```bash
python -m monitor_me.cli --db data/events/monitorme.db events \
  --event-type gpu_workload_profiled \
  --limit 20
```

## Profiler plan

Use Nsight Compute against the native binary first:

```bash
ncu --set full --target-processes all \
  native/node1_non_llm_gpu_inference_lab/build/node1_non_llm_gpu_lab \
  --mode synthetic --scenario dense --gpu
```

Measure:

```text
DRAM throughput
L1/TEX and L2 hit rate
SM occupancy
warp divergence
atomic overhead for tile mask aggregation
copy time versus kernel time
```

## Next implementation slices

1. Replace temporary raw-gray bridge files with pinned host memory and a long-lived native worker.
2. Add ROI crop/resize kernels for sparse path.
3. Add grouped region kernels for mixed path.
4. Add full-frame coalesced dense kernels for overlay/motion transforms.
5. Add batched AudioBox float32 window analysis and optional CUDA STFT later.
6. Add Prometheus counters for path distribution and CUDA timing.

## Phase 0: v0.1 hardening for future CPU/GPU modules

Phase 0 is intentionally a refactor and test-hardening step. It does not add new
ISP, ROI, dense-frame, mixed-region, overlay, audio-sync, storage, or visual
fingerprint kernels yet. It prepares the native lab for those phases by making
core contracts reusable.

### Phase 0 changes

```text
refactor shared structs      -> gpu_lab_types.hpp / gpu_lab_types.cpp
split JSON helper            -> gpu_lab_json.hpp / gpu_lab_json.cpp
split CUDA error helper      -> gpu_lab_cuda_utils.cuh / gpu_lab_cuda_utils.cu
add timing struct            -> gpu_lab_timing.hpp / gpu_lab_timing.cpp
add stricter tests           -> gpu_lab_selftest.cpp and pytest bridge checks
```

### Why this matters

The next modules from the overview context all need the same foundation:

```text
Scenario 1 sparse camera activity       -> ROI route/crop/resize/normalize
Scenario 2 dense camera activity        -> coalesced full-frame transforms/reductions
Scenario 3 mixed camera activity        -> connected tile groups/grouped ROI batches
Scenario 4 overlay-heavy processing     -> alpha blend/heatmap/thumbnail paths
Scenario 5 AudioBox soundtrack          -> RMS/onset/silence/sync-drift paths
ISP Filters implementation              -> rolling CPU line buffer and CUDA tiled 3x3 filters
```

Each of those phases needs shared route structs, validation rules, JSON output,
CUDA error handling, and timing facts. Phase 0 provides those shared pieces first.

### Evidence safety contract

The module remains facts-only. Phase 0 does not introduce any semantic labels. It
still emits workload routing metadata only:

```text
sparse
mixed
dense
unavailable
```

It must not emit identity, intent, behavior, person, vehicle, weapon, or other
semantic claims.

### Validation commands

```bash
cd native/node1_non_llm_gpu_inference_lab
./scripts/run_node1_gpu_lab_phase0_selftest.sh
./scripts/build_node1_gpu_lab.sh
./build/node1_non_llm_gpu_lab --mode synthetic --scenario sparse --gpu
./build/node1_non_llm_gpu_lab --mode synthetic --scenario mixed --gpu
./build/node1_non_llm_gpu_lab --mode synthetic --scenario dense --gpu
```

Then from repo root:

```bash
python -m compileall -q monitor_me tests
python -m pytest -q tests -k "gpu or capture or cli or model"
python -m monitor_me.cli gpu-lab-health --enabled --probe
python -m monitor_me.cli gpu-lab-synthetic --scenario sparse
python -m monitor_me.cli gpu-lab-synthetic --scenario mixed
python -m monitor_me.cli gpu-lab-synthetic --scenario dense
```

## Phase 1 — CPU ISP rolling line-buffer filters

Phase 1 extends the native lab with CPU-only ISP filters. The scope is a
self-contained memory-locality lab and does not depend on camera semantics,
YOLO labels, VLM summaries, or external services.

### Implemented CPU ISP filters

- `blur`: 3x3 box blur
- `sharpen`: 3x3 cross sharpen
- `edge` / `conv-edge`: 3x3 Laplacian-style edge response
- `sobel-x`: horizontal Sobel response
- `sobel-y`: vertical Sobel response
- `sobel-mag`: Sobel magnitude

### Rolling 3-row buffer contract

The implementation uses a true rolling line buffer. It does not reload
`y-1`, `y`, and `y+1` from scratch for every output row. Instead it initializes
three rows, computes one output row, rotates row ownership, and loads only the
new bottom row:

```text
line0 = clamp(y - 1)
line1 = y
line2 = clamp(y + 1)

for y in output rows:
  compute 3x3 window from line0/line1/line2
  line0 <- line1
  line1 <- line2
  line2 <- clamp(y + 2)
```

This gives the CPU baseline needed before the later Phase 2 CUDA shared-memory
tiled implementation.

### PGM/PPM I/O

The native binary now supports binary P5 PGM and P6 PPM input for ISP testing.
PPM input is converted to grayscale before filtering. Output can be written as
PGM or as grayscale-expanded PPM.

```bash
native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab \
  --mode isp-pgm \
  --input input.pgm \
  --output output.pgm \
  --isp-filter sobel-mag
```

Synthetic mode is useful for reproducible tests:

```bash
native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab \
  --mode isp-synthetic \
  --isp-filter sobel-mag \
  --width 64 \
  --height 48
```

### Facts-only evidence safety

The ISP output is metrics-only. It does not emit object labels, person labels,
identity, intent, behavior, weapon, or suspiciousness claims. Its schema is:

```text
node1_non_llm_isp_filters.v0.1
```

Key fields:

```text
filter
pixels_processed
bytes_read
bytes_written
output_min / output_max / output_mean
edge_energy
focus_score
noise_score
lighting_delta
saturation_pixels / saturation_ratio
timing
facts_only=true
```

### Phase 1 validation plan

```bash
cd native/node1_non_llm_gpu_inference_lab
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j"$(nproc)"
./build-cpu/node1_non_llm_gpu_lab_selftest
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter blur --width 64 --height 48
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sharpen --width 64 --height 48
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter edge --width 64 --height 48
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-x --width 64 --height 48
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-y --width 64 --height 48
./build-cpu/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-mag --width 64 --height 48
```

Python/CLI validation:

```bash
python -m pytest -q tests/test_node1_isp_filters_phase1.py
MONITORME_GPU_LAB_BIN=native/node1_non_llm_gpu_inference_lab/build-cpu/node1_non_llm_gpu_lab \
  python -m monitor_me.cli gpu-lab-isp-synthetic --filter sobel-mag --width 64 --height 48
```

### Out of scope for Phase 1

Phase 1 intentionally did not add CUDA ISP kernels. Phase 2, documented below, adds
shared-memory tiled CUDA kernels and compares CPU rolling-buffer output against
CUDA output.

## Phase 2 — CUDA ISP filters and metrics

Phase 2 implements the CUDA counterpart to the Phase 1 CPU ISP rolling-buffer
baseline. It does not add semantic vision, object detection, identity, intent,
or behavior claims. It only emits deterministic image-processing workload facts.

### Implemented CUDA ISP filters

The CUDA path supports the same filter names as Phase 1:

```text
blur
sharpen
edge / conv-edge
sobel-x
sobel-y
sobel-mag
```

### CUDA design

The filter kernel uses a 16x16 output tile and a shared-memory tile with a
1-pixel halo on all sides:

```text
global grayscale input
  -> shared tile [16 + 2][16 + 2]
  -> clamped border halo loads
  -> 3x3 stencil per output pixel
  -> global grayscale output
```

This is the shared-memory tiled 3x3 filter baseline requested in the TASK1 plan.
It prepares the later Nsight Compute profiling work for shared-memory bank
conflicts, global memory coalescing, and kernel timing.

### CUDA metrics

A second CUDA reduction kernel computes facts-only ISP metrics from the input and
filtered output:

```text
edge_energy       = output mean
focus_score       = output variance
noise_score       = mean absolute input/output difference
lighting_delta    = absolute input/output mean delta
saturation_pixels = count(output == 0 or output == 255)
saturation_ratio  = saturation_pixels / pixels_processed
```

### JSON contract

CPU output remains under `isp`. CUDA output is emitted under `isp_cuda` only when
the binary is compiled with CUDA and the native CLI receives `--gpu`.

```json
{
  "isp": {"backend": "cpu", "schema": "node1_non_llm_isp_filters.v0.1"},
  "isp_cuda": {"backend": "cuda", "schema": "node1_non_llm_isp_filters.v0.1"},
  "isp_cpu_cuda_comparison": {
    "schema": "node1_non_llm_isp_cpu_cuda_compare.v0.1",
    "ok": true,
    "output_equal": true,
    "metrics_close": true,
    "mismatch_count": 0,
    "max_abs_diff": 0,
    "facts_only": true
  }
}
```

### Validation plan

```bash
cd native/node1_non_llm_gpu_inference_lab
./scripts/build_node1_gpu_lab.sh
./scripts/run_node1_gpu_lab_phase2_isp_cuda_selftest.sh
```

Manual checks:

```bash
./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter blur --width 64 --height 48 --gpu --include-output
./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sharpen --width 64 --height 48 --gpu --include-output
./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter edge --width 64 --height 48 --gpu --include-output
./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-x --width 64 --height 48 --gpu --include-output
./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-y --width 64 --height 48 --gpu --include-output
./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-mag --width 64 --height 48 --gpu --include-output
```

Compute Sanitizer:

```bash
compute-sanitizer --tool memcheck ./build/node1_non_llm_gpu_lab --mode isp-synthetic --isp-filter sobel-mag --width 64 --height 48 --gpu --include-output
```

Python validation:

```bash
python -m pytest -q tests/test_node1_isp_filters_phase2.py
python -m pytest -q tests -k "gpu or isp or capture or cli or model"
```

### Phase 2 out of scope

Phase 2 does not add demosaic RGGB, sparse ROI crop/resize/normalize, mixed
region grouping, dense full-frame reductions, overlay-heavy processing,
AudioBox signal processing, storage batch planning, or visual fingerprinting.
Those remain later phases from the TASK1 module checklist.

## Phase 3: Sparse ROI crop/resize/normalize

Phase 3 implements the sparse ROI path from the CPU/GPU workload-optimization roadmap. It is designed for sparse camera activity where only a few motion tiles are active and full-frame GPU processing would waste memory bandwidth.

Pipeline:

```text
previous/current gray frame
  -> frame tile-mask analysis
  -> active tile walking
  -> ROI rectangle list
  -> crop from current gray frame
  -> deterministic nearest-neighbor resize
  -> uint8-to-float32 normalize
  -> CPU-vs-CUDA comparison facts
```

Native mode:

```bash
node1_non_llm_gpu_lab \
  --mode sparse-roi-synthetic \
  --scenario sparse \
  --target-width 16 \
  --target-height 16 \
  --gpu \
  --include-output
```

JSON fields:

```text
sparse_roi                         CPU reference ROI result
sparse_roi_cuda                    CUDA ROI result when --gpu is used
sparse_roi_cpu_cuda_comparison     rois_equal/output_close/metrics_close facts
```

The ROI path is facts-only. It reports tile-derived rectangles, crop/resize/normalize counts, normalized value statistics, bytes read/written, and timing. It does not report object class, person, identity, behavior, intent, weapon, or suspiciousness claims.

Phase 3 validation scripts:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase3_sparse_roi_selftest.sh
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase3_sparse_roi_cuda_selftest.sh
```

The CUDA selftest validates sparse, mixed, and dense synthetic masks to ensure active tile walking, ROI rectangle generation, normalized output, and CPU-vs-CUDA parity remain deterministic across route shapes.

## Phase 4 — Mixed region path

Phase 4 implements the mixed-region processing path after tile-mask routing.
The module converts active tiles into 4-neighbor connected components, classifies
the tile layout as `contiguous` or `scattered`, and performs grouped crop
batching by resizing and normalizing each connected-component bounding box.

Outputs:

- `mixed_region`: CPU reference connected components and grouped normalized crops.
- `mixed_region_cuda`: CUDA grouped crop/resize/normalize path when `--gpu` is used.
- `mixed_region_cpu_cuda_comparison`: CPU-vs-CUDA parity facts, including
  `groups_equal`, `output_close`, `mismatch_count`, `max_abs_diff`, and
  `metrics_close`.

Supported synthetic scenarios for Phase 4 validation:

- `contiguous`: one rectangular component on the mixed route.
- `scattered`: checkerboard active tiles, many separated components on the mixed route.
- `dense`: all tiles active, one large contiguous component.

The module is intentionally non-semantic. It emits only workload, grouping,
crop-batching, and normalization facts.

## Phase 5 — Dense full-frame path

Phase 5 implements the dense full-frame processing path from the TASK1 roadmap.
It is intended for high-activity frames where most tiles are active and ROI-style
sparse processing is no longer the right memory-access pattern.

Pipeline:

```text
previous/current gray frame
  -> full-frame absolute diff
  -> changed-pixel reduction using pixel_threshold
  -> 256-bin diff histogram
  -> previous/current/diff reductions
  -> lighting delta
  -> dense current-frame normalize to float32 [0, 1]
  -> CPU-vs-CUDA parity facts
```

Native mode:

```bash
node1_non_llm_gpu_lab \
  --mode dense-full-frame-synthetic \
  --scenario dense \
  --width 320 \
  --height 240 \
  --gpu \
  --include-output
```

JSON fields:

```text
dense_full_frame                         CPU reference dense-frame result
dense_full_frame_cuda                    CUDA dense-frame result when --gpu is used
dense_full_frame_cpu_cuda_comparison     histogram/reduction/normalization parity facts
```

The comparison reports `histogram_equal`, `normalized_close`,
`mismatch_count`, `max_abs_diff`, `reductions_close`, changed-pixel equality,
and per-reduction absolute differences.

Validation scripts:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase5_dense_full_frame_selftest.sh
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase5_dense_full_frame_cuda_selftest.sh
```

Supported synthetic scenarios:

- `dense`: all pixels changed, dense route, one dominant diff histogram bin.
- `mixed`: middle-region activity, mixed route, dense full-frame reductions still valid.
- `sparse`: sparse activity, sparse route, dense full-frame reductions still valid.

This module is intentionally non-semantic. It emits only workload, histogram,
reduction, lighting, normalization, and timing facts. It does not report object
class, person, identity, behavior, intent, weapon, or suspiciousness claims.

## Phase 6 — Overlay-heavy path

Phase 6 implements the overlay-heavy processing path from the TASK1 roadmap. It
is intended for workloads where the expensive step is not classification but
visual artifact generation: motion heatmaps, alpha-blended overlays,
thumbnails, and before/after comparison facts.

Pipeline:

```text
previous/current gray frame
  -> full-frame absolute diff
  -> motion heatmap uint8 image
  -> alpha blend current frame with deterministic heat color
  -> RGB overlay artifact buffer
  -> thumbnail generation from overlay RGB
  -> before/after comparison reductions
  -> CPU-vs-CUDA parity facts
```

Native mode:

```bash
node1_non_llm_gpu_lab \
  --mode overlay-heavy-synthetic \
  --scenario mixed \
  --width 320 \
  --height 240 \
  --thumbnail-width 64 \
  --thumbnail-height 48 \
  --gpu \
  --include-output
```

JSON fields:

```text
overlay_heavy                         CPU reference overlay-heavy result
overlay_heavy_cuda                    CUDA overlay-heavy result when --gpu is used
overlay_heavy_cpu_cuda_comparison     heatmap/overlay/thumbnail parity facts
```

The comparison reports `heatmap_equal`, `overlay_equal`, `thumbnail_equal`,
`mismatch_count`, `max_abs_diff`, `metrics_close`, changed-pixel equality,
heatmap/overlay/thumbnail mean differences, before/after diff agreement, and
lighting-delta agreement.

Validation scripts:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase6_overlay_heavy_selftest.sh
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase6_overlay_heavy_cuda_selftest.sh
```

Supported synthetic scenarios:

- `mixed`: middle-region activity and mixed route.
- `dense`: all pixels changed and dense route.
- `sparse`: sparse activity and sparse route.

This module is intentionally non-semantic. It emits only overlay workload,
heatmap, thumbnail, before/after comparison, and timing facts. It does not
report object class, person, identity, behavior, intent, weapon, or
suspiciousness claims.

## Phase 7 — AudioBox path

Phase 7 implements the AudioBox processing path from the TASK1 roadmap. It is a facts-only audio signal workload that measures numeric signal properties without transcribing speech or making identity, behavior, or intent claims.

Implemented operations:

```text
primary/reference float32 audio
  -> per-window RMS
  -> per-window peak
  -> silence mask
  -> onset mask
  -> bounded cross-correlation
  -> sync drift in samples and milliseconds
```

Native mode:

```bash
node1_non_llm_gpu_lab \
  --mode audiobox-synthetic \
  --audio-samples 32768 \
  --sample-rate 48000 \
  --audio-window-samples 1024 \
  --audio-max-windows 32 \
  --silence-threshold 0.02 \
  --onset-threshold 0.08 \
  --max-lag 128 \
  --sync-drift-samples 64 \
  --gpu \
  --include-output
```

JSON fields:

```text
audiobox                         CPU reference AudioBox result
audiobox_cuda                    CUDA AudioBox result when --gpu is used
audiobox_cpu_cuda_comparison     CPU-vs-CUDA parity facts
```

The comparison object validates RMS, peak, silence/onset masks, correlation scores, drift samples, drift milliseconds, and aggregate metrics. The selftest scripts are:

```text
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase7_audiobox_selftest.sh
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase7_audiobox_cuda_selftest.sh
```

## Phase 8 — Storage batch planner and clip sampler

Phase 8 implements the storage batch planning path from the TASK1 roadmap. It is a facts-only metadata workload for preparing clip reads and selecting deterministic key moments without decoding media payloads.

Implemented pieces:

```text
manifest scan
batch read plan
key moment selection
clip timeline features
```

Native synthetic run:

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

Manifest-backed run:

```bash
./build-cpu/node1_non_llm_gpu_lab \
  --mode storage-batch-manifest \
  --manifest clips.csv \
  --max-batch-bytes 1600000 \
  --max-batch-clips 3 \
  --key-moments 4
```

CSV manifest fields:

```text
clip_id,path,start_ms,duration_ms,bytes,motion_score,audio_score,lighting_delta,changed_pixels
```

Output contract:

```text
storage_batch                     CPU storage planner result
storage_batch.batches             deterministic batch read plan
storage_batch.key_moments         selected key moments with reason strings
storage_batch.timeline            clip timeline features and byte totals
storage_batch.facts_only          true
```

Safety boundary: this path reports storage and timeline metadata only. It does not emit object, person, identity, speech content, behavior, intent, weapon, suspiciousness, or semantic audio/visual claims.

## Phase 9 — Evidence pipeline expansion

Phase 9 implements the next roadmap block after storage planning:

```text
visual fingerprint / evidence dedup
clip sampler / key-moment selector
storage batch-read planner reuse
latency / throughput monitor
evidence safety validator
native / Python bridge expansion
CLI and docs expansion
```

The module is intentionally facts-only. It generates deterministic visual-fingerprint workload vectors from clip manifest/timeline metadata and does not decode media. This keeps the lab suitable for benchmarking evidence-index primitives without making semantic visual or audio claims.

Native modes:

```bash
./build-cpu/node1_non_llm_gpu_lab \
  --mode evidence-pipeline-synthetic \
  --clips 12 \
  --max-batch-bytes 1600000 \
  --max-batch-clips 3 \
  --key-moments 4 \
  --min-key-gap-ms 1000 \
  --dedup-hamming-threshold 0 \
  --fingerprint-width 16 \
  --fingerprint-height 16 \
  --fingerprint-cycle 6 \
  --include-output
```

```bash
./build-cpu/node1_non_llm_gpu_lab \
  --mode evidence-pipeline-manifest \
  --manifest clips.csv \
  --max-batch-bytes 1600000 \
  --max-batch-clips 3 \
  --key-moments 4 \
  --min-key-gap-ms 1000 \
  --dedup-hamming-threshold 0 \
  --include-output
```

JSON fields:

```text
evidence_pipeline.storage_batch       reused storage planner facts
evidence_pipeline.fingerprints         ahash/dhash/fingerprint/histogram facts
evidence_pipeline.duplicate_groups     dedup groups and representative clips
evidence_pipeline.key_moments          dedup-aware key-moment selector output
evidence_pipeline.timeline             clip timeline features
evidence_pipeline.latency              per-stage latency and throughput counters
evidence_pipeline.safety               evidence safety validator result
```

Validation:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase9_evidence_pipeline_selftest.sh
python -m pytest -q tests/test_node1_evidence_pipeline_phase9.py
```

The safety validator checks manifest count consistency, batch constraints, fingerprint shape, dedup group accounting, key-moment spacing/count limits, and timeline byte/count agreement. It emits violations as strings and keeps `facts_only=true`.

## Phase 10 — Capture-run evidence pipeline integration

Phase 10 wires the Phase 9 evidence pipeline into the real MonitorMe `capture-run` path. The integration runs after local motion keyframes have been stored and before the final capture manifest is written.

Flow:

```text
capture-run motion keyframes
  -> capture manifest frame records
  -> facts-only CSV evidence manifest
  -> native evidence-pipeline-manifest mode
  -> evidence_pipeline_profile JSON artifact
  -> evidence_pipeline_indexed DB event
  -> final capture manifest references event/artifacts
```

The bridge converts each stored keyframe record into the native storage/evidence manifest contract:

```text
clip_id,path,start_ms,duration_ms,bytes,motion_score,audio_score,lighting_delta,changed_pixels
```

The integration stores two capture artifacts:

- `evidence_pipeline_manifest_csv`
- `evidence_pipeline_profile`

It also inserts one session-level event:

```text
event_type=evidence_pipeline_indexed
label=facts_only_evidence_pipeline
model_id=node1-non-llm-evidence-pipeline-v0.1
```

The event attributes include native evidence counts, duplicate group counts, key-moment counts, planned read bytes, latency counters, safety validator output, artifact references, and privacy metadata.

Safety boundary:

- no media upload
- no identity inference
- no intent inference
- no semantic visual/audio claim
- no person/object claim from the evidence pipeline
- no media decoding inside the evidence pipeline path

Validation entry point:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase10_capture_evidence_pipeline_selftest.sh
```

## Phase 11 — real media fingerprint ingestion after decoded keyframes

Phase 11 explicitly routes decoded capture-run keyframes into the facts-only evidence pipeline. The Python capture runner performs the decode after the keyframe JPEG has already been stored as local evidence. It computes an average hash, difference hash, combined fingerprint64, decoded dimensions, and a 16-bin luminance histogram. These are written into the evidence CSV manifest and consumed by the native `evidence-pipeline-manifest` mode.

Pipeline view:

```text
stored keyframe JPEG
  -> local grayscale decode
  -> resize to fingerprint_width x fingerprint_height
  -> ahash64 + dhash64 + fingerprint64 + histogram16
  -> evidence_pipeline/capture_evidence_manifest.csv
  -> native evidence-pipeline-manifest
  -> evidence_pipeline_profile.json
  -> evidence_pipeline_indexed event
```

New CSV fields:

```text
fingerprint_source
  decoded_keyframe | metadata_synthetic | decode_unavailable

decoded_width, decoded_height
ahash64, dhash64, fingerprint64
histogram16     # 16 unsigned integer bins separated by |
```

Native evidence JSON now includes:

```text
real_media_ingestion
media_fingerprint_count
synthetic_fingerprint_count
fingerprints[].from_media
fingerprints[].fingerprint_source
fingerprints[].decoded_width
fingerprints[].decoded_height
```

The safety validator treats decoded-keyframe fingerprints as facts-only workload metadata. It checks that media fingerprints have a valid source, dimensions, hex hash, and 16-bin histogram. No object/person/identity/behavior/intent/suspiciousness claim is emitted.

CLI control:

```bash
--evidence-pipeline-enabled
--evidence-pipeline-no-real-fingerprints   # fallback to manifest-metadata synthetic fingerprints
```

Validation:

```bash
native/node1_non_llm_gpu_inference_lab/scripts/run_node1_gpu_lab_phase11_real_media_fingerprint_selftest.sh
python -m pytest -q tests/test_node1_capture_real_fingerprint_phase11.py
```
