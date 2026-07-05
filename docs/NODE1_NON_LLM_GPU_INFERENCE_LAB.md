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
