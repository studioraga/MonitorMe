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
