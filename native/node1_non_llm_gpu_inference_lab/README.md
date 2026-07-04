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
