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
