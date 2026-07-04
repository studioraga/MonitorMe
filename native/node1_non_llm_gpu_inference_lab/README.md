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

This is the Phase 1 foundation for the later Phase 2 CUDA shared-memory tiled
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
