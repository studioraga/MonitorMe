#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$LAB_DIR/build}"
CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-/usr/local/cuda-13.3/bin/nvcc}"
CUDA_ARCH="${CMAKE_CUDA_ARCHITECTURES:-120}"

cmake -S "$LAB_DIR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DNODE1_NON_LLM_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build "$BUILD_DIR" -j"$(nproc)"
"$BUILD_DIR/node1_non_llm_gpu_lab" --mode synthetic --scenario sparse --gpu
