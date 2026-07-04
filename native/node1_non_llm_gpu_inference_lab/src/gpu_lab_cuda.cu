#include "node1_non_llm/gpu_lab.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {
namespace {

static std::string cuda_error(cudaError_t err, const char* where) {
    std::string out(where);
    out += ": ";
    out += cudaGetErrorString(err);
    return out;
}

__global__ void frame_diff_tile_mask_kernel(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    int width,
    int height,
    int tile_cols,
    int tile_rows,
    int pixel_threshold,
    std::uint32_t* tile_counts,
    std::uint32_t* tile_mask,
    unsigned long long* changed_pixels) {

    const int total = width * height;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }

    const int diff = abs(static_cast<int>(current_gray[idx]) - static_cast<int>(previous_gray[idx]));
    if (diff <= pixel_threshold) {
        return;
    }

    const int x = idx % width;
    const int y = idx / width;
    const int tile_x = min((x * tile_cols) / width, tile_cols - 1);
    const int tile_y = min((y * tile_rows) / height, tile_rows - 1);
    const int tile = tile_y * tile_cols + tile_x;

    atomicAdd(&tile_counts[tile], 1U);
    atomicOr(tile_mask, (1U << tile));
    atomicAdd(changed_pixels, 1ULL);
}

__global__ void audio_energy_kernel(
    const float* samples,
    int sample_count,
    int window_samples,
    float threshold,
    int windows,
    float* rms,
    std::uint32_t* event_mask) {

    const int w = blockIdx.x;
    if (w >= windows) {
        return;
    }

    extern __shared__ float scratch[];
    const int start = w * window_samples;
    const int end = min(sample_count, start + window_samples);

    float local = 0.0f;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const float v = samples[i];
        local += v * v;
    }
    scratch[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int n = max(1, end - start);
        const float value = sqrtf(scratch[0] / static_cast<float>(n));
        rms[w] = value;
        if (value >= threshold) {
            atomicOr(event_mask, (1U << w));
        }
    }
}

static WorkloadPath choose_path(int active_tiles, int sparse_threshold, int dense_threshold) noexcept {
    if (active_tiles <= sparse_threshold) {
        return WorkloadPath::Sparse;
    }
    if (active_tiles >= dense_threshold) {
        return WorkloadPath::Dense;
    }
    return WorkloadPath::Mixed;
}

} // namespace

FrameAnalysis analyze_gray_frames_cuda(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const TileAnalysisConfig& cfg) {

    FrameAnalysis out;
    out.backend = "cuda";
    out.width = cfg.width;
    out.height = cfg.height;
    out.tile_cols = cfg.tile_cols;
    out.tile_rows = cfg.tile_rows;
    out.pixel_threshold = cfg.pixel_threshold;
    out.sparse_threshold = cfg.sparse_threshold;
    out.dense_threshold = cfg.dense_threshold;

    std::string error;
    if (!validate_tile_config(cfg, error)) {
        out.error = error;
        return out;
    }
    if (previous_gray == nullptr || current_gray == nullptr) {
        out.error = "previous_gray and current_gray must not be null";
        return out;
    }

    const int total_pixels = cfg.width * cfg.height;
    const std::size_t frame_bytes = static_cast<std::size_t>(total_pixels);
    const int tile_count = cfg.tile_cols * cfg.tile_rows;

    std::uint8_t* d_prev = nullptr;
    std::uint8_t* d_curr = nullptr;
    std::uint32_t* d_tile_counts = nullptr;
    std::uint32_t* d_mask = nullptr;
    unsigned long long* d_changed = nullptr;

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_prev, frame_bytes);
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMalloc previous frame"); return out; }
    err = cudaMalloc(&d_curr, frame_bytes);
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMalloc current frame"); cudaFree(d_prev); return out; }
    err = cudaMalloc(&d_tile_counts, static_cast<std::size_t>(tile_count) * sizeof(std::uint32_t));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMalloc tile counts"); cudaFree(d_prev); cudaFree(d_curr); return out; }
    err = cudaMalloc(&d_mask, sizeof(std::uint32_t));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMalloc mask"); cudaFree(d_prev); cudaFree(d_curr); cudaFree(d_tile_counts); return out; }
    err = cudaMalloc(&d_changed, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMalloc changed counter"); cudaFree(d_prev); cudaFree(d_curr); cudaFree(d_tile_counts); cudaFree(d_mask); return out; }

    err = cudaMemcpy(d_prev, previous_gray, frame_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemcpy previous frame"); goto cleanup; }
    err = cudaMemcpy(d_curr, current_gray, frame_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemcpy current frame"); goto cleanup; }
    err = cudaMemset(d_tile_counts, 0, static_cast<std::size_t>(tile_count) * sizeof(std::uint32_t));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemset tile counts"); goto cleanup; }
    err = cudaMemset(d_mask, 0, sizeof(std::uint32_t));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemset mask"); goto cleanup; }
    err = cudaMemset(d_changed, 0, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemset changed counter"); goto cleanup; }

    {
        const int threads = 256;
        const int blocks = (total_pixels + threads - 1) / threads;
        frame_diff_tile_mask_kernel<<<blocks, threads>>>(
            d_prev, d_curr, cfg.width, cfg.height, cfg.tile_cols, cfg.tile_rows,
            cfg.pixel_threshold, d_tile_counts, d_mask, d_changed);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) { out.error = cuda_error(err, "frame_diff_tile_mask_kernel launch"); goto cleanup; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { out.error = cuda_error(err, "frame_diff_tile_mask_kernel sync"); goto cleanup; }

    out.tile_changed_pixels.assign(static_cast<std::size_t>(tile_count), 0U);
    err = cudaMemcpy(out.tile_changed_pixels.data(), d_tile_counts, static_cast<std::size_t>(tile_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemcpy tile counts to host"); goto cleanup; }
    err = cudaMemcpy(&out.tile_mask, d_mask, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemcpy mask to host"); goto cleanup; }
    {
        unsigned long long changed = 0ULL;
        err = cudaMemcpy(&changed, d_changed, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemcpy changed counter to host"); goto cleanup; }
        out.changed_pixels = static_cast<std::uint64_t>(changed);
    }

    out.low_half_mask = out.tile_mask & 0x0000FFFFU;
    out.high_half_mask = (out.tile_mask >> 16U) & 0x0000FFFFU;
    out.active_tiles = popcount32(out.tile_mask);
    out.low_half_active_tiles = popcount32(out.low_half_mask);
    out.high_half_active_tiles = popcount32(out.high_half_mask);
    out.changed_ratio = static_cast<double>(out.changed_pixels) / static_cast<double>(std::max(1, total_pixels));
    out.path = choose_path(out.active_tiles, cfg.sparse_threshold, cfg.dense_threshold);
    out.ok = true;

cleanup:
    cudaFree(d_prev);
    cudaFree(d_curr);
    cudaFree(d_tile_counts);
    cudaFree(d_mask);
    cudaFree(d_changed);
    return out;
}

AudioEnergyAnalysis analyze_audio_energy_cuda(
    const float* samples,
    int sample_count,
    const AudioEnergyConfig& cfg) {

    AudioEnergyAnalysis out;
    out.backend = "cuda";
    out.samples = sample_count;
    out.window_samples = cfg.window_samples;
    out.threshold = cfg.threshold;

    if (samples == nullptr && sample_count > 0) {
        out.error = "samples must not be null";
        return out;
    }
    if (sample_count < 0 || cfg.window_samples <= 0 || cfg.max_windows <= 0 || cfg.max_windows > 32) {
        out.error = "invalid audio config";
        return out;
    }

    const int windows = std::min(cfg.max_windows, (sample_count + cfg.window_samples - 1) / cfg.window_samples);
    out.rms.assign(static_cast<std::size_t>(windows), 0.0f);
    if (sample_count == 0 || windows == 0) {
        out.ok = true;
        return out;
    }

    float* d_samples = nullptr;
    float* d_rms = nullptr;
    std::uint32_t* d_mask = nullptr;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&d_samples, static_cast<std::size_t>(sample_count) * sizeof(float));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMalloc audio samples"); return out; }
    err = cudaMalloc(&d_rms, static_cast<std::size_t>(windows) * sizeof(float));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMalloc rms"); cudaFree(d_samples); return out; }
    err = cudaMalloc(&d_mask, sizeof(std::uint32_t));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMalloc audio mask"); cudaFree(d_samples); cudaFree(d_rms); return out; }
    err = cudaMemcpy(d_samples, samples, static_cast<std::size_t>(sample_count) * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemcpy audio samples"); goto cleanup; }
    err = cudaMemset(d_mask, 0, sizeof(std::uint32_t));
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemset audio mask"); goto cleanup; }

    {
        const int threads = 256;
        audio_energy_kernel<<<windows, threads, static_cast<std::size_t>(threads) * sizeof(float)>>>(
            d_samples, sample_count, cfg.window_samples, cfg.threshold, windows, d_rms, d_mask);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) { out.error = cuda_error(err, "audio_energy_kernel launch"); goto cleanup; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { out.error = cuda_error(err, "audio_energy_kernel sync"); goto cleanup; }
    err = cudaMemcpy(out.rms.data(), d_rms, static_cast<std::size_t>(windows) * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemcpy rms to host"); goto cleanup; }
    err = cudaMemcpy(&out.event_mask, d_mask, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { out.error = cuda_error(err, "cudaMemcpy audio mask to host"); goto cleanup; }

    out.active_windows = popcount32(out.event_mask);
    out.ok = true;

cleanup:
    cudaFree(d_samples);
    cudaFree(d_rms);
    cudaFree(d_mask);
    return out;
}

} // namespace node1_non_llm
