#include "node1_non_llm/gpu_lab.hpp"

#include "node1_non_llm/gpu_lab_cuda_utils.cuh"
#include "node1_non_llm/gpu_lab_timing.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {
namespace {

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

} // namespace

FrameAnalysis analyze_gray_frames_cuda(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const TileAnalysisConfig& cfg) {

    HostStageTimer total_timer;

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
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (previous_gray == nullptr || current_gray == nullptr) {
        out.error = "previous_gray and current_gray must not be null";
        out.timing.total_ms = total_timer.elapsed_ms();
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

    auto cleanup = [&]() noexcept {
        cudaFree(d_prev);
        cudaFree(d_curr);
        cudaFree(d_tile_counts);
        cudaFree(d_mask);
        cudaFree(d_changed);
    };

    cudaError_t err = cudaMalloc(&d_prev, frame_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc previous frame"); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_curr, frame_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc current frame"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_tile_counts, static_cast<std::size_t>(tile_count) * sizeof(std::uint32_t));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc tile counts"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_mask, sizeof(std::uint32_t));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc mask"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_changed, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc changed counter"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }

    {
        HostStageTimer h2d_timer;
        err = cudaMemcpy(d_prev, previous_gray, frame_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy previous frame"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(d_curr, current_gray, frame_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy current frame"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_tile_counts, 0, static_cast<std::size_t>(tile_count) * sizeof(std::uint32_t));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset tile counts"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_mask, 0, sizeof(std::uint32_t));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset mask"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_changed, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset changed counter"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.timing.h2d_ms = h2d_timer.elapsed_ms();
    }

    CudaEventTimer kernel_timer;
    if (kernel_timer.ok()) {
        err = kernel_timer.start();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaEventRecord frame start"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    }
    {
        const int threads = 256;
        const int blocks = (total_pixels + threads - 1) / threads;
        frame_diff_tile_mask_kernel<<<blocks, threads>>>(
            d_prev, d_curr, cfg.width, cfg.height, cfg.tile_cols, cfg.tile_rows,
            cfg.pixel_threshold, d_tile_counts, d_mask, d_changed);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "frame_diff_tile_mask_kernel launch"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    if (kernel_timer.ok()) {
        err = kernel_timer.stop(out.timing.kernel_ms);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "frame_diff_tile_mask_kernel event sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    } else {
        HostStageTimer sync_timer;
        err = cudaDeviceSynchronize();
        out.timing.kernel_ms = sync_timer.elapsed_ms();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "frame_diff_tile_mask_kernel sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    }

    out.tile_changed_pixels.assign(static_cast<std::size_t>(tile_count), 0U);
    {
        HostStageTimer d2h_timer;
        err = cudaMemcpy(out.tile_changed_pixels.data(), d_tile_counts, static_cast<std::size_t>(tile_count) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy tile counts to host"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&out.tile_mask, d_mask, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mask to host"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        unsigned long long changed = 0ULL;
        err = cudaMemcpy(&changed, d_changed, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy changed counter to host"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.changed_pixels = static_cast<std::uint64_t>(changed);
        out.timing.d2h_ms = d2h_timer.elapsed_ms();
    }

    out.low_half_mask = out.tile_mask & 0x0000FFFFU;
    out.high_half_mask = (out.tile_mask >> 16U) & 0x0000FFFFU;
    out.active_tiles = popcount32(out.tile_mask);
    out.low_half_active_tiles = popcount32(out.low_half_mask);
    out.high_half_active_tiles = popcount32(out.high_half_mask);
    out.changed_ratio = static_cast<double>(out.changed_pixels) / static_cast<double>(std::max(1, total_pixels));
    out.path = choose_workload_path(out.active_tiles, cfg.sparse_threshold, cfg.dense_threshold);
    out.ok = true;
    out.timing.total_ms = total_timer.elapsed_ms();

    cleanup();
    return out;
}

AudioEnergyAnalysis analyze_audio_energy_cuda(
    const float* samples,
    int sample_count,
    const AudioEnergyConfig& cfg) {

    HostStageTimer total_timer;

    AudioEnergyAnalysis out;
    out.backend = "cuda";
    out.samples = sample_count;
    out.window_samples = cfg.window_samples;
    out.threshold = cfg.threshold;

    if (samples == nullptr && sample_count > 0) {
        out.error = "samples must not be null";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    std::string error;
    if (!validate_audio_config(sample_count, cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    const int windows = std::min(cfg.max_windows, (sample_count + cfg.window_samples - 1) / cfg.window_samples);
    const std::size_t sample_bytes = static_cast<std::size_t>(sample_count) * sizeof(float);
    const std::size_t rms_bytes = static_cast<std::size_t>(windows) * sizeof(float);

    float* d_samples = nullptr;
    float* d_rms = nullptr;
    std::uint32_t* d_mask = nullptr;

    auto cleanup = [&]() noexcept {
        cudaFree(d_samples);
        cudaFree(d_rms);
        cudaFree(d_mask);
    };

    cudaError_t err = cudaMalloc(&d_samples, std::max<std::size_t>(sample_bytes, 1U));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc audio samples"); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_rms, std::max<std::size_t>(rms_bytes, 1U));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc RMS"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_mask, sizeof(std::uint32_t));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc audio mask"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }

    {
        HostStageTimer h2d_timer;
        if (sample_bytes > 0U) {
            err = cudaMemcpy(d_samples, samples, sample_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy audio samples"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        err = cudaMemset(d_rms, 0, std::max<std::size_t>(rms_bytes, 1U));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset RMS"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_mask, 0, sizeof(std::uint32_t));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset audio mask"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.timing.h2d_ms = h2d_timer.elapsed_ms();
    }

    if (windows > 0) {
        CudaEventTimer kernel_timer;
        if (kernel_timer.ok()) {
            err = kernel_timer.start();
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaEventRecord audio start"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        const int threads = 256;
        audio_energy_kernel<<<windows, threads, static_cast<std::size_t>(threads) * sizeof(float)>>>(
            d_samples, sample_count, cfg.window_samples, cfg.threshold, windows, d_rms, d_mask);
        err = cudaGetLastError();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "audio_energy_kernel launch"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        if (kernel_timer.ok()) {
            err = kernel_timer.stop(out.timing.kernel_ms);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "audio_energy_kernel event sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        } else {
            HostStageTimer sync_timer;
            err = cudaDeviceSynchronize();
            out.timing.kernel_ms = sync_timer.elapsed_ms();
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "audio_energy_kernel sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
    }

    out.rms.assign(static_cast<std::size_t>(windows), 0.0f);
    {
        HostStageTimer d2h_timer;
        if (rms_bytes > 0U) {
            err = cudaMemcpy(out.rms.data(), d_rms, rms_bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy RMS to host"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        err = cudaMemcpy(&out.event_mask, d_mask, sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy audio mask to host"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.timing.d2h_ms = d2h_timer.elapsed_ms();
    }

    out.ok = true;
    out.active_windows = popcount32(out.event_mask);
    out.timing.total_ms = total_timer.elapsed_ms();

    cleanup();
    return out;
}

namespace {

constexpr int ISP_TILE_W = 16;
constexpr int ISP_TILE_H = 16;

__device__ __forceinline__ int clamp_device_int(int value, int lo, int hi) {
    return max(lo, min(hi, value));
}

__device__ __forceinline__ std::uint8_t clamp_device_u8(int value) {
    return static_cast<std::uint8_t>(clamp_device_int(value, 0, 255));
}


__device__ __forceinline__ int rounded_sqrt_int_device(int value) {
    if (value <= 0) {
        return 0;
    }
    int root = static_cast<int>(sqrt(static_cast<double>(value)));
    while ((root + 1) * (root + 1) <= value) {
        ++root;
    }
    while (root * root > value) {
        --root;
    }
    // Match std::lround(std::sqrt(double_value)) used by the CPU path.
    // For integer value, round up when sqrt(value) >= root + 0.5.
    const int round_up_threshold = root * root + root + 1;
    if (value >= round_up_threshold) {
        return root + 1;
    }
    return root;
}

__device__ __forceinline__ std::uint8_t apply_isp_window_device(int filter, const int px[3][3]) {
    switch (filter) {
        case 0: { // blur
            int sum = 0;
            #pragma unroll
            for (int y = 0; y < 3; ++y) {
                #pragma unroll
                for (int x = 0; x < 3; ++x) sum += px[y][x];
            }
            return clamp_device_u8((sum + 4) / 9);
        }
        case 1: { // sharpen
            const int v = 5 * px[1][1] - px[0][1] - px[1][0] - px[1][2] - px[2][1];
            return clamp_device_u8(v);
        }
        case 2: { // edge
            const int v = 8 * px[1][1]
                - px[0][0] - px[0][1] - px[0][2]
                - px[1][0]            - px[1][2]
                - px[2][0] - px[2][1] - px[2][2];
            return clamp_device_u8(abs(v));
        }
        case 3: { // sobel-x
            const int sx = -px[0][0] + px[0][2]
                         - 2 * px[1][0] + 2 * px[1][2]
                         - px[2][0] + px[2][2];
            return clamp_device_u8(abs(sx));
        }
        case 4: { // sobel-y
            const int sy = -px[0][0] - 2 * px[0][1] - px[0][2]
                         + px[2][0] + 2 * px[2][1] + px[2][2];
            return clamp_device_u8(abs(sy));
        }
        case 5: { // sobel-mag
            const int sx = -px[0][0] + px[0][2]
                         - 2 * px[1][0] + 2 * px[1][2]
                         - px[2][0] + px[2][2];
            const int sy = -px[0][0] - 2 * px[0][1] - px[0][2]
                         + px[2][0] + 2 * px[2][1] + px[2][2];
            const int mag = rounded_sqrt_int_device(sx * sx + sy * sy);
            return clamp_device_u8(mag);
        }
    }
    return 0U;
}

__global__ void isp_filter_3x3_tiled_kernel(
    const std::uint8_t* input,
    std::uint8_t* output,
    int width,
    int height,
    int filter) {

    __shared__ std::uint8_t tile[ISP_TILE_H + 2][ISP_TILE_W + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_x = blockIdx.x * ISP_TILE_W;
    const int block_y = blockIdx.y * ISP_TILE_H;

    const int load_count = (ISP_TILE_W + 2) * (ISP_TILE_H + 2);
    const int linear_threads = blockDim.x * blockDim.y;
    const int linear_tid = ty * blockDim.x + tx;
    for (int i = linear_tid; i < load_count; i += linear_threads) {
        const int lx = i % (ISP_TILE_W + 2);
        const int ly = i / (ISP_TILE_W + 2);
        const int gx = clamp_device_int(block_x + lx - 1, 0, width - 1);
        const int gy = clamp_device_int(block_y + ly - 1, 0, height - 1);
        tile[ly][lx] = input[gy * width + gx];
    }
    __syncthreads();

    const int x = block_x + tx;
    const int y = block_y + ty;
    if (x >= width || y >= height) {
        return;
    }

    int px[3][3];
    #pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            px[ky][kx] = static_cast<int>(tile[ty + ky][tx + kx]);
        }
    }
    output[y * width + x] = apply_isp_window_device(filter, px);
}

__global__ void isp_metrics_kernel(
    const std::uint8_t* input,
    const std::uint8_t* output,
    int total,
    unsigned long long* sum_output,
    unsigned long long* sum_input,
    unsigned long long* sum_abs_diff,
    unsigned long long* sum_sq_output,
    unsigned long long* saturation_count) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const unsigned int out_v = static_cast<unsigned int>(output[idx]);
    const unsigned int in_v = static_cast<unsigned int>(input[idx]);
    atomicAdd(sum_output, static_cast<unsigned long long>(out_v));
    atomicAdd(sum_input, static_cast<unsigned long long>(in_v));
    atomicAdd(sum_abs_diff, static_cast<unsigned long long>(out_v > in_v ? out_v - in_v : in_v - out_v));
    atomicAdd(sum_sq_output, static_cast<unsigned long long>(out_v) * static_cast<unsigned long long>(out_v));
    if (out_v == 0U || out_v == 255U) {
        atomicAdd(saturation_count, 1ULL);
    }
}

} // namespace

IspFilterAnalysis analyze_isp_filter_cuda(
    const std::uint8_t* gray,
    const IspFilterConfig& cfg) {

    HostStageTimer total_timer;
    IspFilterAnalysis out;
    out.backend = "cuda";
    out.filter = isp_filter_name(cfg.filter);
    out.width = cfg.width;
    out.height = cfg.height;
    out.channels = 1;
    out.pixels_processed = static_cast<std::uint64_t>(cfg.width) * static_cast<std::uint64_t>(cfg.height);
    out.bytes_read = out.pixels_processed;
    out.bytes_written = out.pixels_processed;

    std::string error;
    if (!gray) {
        out.error = "gray input pointer is null";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (!validate_isp_filter_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    const int total_pixels = cfg.width * cfg.height;
    const std::size_t bytes = static_cast<std::size_t>(total_pixels);

    std::uint8_t* d_input = nullptr;
    std::uint8_t* d_output = nullptr;
    unsigned long long* d_sum_output = nullptr;
    unsigned long long* d_sum_input = nullptr;
    unsigned long long* d_sum_abs_diff = nullptr;
    unsigned long long* d_sum_sq_output = nullptr;
    unsigned long long* d_saturation = nullptr;

    auto cleanup = [&]() noexcept {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_sum_output);
        cudaFree(d_sum_input);
        cudaFree(d_sum_abs_diff);
        cudaFree(d_sum_sq_output);
        cudaFree(d_saturation);
    };

    cudaError_t err = cudaMalloc(&d_input, bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc ISP input"); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_output, bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc ISP output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_sum_output, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc ISP sum output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_sum_input, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc ISP sum input"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_sum_abs_diff, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc ISP diff sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_sum_sq_output, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc ISP sq sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_saturation, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc ISP saturation"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }

    {
        HostStageTimer h2d_timer;
        err = cudaMemcpy(d_input, gray, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy ISP input"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_output, 0, bytes);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset ISP output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_sum_output, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset ISP sum output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_sum_input, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset ISP sum input"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_sum_abs_diff, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset ISP diff sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_sum_sq_output, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset ISP sq sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_saturation, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset ISP saturation"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.timing.h2d_ms = h2d_timer.elapsed_ms();
    }

    CudaEventTimer kernel_timer;
    if (kernel_timer.ok()) {
        err = kernel_timer.start();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaEventRecord ISP start"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    }
    dim3 threads(ISP_TILE_W, ISP_TILE_H);
    dim3 blocks((cfg.width + ISP_TILE_W - 1) / ISP_TILE_W, (cfg.height + ISP_TILE_H - 1) / ISP_TILE_H);
    isp_filter_3x3_tiled_kernel<<<blocks, threads>>>(d_input, d_output, cfg.width, cfg.height, static_cast<int>(cfg.filter));
    err = cudaGetLastError();
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "isp_filter_3x3_tiled_kernel launch"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }

    const int metrics_threads = 256;
    const int metrics_blocks = (total_pixels + metrics_threads - 1) / metrics_threads;
    isp_metrics_kernel<<<metrics_blocks, metrics_threads>>>(
        d_input, d_output, total_pixels, d_sum_output, d_sum_input,
        d_sum_abs_diff, d_sum_sq_output, d_saturation);
    err = cudaGetLastError();
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "isp_metrics_kernel launch"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }

    if (kernel_timer.ok()) {
        err = kernel_timer.stop(out.timing.kernel_ms);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "ISP CUDA event sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    } else {
        HostStageTimer sync_timer;
        err = cudaDeviceSynchronize();
        out.timing.kernel_ms = sync_timer.elapsed_ms();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "ISP CUDA sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    }

    unsigned long long h_sum_output = 0ULL;
    unsigned long long h_sum_input = 0ULL;
    unsigned long long h_sum_abs_diff = 0ULL;
    unsigned long long h_sum_sq_output = 0ULL;
    unsigned long long h_saturation = 0ULL;

    std::vector<std::uint8_t> values;
    if (cfg.collect_output) {
        values.assign(static_cast<std::size_t>(total_pixels), 0U);
    }
    {
        HostStageTimer d2h_timer;
        if (cfg.collect_output) {
            err = cudaMemcpy(values.data(), d_output, bytes, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy ISP output to host"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        err = cudaMemcpy(&h_sum_output, d_sum_output, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy ISP sum output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_sum_input, d_sum_input, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy ISP sum input"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_sum_abs_diff, d_sum_abs_diff, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy ISP diff sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_sum_sq_output, d_sum_sq_output, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy ISP sq sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_saturation, d_saturation, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy ISP saturation"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.timing.d2h_ms = d2h_timer.elapsed_ms();
    }

    out.ok = true;
    out.output = std::move(values);
    if (cfg.collect_output && !out.output.empty()) {
        auto mm = std::minmax_element(out.output.begin(), out.output.end());
        out.output_min = static_cast<int>(*mm.first);
        out.output_max = static_cast<int>(*mm.second);
    } else {
        out.output_min = 0;
        out.output_max = 0;
    }
    const double n = static_cast<double>(std::max(1, total_pixels));
    out.output_mean = static_cast<double>(h_sum_output) / n;
    out.edge_energy = out.output_mean;
    const double mean = out.output_mean;
    const double mean_sq = static_cast<double>(h_sum_sq_output) / n;
    out.focus_score = std::max(0.0, mean_sq - mean * mean);
    out.noise_score = static_cast<double>(h_sum_abs_diff) / n;
    out.lighting_delta = fabs(mean - (static_cast<double>(h_sum_input) / n));
    out.saturation_pixels = static_cast<std::uint64_t>(h_saturation);
    out.saturation_ratio = static_cast<double>(h_saturation) / n;
    out.timing.total_ms = total_timer.elapsed_ms();

    cleanup();
    return out;
}

IspFilterAnalysis apply_isp_filter_cuda_tiled(
    const std::uint8_t* gray,
    const IspFilterConfig& cfg) {
    return analyze_isp_filter_cuda(gray, cfg);
}

namespace {

struct DeviceSparseRoiRect {
    int tile_index;
    int x;
    int y;
    int width;
    int height;
};

__global__ void sparse_roi_resize_normalize_kernel(
    const std::uint8_t* input,
    int image_width,
    const DeviceSparseRoiRect* rois,
    int roi_count,
    int target_width,
    int target_height,
    float* output) {

    const int target_pixels = target_width * target_height;
    const int total = roi_count * target_pixels;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int roi_idx = idx / target_pixels;
    const int local = idx - roi_idx * target_pixels;
    const int oy = local / target_width;
    const int ox = local - oy * target_width;
    const DeviceSparseRoiRect roi = rois[roi_idx];
    const int rel_x = min((ox * roi.width) / target_width, roi.width - 1);
    const int rel_y = min((oy * roi.height) / target_height, roi.height - 1);
    const int sx = roi.x + rel_x;
    const int sy = roi.y + rel_y;
    output[idx] = static_cast<float>(input[sy * image_width + sx]) / 255.0f;
}

__global__ void sparse_roi_stats_kernel(
    const float* values,
    int total,
    float* min_value,
    float* max_value,
    double* sum_value) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float v = values[idx];
    atomicAdd(sum_value, static_cast<double>(v));
    atomicMin(reinterpret_cast<int*>(min_value), __float_as_int(v));
    atomicMax(reinterpret_cast<int*>(max_value), __float_as_int(v));
}

} // namespace

SparseRoiAnalysis analyze_sparse_roi_cuda(
    const std::uint8_t* gray,
    const SparseRoiConfig& cfg) {

    HostStageTimer total_timer;
    SparseRoiAnalysis out;
    out.backend = "cuda";
    out.width = cfg.width;
    out.height = cfg.height;
    out.tile_cols = cfg.tile_cols;
    out.tile_rows = cfg.tile_rows;
    out.tile_mask = cfg.tile_mask;
    out.active_tiles = popcount32(cfg.tile_mask);
    out.target_width = cfg.target_width;
    out.target_height = cfg.target_height;

    std::string error;
    if (!gray) {
        out.error = "gray input pointer is null";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (!validate_sparse_roi_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    out.rois = active_tile_rois(cfg);
    out.roi_count = static_cast<int>(out.rois.size());
    for (const auto& roi : out.rois) {
        out.source_pixels_covered += static_cast<std::uint64_t>(roi.width) * static_cast<std::uint64_t>(roi.height);
    }
    const int target_pixels = cfg.target_width * cfg.target_height;
    const int total_output = out.roi_count * target_pixels;
    out.output_elements = static_cast<std::uint64_t>(total_output);
    out.bytes_read = out.output_elements;
    out.bytes_written = out.output_elements * sizeof(float);

    const std::size_t image_bytes = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);
    const std::size_t roi_bytes = std::max<std::size_t>(1U, out.rois.size() * sizeof(DeviceSparseRoiRect));
    const std::size_t output_bytes = std::max<std::size_t>(1U, static_cast<std::size_t>(total_output) * sizeof(float));

    std::uint8_t* d_input = nullptr;
    DeviceSparseRoiRect* d_rois = nullptr;
    float* d_output = nullptr;
    float* d_min = nullptr;
    float* d_max = nullptr;
    double* d_sum = nullptr;

    auto cleanup = [&]() noexcept {
        cudaFree(d_input);
        cudaFree(d_rois);
        cudaFree(d_output);
        cudaFree(d_min);
        cudaFree(d_max);
        cudaFree(d_sum);
    };

    cudaError_t err = cudaMalloc(&d_input, image_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc sparse ROI input"); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_rois, roi_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc sparse ROI rects"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_output, output_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc sparse ROI output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_min, sizeof(float));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc sparse ROI min"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_max, sizeof(float));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc sparse ROI max"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_sum, sizeof(double));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc sparse ROI sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }

    std::vector<DeviceSparseRoiRect> h_rois;
    h_rois.reserve(out.rois.size());
    for (const auto& roi : out.rois) {
        h_rois.push_back(DeviceSparseRoiRect{roi.tile_index, roi.x, roi.y, roi.width, roi.height});
    }

    {
        HostStageTimer h2d_timer;
        err = cudaMemcpy(d_input, gray, image_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy sparse ROI input"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        if (!h_rois.empty()) {
            err = cudaMemcpy(d_rois, h_rois.data(), h_rois.size() * sizeof(DeviceSparseRoiRect), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy sparse ROI rects"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        err = cudaMemset(d_output, 0, output_bytes);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset sparse ROI output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        const float init_min = 1.0e30f;
        const float init_max = -1.0e30f;
        const double init_sum = 0.0;
        err = cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy sparse ROI min init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy sparse ROI max init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(d_sum, &init_sum, sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy sparse ROI sum init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.timing.h2d_ms = h2d_timer.elapsed_ms();
    }

    if (total_output > 0) {
        CudaEventTimer kernel_timer;
        if (kernel_timer.ok()) {
            err = kernel_timer.start();
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaEventRecord sparse ROI start"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        const int threads = 256;
        const int blocks = (total_output + threads - 1) / threads;
        sparse_roi_resize_normalize_kernel<<<blocks, threads>>>(
            d_input, cfg.width, d_rois, out.roi_count, cfg.target_width, cfg.target_height, d_output);
        err = cudaGetLastError();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "sparse_roi_resize_normalize_kernel launch"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        sparse_roi_stats_kernel<<<blocks, threads>>>(d_output, total_output, d_min, d_max, d_sum);
        err = cudaGetLastError();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "sparse_roi_stats_kernel launch"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        if (kernel_timer.ok()) {
            err = kernel_timer.stop(out.timing.kernel_ms);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "sparse ROI CUDA event sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        } else {
            HostStageTimer sync_timer;
            err = cudaDeviceSynchronize();
            out.timing.kernel_ms = sync_timer.elapsed_ms();
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "sparse ROI CUDA sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
    }

    if (cfg.collect_output) {
        out.normalized.assign(static_cast<std::size_t>(total_output), 0.0f);
    }
    float h_min = 0.0f;
    float h_max = 0.0f;
    double h_sum = 0.0;
    {
        HostStageTimer d2h_timer;
        if (cfg.collect_output && total_output > 0) {
            err = cudaMemcpy(out.normalized.data(), d_output, static_cast<std::size_t>(total_output) * sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy sparse ROI output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        if (total_output > 0) {
            err = cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy sparse ROI min"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
            err = cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy sparse ROI max"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
            err = cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy sparse ROI sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        out.timing.d2h_ms = d2h_timer.elapsed_ms();
    }

    out.ok = true;
    if (total_output > 0) {
        out.output_min = h_min;
        out.output_max = h_max;
        out.output_mean = h_sum / static_cast<double>(total_output);
    }
    out.timing.total_ms = total_timer.elapsed_ms();
    cleanup();
    return out;
}


namespace {

struct DeviceMixedRegionGroup {
    int component_index;
    unsigned int tile_mask;
    int tile_count;
    int min_tile_col;
    int min_tile_row;
    int max_tile_col;
    int max_tile_row;
    int x;
    int y;
    int width;
    int height;
    int classification_code;
};

__global__ void mixed_region_batch_resize_normalize_kernel(
    const std::uint8_t* input,
    int image_width,
    const DeviceMixedRegionGroup* groups,
    int group_count,
    int target_width,
    int target_height,
    float* output) {

    const int target_pixels = target_width * target_height;
    const int total = group_count * target_pixels;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int group_idx = idx / target_pixels;
    const int local = idx - group_idx * target_pixels;
    const int oy = local / target_width;
    const int ox = local - oy * target_width;
    const DeviceMixedRegionGroup group = groups[group_idx];
    const int rel_x = min((ox * group.width) / target_width, group.width - 1);
    const int rel_y = min((oy * group.height) / target_height, group.height - 1);
    const int sx = group.x + rel_x;
    const int sy = group.y + rel_y;
    output[idx] = static_cast<float>(input[sy * image_width + sx]) / 255.0f;
}

__global__ void mixed_region_stats_kernel(
    const float* values,
    int total,
    float* min_value,
    float* max_value,
    double* sum_value) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float v = values[idx];
    atomicAdd(sum_value, static_cast<double>(v));
    atomicMin(reinterpret_cast<int*>(min_value), __float_as_int(v));
    atomicMax(reinterpret_cast<int*>(max_value), __float_as_int(v));
}

} // namespace

MixedRegionAnalysis analyze_mixed_region_cuda(
    const std::uint8_t* gray,
    const MixedRegionConfig& cfg) {

    HostStageTimer total_timer;
    MixedRegionAnalysis out;
    out.backend = "cuda";
    out.width = cfg.width;
    out.height = cfg.height;
    out.tile_cols = cfg.tile_cols;
    out.tile_rows = cfg.tile_rows;
    out.tile_mask = cfg.tile_mask;
    out.active_tiles = popcount32(cfg.tile_mask);
    out.target_width = cfg.target_width;
    out.target_height = cfg.target_height;

    std::string error;
    if (!gray) {
        out.error = "gray input pointer is null";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (!validate_mixed_region_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    out.groups = connected_tile_components(cfg);
    out.component_count = static_cast<int>(out.groups.size());
    out.group_count = out.component_count;
    out.classification = out.group_count == 0 ? "empty" : (out.group_count == 1 ? out.groups[0].classification : "scattered");
    for (const auto& group : out.groups) {
        if (group.classification == "contiguous") {
            ++out.contiguous_components;
        } else {
            ++out.scattered_components;
        }
        out.source_pixels_covered += static_cast<std::uint64_t>(group.width) * static_cast<std::uint64_t>(group.height);
    }

    const int target_pixels = cfg.target_width * cfg.target_height;
    const int total_output = out.group_count * target_pixels;
    out.output_elements = static_cast<std::uint64_t>(total_output);
    out.bytes_read = out.output_elements;
    out.bytes_written = out.output_elements * sizeof(float);

    const std::size_t image_bytes = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);
    const std::size_t group_bytes = std::max<std::size_t>(1U, out.groups.size() * sizeof(DeviceMixedRegionGroup));
    const std::size_t output_bytes = std::max<std::size_t>(1U, static_cast<std::size_t>(total_output) * sizeof(float));

    std::uint8_t* d_input = nullptr;
    DeviceMixedRegionGroup* d_groups = nullptr;
    float* d_output = nullptr;
    float* d_min = nullptr;
    float* d_max = nullptr;
    double* d_sum = nullptr;

    auto cleanup = [&]() noexcept {
        cudaFree(d_input);
        cudaFree(d_groups);
        cudaFree(d_output);
        cudaFree(d_min);
        cudaFree(d_max);
        cudaFree(d_sum);
    };

    cudaError_t err = cudaMalloc(&d_input, image_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc mixed region input"); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_groups, group_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc mixed region groups"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_output, output_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc mixed region output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_min, sizeof(float));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc mixed region min"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_max, sizeof(float));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc mixed region max"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_sum, sizeof(double));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc mixed region sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }

    std::vector<DeviceMixedRegionGroup> h_groups;
    h_groups.reserve(out.groups.size());
    for (const auto& group : out.groups) {
        h_groups.push_back(DeviceMixedRegionGroup{
            group.component_index,
            group.tile_mask,
            group.tile_count,
            group.min_tile_col,
            group.min_tile_row,
            group.max_tile_col,
            group.max_tile_row,
            group.x,
            group.y,
            group.width,
            group.height,
            group.classification == "contiguous" ? 1 : 2});
    }

    {
        HostStageTimer h2d_timer;
        err = cudaMemcpy(d_input, gray, image_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mixed region input"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        if (!h_groups.empty()) {
            err = cudaMemcpy(d_groups, h_groups.data(), h_groups.size() * sizeof(DeviceMixedRegionGroup), cudaMemcpyHostToDevice);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mixed region groups"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        err = cudaMemset(d_output, 0, output_bytes);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset mixed region output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        const float init_min = 1.0e30f;
        const float init_max = -1.0e30f;
        const double init_sum = 0.0;
        err = cudaMemcpy(d_min, &init_min, sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mixed region min init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mixed region max init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(d_sum, &init_sum, sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mixed region sum init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.timing.h2d_ms = h2d_timer.elapsed_ms();
    }

    if (total_output > 0) {
        CudaEventTimer kernel_timer;
        if (kernel_timer.ok()) {
            err = kernel_timer.start();
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaEventRecord mixed region start"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        const int threads = 256;
        const int blocks = (total_output + threads - 1) / threads;
        mixed_region_batch_resize_normalize_kernel<<<blocks, threads>>>(
            d_input, cfg.width, d_groups, out.group_count, cfg.target_width, cfg.target_height, d_output);
        err = cudaGetLastError();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "mixed_region_batch_resize_normalize_kernel launch"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        mixed_region_stats_kernel<<<blocks, threads>>>(d_output, total_output, d_min, d_max, d_sum);
        err = cudaGetLastError();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "mixed_region_stats_kernel launch"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        if (kernel_timer.ok()) {
            err = kernel_timer.stop(out.timing.kernel_ms);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "mixed region CUDA event sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        } else {
            HostStageTimer sync_timer;
            err = cudaDeviceSynchronize();
            out.timing.kernel_ms = sync_timer.elapsed_ms();
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "mixed region CUDA sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
    }

    if (cfg.collect_output) {
        out.normalized.assign(static_cast<std::size_t>(total_output), 0.0f);
    }
    float h_min = 0.0f;
    float h_max = 0.0f;
    double h_sum = 0.0;
    {
        HostStageTimer d2h_timer;
        if (cfg.collect_output && total_output > 0) {
            err = cudaMemcpy(out.normalized.data(), d_output, static_cast<std::size_t>(total_output) * sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mixed region output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        if (total_output > 0) {
            err = cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mixed region min"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
            err = cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mixed region max"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
            err = cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy mixed region sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        out.timing.d2h_ms = d2h_timer.elapsed_ms();
    }

    out.ok = true;
    if (total_output > 0) {
        out.output_min = h_min;
        out.output_max = h_max;
        out.output_mean = h_sum / static_cast<double>(total_output);
    }
    out.timing.total_ms = total_timer.elapsed_ms();
    cleanup();
    return out;
}


namespace {

__global__ void dense_full_frame_diff_hist_normalize_kernel(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    int total,
    int pixel_threshold,
    float* normalized,
    unsigned long long* diff_histogram,
    unsigned long long* changed_pixels,
    unsigned long long* sum_diff,
    unsigned long long* sum_prev,
    unsigned long long* sum_curr,
    int* diff_min,
    int* diff_max,
    int* curr_min,
    int* curr_max) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const int prev = static_cast<int>(previous_gray[idx]);
    const int curr = static_cast<int>(current_gray[idx]);
    const int diff = abs(curr - prev);
    normalized[idx] = static_cast<float>(curr) / 255.0f;
    atomicAdd(&diff_histogram[diff], 1ULL);
    if (diff > pixel_threshold) {
        atomicAdd(changed_pixels, 1ULL);
    }
    atomicAdd(sum_diff, static_cast<unsigned long long>(diff));
    atomicAdd(sum_prev, static_cast<unsigned long long>(prev));
    atomicAdd(sum_curr, static_cast<unsigned long long>(curr));
    atomicMin(diff_min, diff);
    atomicMax(diff_max, diff);
    atomicMin(curr_min, curr);
    atomicMax(curr_max, curr);
}

} // namespace

DenseFullFrameAnalysis analyze_dense_full_frame_cuda(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const DenseFullFrameConfig& cfg) {

    HostStageTimer total_timer;
    DenseFullFrameAnalysis out;
    out.backend = "cuda";
    out.width = cfg.width;
    out.height = cfg.height;
    out.pixel_threshold = cfg.pixel_threshold;
    out.pixels_processed = static_cast<std::uint64_t>(cfg.width) * static_cast<std::uint64_t>(cfg.height);
    out.bytes_read = out.pixels_processed * 2U;
    out.bytes_written = out.pixels_processed * sizeof(float) + out.diff_histogram.size() * sizeof(std::uint64_t);

    std::string error;
    if (!previous_gray || !current_gray) {
        out.error = "previous_gray and current_gray must not be null";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (!validate_dense_full_frame_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    const int total_pixels = cfg.width * cfg.height;
    const std::size_t frame_bytes = static_cast<std::size_t>(total_pixels);
    const std::size_t output_bytes = std::max<std::size_t>(1U, static_cast<std::size_t>(total_pixels) * sizeof(float));

    std::uint8_t* d_prev = nullptr;
    std::uint8_t* d_curr = nullptr;
    float* d_normalized = nullptr;
    unsigned long long* d_hist = nullptr;
    unsigned long long* d_changed = nullptr;
    unsigned long long* d_sum_diff = nullptr;
    unsigned long long* d_sum_prev = nullptr;
    unsigned long long* d_sum_curr = nullptr;
    int* d_diff_min = nullptr;
    int* d_diff_max = nullptr;
    int* d_curr_min = nullptr;
    int* d_curr_max = nullptr;

    auto cleanup = [&]() noexcept {
        cudaFree(d_prev);
        cudaFree(d_curr);
        cudaFree(d_normalized);
        cudaFree(d_hist);
        cudaFree(d_changed);
        cudaFree(d_sum_diff);
        cudaFree(d_sum_prev);
        cudaFree(d_sum_curr);
        cudaFree(d_diff_min);
        cudaFree(d_diff_max);
        cudaFree(d_curr_min);
        cudaFree(d_curr_max);
    };

    cudaError_t err = cudaMalloc(&d_prev, frame_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense previous frame"); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_curr, frame_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense current frame"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_normalized, output_bytes);
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense normalized output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_hist, 256U * sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense diff histogram"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_changed, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense changed counter"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_sum_diff, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense diff sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_sum_prev, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense previous sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_sum_curr, sizeof(unsigned long long));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense current sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_diff_min, sizeof(int));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense diff min"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_diff_max, sizeof(int));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense diff max"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_curr_min, sizeof(int));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense current min"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    err = cudaMalloc(&d_curr_max, sizeof(int));
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMalloc dense current max"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }

    {
        HostStageTimer h2d_timer;
        err = cudaMemcpy(d_prev, previous_gray, frame_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense previous frame"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(d_curr, current_gray, frame_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense current frame"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_normalized, 0, output_bytes);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset dense normalized output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_hist, 0, 256U * sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset dense diff histogram"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_changed, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset dense changed counter"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_sum_diff, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset dense diff sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_sum_prev, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset dense previous sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemset(d_sum_curr, 0, sizeof(unsigned long long));
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemset dense current sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        const int init_min = 255;
        const int init_zero = 0;
        err = cudaMemcpy(d_diff_min, &init_min, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense diff min init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(d_diff_max, &init_zero, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense diff max init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(d_curr_min, &init_min, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense current min init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(d_curr_max, &init_zero, sizeof(int), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense current max init"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.timing.h2d_ms = h2d_timer.elapsed_ms();
    }

    CudaEventTimer kernel_timer;
    if (kernel_timer.ok()) {
        err = kernel_timer.start();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaEventRecord dense full-frame start"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    }
    const int threads = 256;
    const int blocks = (total_pixels + threads - 1) / threads;
    dense_full_frame_diff_hist_normalize_kernel<<<blocks, threads>>>(
        d_prev, d_curr, total_pixels, cfg.pixel_threshold, d_normalized, d_hist,
        d_changed, d_sum_diff, d_sum_prev, d_sum_curr,
        d_diff_min, d_diff_max, d_curr_min, d_curr_max);
    err = cudaGetLastError();
    if (err != cudaSuccess) { out.error = cuda_error_message(err, "dense_full_frame_diff_hist_normalize_kernel launch"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    if (kernel_timer.ok()) {
        err = kernel_timer.stop(out.timing.kernel_ms);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "dense full-frame CUDA event sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    } else {
        HostStageTimer sync_timer;
        err = cudaDeviceSynchronize();
        out.timing.kernel_ms = sync_timer.elapsed_ms();
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "dense full-frame CUDA sync"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
    }

    if (cfg.collect_output) {
        out.normalized.assign(static_cast<std::size_t>(total_pixels), 0.0f);
    }
    unsigned long long h_hist[256]{};
    unsigned long long h_changed = 0ULL;
    unsigned long long h_sum_diff = 0ULL;
    unsigned long long h_sum_prev = 0ULL;
    unsigned long long h_sum_curr = 0ULL;
    int h_diff_min = 0;
    int h_diff_max = 0;
    int h_curr_min = 0;
    int h_curr_max = 0;

    {
        HostStageTimer d2h_timer;
        if (cfg.collect_output) {
            err = cudaMemcpy(out.normalized.data(), d_normalized, static_cast<std::size_t>(total_pixels) * sizeof(float), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense normalized output"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        }
        err = cudaMemcpy(h_hist, d_hist, 256U * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense diff histogram"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_changed, d_changed, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense changed counter"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_sum_diff, d_sum_diff, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense diff sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_sum_prev, d_sum_prev, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense previous sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_sum_curr, d_sum_curr, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense current sum"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_diff_min, d_diff_min, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense diff min"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_diff_max, d_diff_max, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense diff max"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_curr_min, d_curr_min, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense current min"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        err = cudaMemcpy(&h_curr_max, d_curr_max, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { out.error = cuda_error_message(err, "cudaMemcpy dense current max"); cleanup(); out.timing.total_ms = total_timer.elapsed_ms(); return out; }
        out.timing.d2h_ms = d2h_timer.elapsed_ms();
    }

    out.ok = true;
    out.changed_pixels = static_cast<std::uint64_t>(h_changed);
    out.diff_min = h_diff_min;
    out.diff_max = h_diff_max;
    for (int i = 0; i < 256; ++i) {
        out.diff_histogram[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(h_hist[i]);
        out.histogram_total += static_cast<std::uint64_t>(h_hist[i]);
    }
    const double n = static_cast<double>(std::max(1, total_pixels));
    out.changed_ratio = static_cast<double>(out.changed_pixels) / n;
    out.diff_mean = static_cast<double>(h_sum_diff) / n;
    out.previous_mean = static_cast<double>(h_sum_prev) / n;
    out.current_mean = static_cast<double>(h_sum_curr) / n;
    out.lighting_delta = fabs(out.current_mean - out.previous_mean);
    out.output_min = static_cast<float>(h_curr_min) / 255.0f;
    out.output_max = static_cast<float>(h_curr_max) / 255.0f;
    out.output_mean = out.current_mean / 255.0;
    out.timing.total_ms = total_timer.elapsed_ms();
    cleanup();
    return out;
}

} // namespace node1_non_llm
