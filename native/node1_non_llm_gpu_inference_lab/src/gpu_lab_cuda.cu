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

} // namespace node1_non_llm
