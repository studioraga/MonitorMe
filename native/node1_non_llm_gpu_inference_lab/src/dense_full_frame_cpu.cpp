#include "node1_non_llm/dense_full_frame.hpp"

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace node1_non_llm {

bool validate_dense_full_frame_config(const DenseFullFrameConfig& cfg, std::string& error) noexcept {
    if (cfg.width <= 0 || cfg.height <= 0) {
        error = "width and height must be positive";
        return false;
    }
    if (cfg.pixel_threshold < 0 || cfg.pixel_threshold > 255) {
        error = "pixel_threshold must be in [0, 255]";
        return false;
    }
    error.clear();
    return true;
}

DenseFullFrameAnalysis analyze_dense_full_frame_cpu(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const DenseFullFrameConfig& cfg) {

    HostStageTimer total_timer;
    DenseFullFrameAnalysis out;
    out.backend = "cpu";
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

    const std::size_t total_pixels = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);
    if (cfg.collect_output) {
        out.normalized.assign(total_pixels, 0.0f);
    }

    HostStageTimer kernel_timer;
    std::uint64_t sum_diff = 0;
    std::uint64_t sum_prev = 0;
    std::uint64_t sum_curr = 0;
    int min_diff = 255;
    int max_diff = 0;
    int min_curr = 255;
    int max_curr = 0;

    for (std::size_t i = 0; i < total_pixels; ++i) {
        const int prev = static_cast<int>(previous_gray[i]);
        const int curr = static_cast<int>(current_gray[i]);
        const int diff = std::abs(curr - prev);
        ++out.diff_histogram[static_cast<std::size_t>(diff)];
        ++out.histogram_total;
        if (diff > cfg.pixel_threshold) {
            ++out.changed_pixels;
        }
        sum_diff += static_cast<std::uint64_t>(diff);
        sum_prev += static_cast<std::uint64_t>(prev);
        sum_curr += static_cast<std::uint64_t>(curr);
        min_diff = std::min(min_diff, diff);
        max_diff = std::max(max_diff, diff);
        min_curr = std::min(min_curr, curr);
        max_curr = std::max(max_curr, curr);
        if (cfg.collect_output) {
            out.normalized[i] = static_cast<float>(curr) / 255.0f;
        }
    }
    out.timing.kernel_ms = kernel_timer.elapsed_ms();

    const double n = static_cast<double>(std::max<std::size_t>(1U, total_pixels));
    out.diff_min = min_diff;
    out.diff_max = max_diff;
    out.diff_mean = static_cast<double>(sum_diff) / n;
    out.previous_mean = static_cast<double>(sum_prev) / n;
    out.current_mean = static_cast<double>(sum_curr) / n;
    out.lighting_delta = std::abs(out.current_mean - out.previous_mean);
    out.changed_ratio = static_cast<double>(out.changed_pixels) / n;
    out.output_min = static_cast<float>(min_curr) / 255.0f;
    out.output_max = static_cast<float>(max_curr) / 255.0f;
    out.output_mean = out.current_mean / 255.0;
    out.ok = true;
    out.timing.total_ms = total_timer.elapsed_ms();
    return out;
}

} // namespace node1_non_llm
