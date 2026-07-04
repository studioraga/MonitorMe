#include "node1_non_llm/gpu_lab.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <sstream>

namespace node1_non_llm {

const char* workload_path_name(WorkloadPath path) noexcept {
    switch (path) {
        case WorkloadPath::Sparse: return "sparse";
        case WorkloadPath::Dense: return "dense";
        case WorkloadPath::Mixed: return "mixed";
    }
    return "unknown";
}

std::string hex32(std::uint32_t value) {
    char buf[16] = {0};
    std::snprintf(buf, sizeof(buf), "0x%08X", value);
    return std::string(buf);
}

int popcount32(std::uint32_t value) noexcept {
#if defined(__GNUG__) || defined(__clang__)
    return __builtin_popcount(value);
#else
    int count = 0;
    while (value != 0U) {
        value &= (value - 1U);
        ++count;
    }
    return count;
#endif
}

bool validate_tile_config(const TileAnalysisConfig& cfg, std::string& error) noexcept {
    if (cfg.width <= 0 || cfg.height <= 0) {
        error = "width and height must be positive";
        return false;
    }
    if (cfg.tile_cols <= 0 || cfg.tile_rows <= 0) {
        error = "tile grid must be positive";
        return false;
    }
    if (cfg.tile_cols * cfg.tile_rows > 32) {
        error = "tile_cols * tile_rows must be <= 32 because the tile mask is uint32_t";
        return false;
    }
    if (cfg.pixel_threshold < 0 || cfg.pixel_threshold > 255) {
        error = "pixel_threshold must be in [0, 255]";
        return false;
    }
    if (cfg.sparse_threshold < 0 || cfg.dense_threshold < 0 || cfg.sparse_threshold >= cfg.dense_threshold) {
        error = "sparse_threshold must be lower than dense_threshold";
        return false;
    }
    if (cfg.dense_threshold > cfg.tile_cols * cfg.tile_rows) {
        error = "dense_threshold cannot exceed tile count";
        return false;
    }
    return true;
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

FrameAnalysis analyze_gray_frames_cpu(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const TileAnalysisConfig& cfg) {

    FrameAnalysis out;
    out.backend = "cpu";
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

    const int tile_count = cfg.tile_cols * cfg.tile_rows;
    out.tile_changed_pixels.assign(static_cast<std::size_t>(tile_count), 0U);

    std::uint32_t mask = 0U;
    std::uint64_t changed_pixels = 0U;
    const int total_pixels = cfg.width * cfg.height;

    for (int y = 0; y < cfg.height; ++y) {
        const int tile_y = std::min((y * cfg.tile_rows) / cfg.height, cfg.tile_rows - 1);
        for (int x = 0; x < cfg.width; ++x) {
            const int idx = y * cfg.width + x;
            const int diff = std::abs(static_cast<int>(current_gray[idx]) - static_cast<int>(previous_gray[idx]));
            if (diff > cfg.pixel_threshold) {
                const int tile_x = std::min((x * cfg.tile_cols) / cfg.width, cfg.tile_cols - 1);
                const int tile = tile_y * cfg.tile_cols + tile_x;
                out.tile_changed_pixels[static_cast<std::size_t>(tile)] += 1U;
                mask |= (1U << tile);
                ++changed_pixels;
            }
        }
    }

    const std::uint32_t low_mask = mask & 0x0000FFFFU;
    const std::uint32_t high_mask = (mask >> 16U) & 0x0000FFFFU;
    const int active = popcount32(mask);

    out.ok = true;
    out.tile_mask = mask;
    out.low_half_mask = low_mask;
    out.high_half_mask = high_mask;
    out.active_tiles = active;
    out.low_half_active_tiles = popcount32(low_mask);
    out.high_half_active_tiles = popcount32(high_mask);
    out.changed_pixels = changed_pixels;
    out.changed_ratio = static_cast<double>(changed_pixels) / static_cast<double>(std::max(1, total_pixels));
    out.path = choose_path(active, cfg.sparse_threshold, cfg.dense_threshold);
    return out;
}

AudioEnergyAnalysis analyze_audio_energy_cpu(
    const float* samples,
    int sample_count,
    const AudioEnergyConfig& cfg) {

    AudioEnergyAnalysis out;
    out.backend = "cpu";
    out.samples = sample_count;
    out.window_samples = cfg.window_samples;
    out.threshold = cfg.threshold;

    if (samples == nullptr && sample_count > 0) {
        out.error = "samples must not be null";
        return out;
    }
    if (sample_count < 0) {
        out.error = "sample_count must be non-negative";
        return out;
    }
    if (cfg.window_samples <= 0) {
        out.error = "window_samples must be positive";
        return out;
    }
    if (cfg.max_windows <= 0 || cfg.max_windows > 32) {
        out.error = "max_windows must be in [1, 32]";
        return out;
    }

    const int windows = std::min(cfg.max_windows, (sample_count + cfg.window_samples - 1) / cfg.window_samples);
    out.rms.assign(static_cast<std::size_t>(windows), 0.0f);

    std::uint32_t mask = 0U;
    for (int w = 0; w < windows; ++w) {
        const int start = w * cfg.window_samples;
        const int end = std::min(sample_count, start + cfg.window_samples);
        double sum_squares = 0.0;
        for (int i = start; i < end; ++i) {
            const double v = static_cast<double>(samples[i]);
            sum_squares += v * v;
        }
        const int n = std::max(1, end - start);
        const float rms = static_cast<float>(std::sqrt(sum_squares / static_cast<double>(n)));
        out.rms[static_cast<std::size_t>(w)] = rms;
        if (rms >= cfg.threshold) {
            mask |= (1U << w);
        }
    }

    out.ok = true;
    out.event_mask = mask;
    out.active_windows = popcount32(mask);
    return out;
}

} // namespace node1_non_llm
