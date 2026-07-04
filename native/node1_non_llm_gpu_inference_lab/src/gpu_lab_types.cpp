#include "node1_non_llm/gpu_lab_types.hpp"

#include <cstdio>

namespace node1_non_llm {

const char* workload_path_name(WorkloadPath path) noexcept {
    switch (path) {
        case WorkloadPath::Sparse: return "sparse";
        case WorkloadPath::Dense: return "dense";
        case WorkloadPath::Mixed: return "mixed";
    }
    return "unknown";
}

WorkloadPath choose_workload_path(int active_tiles, int sparse_threshold, int dense_threshold) noexcept {
    if (active_tiles <= sparse_threshold) {
        return WorkloadPath::Sparse;
    }
    if (active_tiles >= dense_threshold) {
        return WorkloadPath::Dense;
    }
    return WorkloadPath::Mixed;
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

bool validate_audio_config(int sample_count, const AudioEnergyConfig& cfg, std::string& error) noexcept {
    if (sample_count < 0) {
        error = "sample_count must be non-negative";
        return false;
    }
    if (cfg.window_samples <= 0) {
        error = "window_samples must be positive";
        return false;
    }
    if (cfg.max_windows <= 0 || cfg.max_windows > 32) {
        error = "max_windows must be in [1, 32]";
        return false;
    }
    if (cfg.threshold < 0.0f) {
        error = "threshold must be non-negative";
        return false;
    }
    return true;
}

} // namespace node1_non_llm
