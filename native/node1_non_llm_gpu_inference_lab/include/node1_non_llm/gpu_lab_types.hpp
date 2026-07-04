#pragma once

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

enum class WorkloadPath {
    Sparse = 0,
    Dense = 1,
    Mixed = 2,
};

struct TileAnalysisConfig {
    int width = 0;
    int height = 0;
    int tile_cols = 8;
    int tile_rows = 4;
    int pixel_threshold = 30;
    int sparse_threshold = 8;
    int dense_threshold = 24;
};

struct FrameAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    int width = 0;
    int height = 0;
    int tile_cols = 8;
    int tile_rows = 4;
    int pixel_threshold = 30;
    int sparse_threshold = 8;
    int dense_threshold = 24;
    std::uint32_t tile_mask = 0;
    std::uint32_t low_half_mask = 0;
    std::uint32_t high_half_mask = 0;
    int active_tiles = 0;
    int low_half_active_tiles = 0;
    int high_half_active_tiles = 0;
    std::uint64_t changed_pixels = 0;
    double changed_ratio = 0.0;
    WorkloadPath path = WorkloadPath::Sparse;
    std::vector<std::uint32_t> tile_changed_pixels;
    StageTiming timing;
    std::string error;
};

struct AudioEnergyConfig {
    int window_samples = 1024;
    float threshold = 0.05f;
    int max_windows = 32;
};

struct AudioEnergyAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    int samples = 0;
    int window_samples = 1024;
    float threshold = 0.05f;
    std::uint32_t event_mask = 0;
    int active_windows = 0;
    std::vector<float> rms;
    StageTiming timing;
    std::string error;
};

const char* workload_path_name(WorkloadPath path) noexcept;
WorkloadPath choose_workload_path(int active_tiles, int sparse_threshold, int dense_threshold) noexcept;
std::string hex32(std::uint32_t value);
int popcount32(std::uint32_t value) noexcept;

bool validate_tile_config(const TileAnalysisConfig& cfg, std::string& error) noexcept;
bool validate_audio_config(int sample_count, const AudioEnergyConfig& cfg, std::string& error) noexcept;

} // namespace node1_non_llm
