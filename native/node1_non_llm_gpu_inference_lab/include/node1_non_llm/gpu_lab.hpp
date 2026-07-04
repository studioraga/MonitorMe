#pragma once

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
    std::string error;
};

const char* workload_path_name(WorkloadPath path) noexcept;
std::string hex32(std::uint32_t value);
int popcount32(std::uint32_t value) noexcept;

bool validate_tile_config(const TileAnalysisConfig& cfg, std::string& error) noexcept;

FrameAnalysis analyze_gray_frames_cpu(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const TileAnalysisConfig& cfg);

AudioEnergyAnalysis analyze_audio_energy_cpu(
    const float* samples,
    int sample_count,
    const AudioEnergyConfig& cfg);

#ifdef NODE1_NON_LLM_WITH_CUDA
FrameAnalysis analyze_gray_frames_cuda(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const TileAnalysisConfig& cfg);

AudioEnergyAnalysis analyze_audio_energy_cuda(
    const float* samples,
    int sample_count,
    const AudioEnergyConfig& cfg);
#endif

} // namespace node1_non_llm
