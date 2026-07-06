#pragma once

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

struct OverlayHeavyConfig {
    int width = 0;
    int height = 0;
    int pixel_threshold = 30;
    int alpha = 128; // 0..255 fixed-point blend factor.
    int thumbnail_width = 64;
    int thumbnail_height = 48;
    bool collect_output = true;
};

struct OverlayHeavyAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    std::string schema = "node1_non_llm_overlay_heavy.v0.1";
    int width = 0;
    int height = 0;
    int pixel_threshold = 30;
    int alpha = 128;
    double alpha_ratio = 128.0 / 255.0;
    int thumbnail_width = 64;
    int thumbnail_height = 48;
    std::uint64_t pixels_processed = 0;
    std::uint64_t changed_pixels = 0;
    double changed_ratio = 0.0;
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_written = 0;
    int heatmap_min = 0;
    int heatmap_max = 0;
    double heatmap_mean = 0.0;
    int before_after_max_diff = 0;
    double before_after_abs_mean = 0.0;
    double previous_mean = 0.0;
    double current_mean = 0.0;
    double lighting_delta = 0.0;
    double overlay_mean = 0.0;
    double thumbnail_mean = 0.0;
    std::vector<std::uint8_t> heatmap;
    std::vector<std::uint8_t> overlay_rgb;
    std::vector<std::uint8_t> thumbnail_rgb;
    StageTiming timing;
    std::string error;
};

bool validate_overlay_heavy_config(const OverlayHeavyConfig& cfg, std::string& error) noexcept;
OverlayHeavyAnalysis analyze_overlay_heavy_cpu(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const OverlayHeavyConfig& cfg);

#ifdef NODE1_NON_LLM_WITH_CUDA
OverlayHeavyAnalysis analyze_overlay_heavy_cuda(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const OverlayHeavyConfig& cfg);
#endif

} // namespace node1_non_llm
