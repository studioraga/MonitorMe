#pragma once

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

struct DenseFullFrameConfig {
    int width = 0;
    int height = 0;
    int pixel_threshold = 30;
    bool collect_output = true;
};

struct DenseFullFrameAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    std::string schema = "node1_non_llm_dense_full_frame.v0.1";
    int width = 0;
    int height = 0;
    int pixel_threshold = 30;
    std::uint64_t pixels_processed = 0;
    std::uint64_t changed_pixels = 0;
    double changed_ratio = 0.0;
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_written = 0;
    int diff_min = 0;
    int diff_max = 0;
    double diff_mean = 0.0;
    double previous_mean = 0.0;
    double current_mean = 0.0;
    double lighting_delta = 0.0;
    std::array<std::uint64_t, 256> diff_histogram{};
    std::uint64_t histogram_total = 0;
    std::vector<float> normalized;
    float output_min = 0.0f;
    float output_max = 0.0f;
    double output_mean = 0.0;
    StageTiming timing;
    std::string error;
};

bool validate_dense_full_frame_config(const DenseFullFrameConfig& cfg, std::string& error) noexcept;
DenseFullFrameAnalysis analyze_dense_full_frame_cpu(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const DenseFullFrameConfig& cfg);

#ifdef NODE1_NON_LLM_WITH_CUDA
DenseFullFrameAnalysis analyze_dense_full_frame_cuda(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const DenseFullFrameConfig& cfg);
#endif

} // namespace node1_non_llm
