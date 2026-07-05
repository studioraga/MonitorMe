#pragma once

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

struct SparseRoiRect {
    int tile_index = 0;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

struct SparseRoiConfig {
    int width = 0;
    int height = 0;
    int tile_cols = 8;
    int tile_rows = 4;
    std::uint32_t tile_mask = 0U;
    int target_width = 16;
    int target_height = 16;
    int max_rois = 32;
    bool collect_output = true;
};

struct SparseRoiAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    std::string schema = "node1_non_llm_sparse_roi.v0.1";
    int width = 0;
    int height = 0;
    int tile_cols = 0;
    int tile_rows = 0;
    std::uint32_t tile_mask = 0U;
    int active_tiles = 0;
    int roi_count = 0;
    int target_width = 0;
    int target_height = 0;
    std::uint64_t source_pixels_covered = 0;
    std::uint64_t output_elements = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_written = 0;
    std::vector<SparseRoiRect> rois;
    std::vector<float> normalized;
    float output_min = 0.0f;
    float output_max = 0.0f;
    double output_mean = 0.0;
    StageTiming timing;
    std::string error;
};

bool validate_sparse_roi_config(const SparseRoiConfig& cfg, std::string& error) noexcept;
std::vector<SparseRoiRect> active_tile_rois(const SparseRoiConfig& cfg);
SparseRoiAnalysis analyze_sparse_roi_cpu(const std::uint8_t* gray, const SparseRoiConfig& cfg);

#ifdef NODE1_NON_LLM_WITH_CUDA
SparseRoiAnalysis analyze_sparse_roi_cuda(const std::uint8_t* gray, const SparseRoiConfig& cfg);
#endif

} // namespace node1_non_llm
