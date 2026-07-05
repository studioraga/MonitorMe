#pragma once

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

struct MixedRegionGroup {
    int component_index = 0;
    std::uint32_t tile_mask = 0U;
    int tile_count = 0;
    int min_tile_col = 0;
    int min_tile_row = 0;
    int max_tile_col = 0;
    int max_tile_row = 0;
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
    std::string classification = "contiguous";
    std::vector<int> tile_indices;
};

struct MixedRegionConfig {
    int width = 0;
    int height = 0;
    int tile_cols = 8;
    int tile_rows = 4;
    std::uint32_t tile_mask = 0U;
    int target_width = 32;
    int target_height = 32;
    int max_groups = 32;
    bool collect_output = true;
};

struct MixedRegionAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    std::string schema = "node1_non_llm_mixed_region.v0.1";
    int width = 0;
    int height = 0;
    int tile_cols = 0;
    int tile_rows = 0;
    std::uint32_t tile_mask = 0U;
    int active_tiles = 0;
    int component_count = 0;
    int contiguous_components = 0;
    int scattered_components = 0;
    std::string classification = "empty";
    int group_count = 0;
    int target_width = 0;
    int target_height = 0;
    std::uint64_t source_pixels_covered = 0;
    std::uint64_t output_elements = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_written = 0;
    std::vector<MixedRegionGroup> groups;
    std::vector<float> normalized;
    float output_min = 0.0f;
    float output_max = 0.0f;
    double output_mean = 0.0;
    StageTiming timing;
    std::string error;
};

bool validate_mixed_region_config(const MixedRegionConfig& cfg, std::string& error) noexcept;
std::vector<MixedRegionGroup> connected_tile_components(const MixedRegionConfig& cfg);
MixedRegionAnalysis analyze_mixed_region_cpu(const std::uint8_t* gray, const MixedRegionConfig& cfg);

#ifdef NODE1_NON_LLM_WITH_CUDA
MixedRegionAnalysis analyze_mixed_region_cuda(const std::uint8_t* gray, const MixedRegionConfig& cfg);
#endif

} // namespace node1_non_llm
