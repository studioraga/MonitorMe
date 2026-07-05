#include "node1_non_llm/mixed_region.hpp"

#include "node1_non_llm/gpu_lab_types.hpp"
#include "node1_non_llm/gpu_lab_timing.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <string>
#include <vector>

namespace node1_non_llm {

bool validate_mixed_region_config(const MixedRegionConfig& cfg, std::string& error) noexcept {
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
    if (cfg.target_width <= 0 || cfg.target_height <= 0) {
        error = "target width and height must be positive";
        return false;
    }
    if (cfg.target_width > 1024 || cfg.target_height > 1024) {
        error = "target width and height must be <= 1024";
        return false;
    }
    if (cfg.max_groups < 0 || cfg.max_groups > 32) {
        error = "max_groups must be in [0, 32]";
        return false;
    }
    const int tile_count = cfg.tile_cols * cfg.tile_rows;
    if (tile_count < 32) {
        const std::uint32_t valid_mask = (1U << tile_count) - 1U;
        if ((cfg.tile_mask & ~valid_mask) != 0U) {
            error = "tile_mask has bits outside the configured tile grid";
            return false;
        }
    }
    error.clear();
    return true;
}

std::vector<MixedRegionGroup> connected_tile_components(const MixedRegionConfig& cfg) {
    std::vector<MixedRegionGroup> groups;
    const int tile_count = cfg.tile_cols * cfg.tile_rows;
    std::array<bool, 32> visited{};

    auto is_active = [&](int tile) -> bool {
        return tile >= 0 && tile < tile_count && ((cfg.tile_mask & (1U << tile)) != 0U);
    };

    for (int start = 0; start < tile_count && static_cast<int>(groups.size()) < cfg.max_groups; ++start) {
        if (!is_active(start) || visited[static_cast<std::size_t>(start)]) {
            continue;
        }

        MixedRegionGroup group;
        group.component_index = static_cast<int>(groups.size());
        group.min_tile_col = cfg.tile_cols;
        group.min_tile_row = cfg.tile_rows;
        group.max_tile_col = 0;
        group.max_tile_row = 0;

        std::queue<int> q;
        q.push(start);
        visited[static_cast<std::size_t>(start)] = true;

        while (!q.empty()) {
            const int tile = q.front();
            q.pop();
            const int col = tile % cfg.tile_cols;
            const int row = tile / cfg.tile_cols;
            group.tile_indices.push_back(tile);
            group.tile_mask |= (1U << tile);
            group.min_tile_col = std::min(group.min_tile_col, col);
            group.min_tile_row = std::min(group.min_tile_row, row);
            group.max_tile_col = std::max(group.max_tile_col, col);
            group.max_tile_row = std::max(group.max_tile_row, row);

            const int neighbors[4] = {
                (col > 0) ? tile - 1 : -1,
                (col + 1 < cfg.tile_cols) ? tile + 1 : -1,
                (row > 0) ? tile - cfg.tile_cols : -1,
                (row + 1 < cfg.tile_rows) ? tile + cfg.tile_cols : -1,
            };
            for (int nb : neighbors) {
                if (is_active(nb) && !visited[static_cast<std::size_t>(nb)]) {
                    visited[static_cast<std::size_t>(nb)] = true;
                    q.push(nb);
                }
            }
        }

        std::sort(group.tile_indices.begin(), group.tile_indices.end());
        group.tile_count = static_cast<int>(group.tile_indices.size());
        const int bbox_cols = group.max_tile_col - group.min_tile_col + 1;
        const int bbox_rows = group.max_tile_row - group.min_tile_row + 1;
        group.classification = (group.tile_count == bbox_cols * bbox_rows) ? "contiguous" : "scattered";

        const int x0 = (group.min_tile_col * cfg.width) / cfg.tile_cols;
        const int x1 = ((group.max_tile_col + 1) * cfg.width) / cfg.tile_cols;
        const int y0 = (group.min_tile_row * cfg.height) / cfg.tile_rows;
        const int y1 = ((group.max_tile_row + 1) * cfg.height) / cfg.tile_rows;
        group.x = x0;
        group.y = y0;
        group.width = std::max(1, x1 - x0);
        group.height = std::max(1, y1 - y0);
        groups.push_back(std::move(group));
    }
    return groups;
}

MixedRegionAnalysis analyze_mixed_region_cpu(const std::uint8_t* gray, const MixedRegionConfig& cfg) {
    HostStageTimer total_timer;
    MixedRegionAnalysis out;
    out.backend = "cpu";
    out.width = cfg.width;
    out.height = cfg.height;
    out.tile_cols = cfg.tile_cols;
    out.tile_rows = cfg.tile_rows;
    out.tile_mask = cfg.tile_mask;
    out.active_tiles = popcount32(cfg.tile_mask);
    out.target_width = cfg.target_width;
    out.target_height = cfg.target_height;

    std::string error;
    if (!gray) {
        out.error = "gray input pointer is null";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (!validate_mixed_region_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    HostStageTimer kernel_timer;
    out.groups = connected_tile_components(cfg);
    out.component_count = static_cast<int>(out.groups.size());
    out.group_count = out.component_count;
    out.classification = out.group_count == 0 ? "empty" : (out.group_count == 1 ? out.groups[0].classification : "scattered");
    for (const auto& group : out.groups) {
        if (group.classification == "contiguous") {
            ++out.contiguous_components;
        } else {
            ++out.scattered_components;
        }
        out.source_pixels_covered += static_cast<std::uint64_t>(group.width) * static_cast<std::uint64_t>(group.height);
    }

    const std::size_t target_pixels = static_cast<std::size_t>(cfg.target_width) * static_cast<std::size_t>(cfg.target_height);
    out.output_elements = static_cast<std::uint64_t>(out.group_count) * static_cast<std::uint64_t>(target_pixels);
    out.bytes_read = out.output_elements;
    out.bytes_written = out.output_elements * sizeof(float);
    if (cfg.collect_output) {
        out.normalized.assign(static_cast<std::size_t>(out.output_elements), 0.0f);
    }

    double sum = 0.0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    std::size_t write_index = 0;
    for (const auto& group : out.groups) {
        for (int oy = 0; oy < cfg.target_height; ++oy) {
            const int rel_y = std::min((oy * group.height) / cfg.target_height, group.height - 1);
            const int sy = group.y + rel_y;
            for (int ox = 0; ox < cfg.target_width; ++ox) {
                const int rel_x = std::min((ox * group.width) / cfg.target_width, group.width - 1);
                const int sx = group.x + rel_x;
                const float value = static_cast<float>(gray[static_cast<std::size_t>(sy) * cfg.width + sx]) / 255.0f;
                if (cfg.collect_output) {
                    out.normalized[write_index] = value;
                }
                ++write_index;
                sum += static_cast<double>(value);
                min_v = std::min(min_v, value);
                max_v = std::max(max_v, value);
            }
        }
    }
    out.timing.kernel_ms = kernel_timer.elapsed_ms();
    if (out.output_elements > 0) {
        out.output_min = min_v;
        out.output_max = max_v;
        out.output_mean = sum / static_cast<double>(out.output_elements);
    }
    out.ok = true;
    out.timing.total_ms = total_timer.elapsed_ms();
    return out;
}

} // namespace node1_non_llm
