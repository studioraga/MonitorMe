#include "node1_non_llm/sparse_roi.hpp"

#include "node1_non_llm/gpu_lab_types.hpp"
#include "node1_non_llm/gpu_lab_timing.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace node1_non_llm {

bool validate_sparse_roi_config(const SparseRoiConfig& cfg, std::string& error) noexcept {
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
    if (cfg.max_rois < 0 || cfg.max_rois > 32) {
        error = "max_rois must be in [0, 32]";
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

std::vector<SparseRoiRect> active_tile_rois(const SparseRoiConfig& cfg) {
    std::vector<SparseRoiRect> rois;
    const int tile_count = cfg.tile_cols * cfg.tile_rows;
    const int limit = std::min(cfg.max_rois, tile_count);
    for (int tile = 0; tile < tile_count && static_cast<int>(rois.size()) < limit; ++tile) {
        if ((cfg.tile_mask & (1U << tile)) == 0U) {
            continue;
        }
        const int tile_x = tile % cfg.tile_cols;
        const int tile_y = tile / cfg.tile_cols;
        const int x0 = (tile_x * cfg.width) / cfg.tile_cols;
        const int x1 = ((tile_x + 1) * cfg.width) / cfg.tile_cols;
        const int y0 = (tile_y * cfg.height) / cfg.tile_rows;
        const int y1 = ((tile_y + 1) * cfg.height) / cfg.tile_rows;
        SparseRoiRect roi;
        roi.tile_index = tile;
        roi.x = x0;
        roi.y = y0;
        roi.width = std::max(1, x1 - x0);
        roi.height = std::max(1, y1 - y0);
        rois.push_back(roi);
    }
    return rois;
}

SparseRoiAnalysis analyze_sparse_roi_cpu(const std::uint8_t* gray, const SparseRoiConfig& cfg) {
    HostStageTimer total_timer;
    SparseRoiAnalysis out;
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
    if (!validate_sparse_roi_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    HostStageTimer kernel_timer;
    out.rois = active_tile_rois(cfg);
    out.roi_count = static_cast<int>(out.rois.size());
    for (const auto& roi : out.rois) {
        out.source_pixels_covered += static_cast<std::uint64_t>(roi.width) * static_cast<std::uint64_t>(roi.height);
    }

    const std::size_t target_pixels = static_cast<std::size_t>(cfg.target_width) * static_cast<std::size_t>(cfg.target_height);
    out.output_elements = static_cast<std::uint64_t>(out.rois.size()) * static_cast<std::uint64_t>(target_pixels);
    out.bytes_read = out.output_elements; // nearest-neighbor resize samples one source byte per output element.
    out.bytes_written = out.output_elements * sizeof(float);
    if (cfg.collect_output) {
        out.normalized.assign(static_cast<std::size_t>(out.output_elements), 0.0f);
    }

    double sum = 0.0;
    float min_v = std::numeric_limits<float>::infinity();
    float max_v = -std::numeric_limits<float>::infinity();
    std::size_t write_index = 0;
    for (const auto& roi : out.rois) {
        for (int oy = 0; oy < cfg.target_height; ++oy) {
            const int rel_y = std::min((oy * roi.height) / cfg.target_height, roi.height - 1);
            const int sy = roi.y + rel_y;
            for (int ox = 0; ox < cfg.target_width; ++ox) {
                const int rel_x = std::min((ox * roi.width) / cfg.target_width, roi.width - 1);
                const int sx = roi.x + rel_x;
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
