#include "node1_non_llm/gpu_lab_json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>

namespace node1_non_llm {

std::string json_escape(const std::string& value) {
    std::ostringstream os;
    for (unsigned char c : value) {
        switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20U) {
                    os << "\\u00";
                    const char* hex = "0123456789ABCDEF";
                    os << hex[(c >> 4U) & 0x0FU] << hex[c & 0x0FU];
                } else {
                    os << static_cast<char>(c);
                }
                break;
        }
    }
    return os.str();
}

const char* bool_json(bool value) noexcept {
    return value ? "true" : "false";
}

std::string stage_timing_json(const StageTiming& timing) {
    std::ostringstream os;
    os << "{";
    os << "\"h2d_ms\":" << timing.h2d_ms << ",";
    os << "\"kernel_ms\":" << timing.kernel_ms << ",";
    os << "\"d2h_ms\":" << timing.d2h_ms << ",";
    os << "\"total_ms\":" << timing.total_ms;
    os << "}";
    return os.str();
}

std::string uint32_vector_json(const std::vector<std::uint32_t>& values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) os << ",";
        os << values[i];
    }
    os << "]";
    return os.str();
}


std::string uint8_vector_json(const std::vector<std::uint8_t>& values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) os << ",";
        os << static_cast<int>(values[i]);
    }
    os << "]";
    return os.str();
}

std::string float_vector_json(const std::vector<float>& values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) os << ",";
        os << values[i];
    }
    os << "]";
    return os.str();
}

std::string frame_analysis_json(const FrameAnalysis& a) {
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(a.ok) << ",";
    os << "\"backend\":\"" << json_escape(a.backend) << "\",";
    os << "\"error\":\"" << json_escape(a.error) << "\",";
    os << "\"width\":" << a.width << ",";
    os << "\"height\":" << a.height << ",";
    os << "\"tile_cols\":" << a.tile_cols << ",";
    os << "\"tile_rows\":" << a.tile_rows << ",";
    os << "\"pixel_threshold\":" << a.pixel_threshold << ",";
    os << "\"sparse_threshold\":" << a.sparse_threshold << ",";
    os << "\"dense_threshold\":" << a.dense_threshold << ",";
    os << "\"tile_mask\":" << a.tile_mask << ",";
    os << "\"tile_mask_hex\":\"" << hex32(a.tile_mask) << "\",";
    os << "\"low_half_mask_hex\":\"" << hex32(a.low_half_mask) << "\",";
    os << "\"high_half_mask_hex\":\"" << hex32(a.high_half_mask) << "\",";
    os << "\"active_tiles\":" << a.active_tiles << ",";
    os << "\"low_half_active_tiles\":" << a.low_half_active_tiles << ",";
    os << "\"high_half_active_tiles\":" << a.high_half_active_tiles << ",";
    os << "\"changed_pixels\":" << a.changed_pixels << ",";
    os << "\"changed_ratio\":" << a.changed_ratio << ",";
    os << "\"path\":\"" << workload_path_name(a.path) << "\",";
    os << "\"tile_changed_pixels\":" << uint32_vector_json(a.tile_changed_pixels) << ",";
    os << "\"timing\":" << stage_timing_json(a.timing);
    os << "}";
    return os.str();
}

std::string audio_analysis_json(const AudioEnergyAnalysis& a) {
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(a.ok) << ",";
    os << "\"backend\":\"" << json_escape(a.backend) << "\",";
    os << "\"error\":\"" << json_escape(a.error) << "\",";
    os << "\"samples\":" << a.samples << ",";
    os << "\"window_samples\":" << a.window_samples << ",";
    os << "\"threshold\":" << a.threshold << ",";
    os << "\"event_mask\":" << a.event_mask << ",";
    os << "\"event_mask_hex\":\"" << hex32(a.event_mask) << "\",";
    os << "\"active_windows\":" << a.active_windows << ",";
    os << "\"rms\":" << float_vector_json(a.rms) << ",";
    os << "\"timing\":" << stage_timing_json(a.timing);
    os << "}";
    return os.str();
}

std::string isp_filter_analysis_json(const IspFilterAnalysis& a, bool include_output) {
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(a.ok) << ",";
    os << "\"backend\":\"" << json_escape(a.backend) << "\",";
    os << "\"schema\":\"" << json_escape(a.schema) << "\",";
    os << "\"profile\":\"" << json_escape(a.profile) << "\",";
    os << "\"filter\":\"" << json_escape(a.filter) << "\",";
    os << "\"error\":\"" << json_escape(a.error) << "\",";
    os << "\"width\":" << a.width << ",";
    os << "\"height\":" << a.height << ",";
    os << "\"channels\":" << a.channels << ",";
    os << "\"pixels_processed\":" << a.pixels_processed << ",";
    os << "\"bytes_read\":" << a.bytes_read << ",";
    os << "\"bytes_written\":" << a.bytes_written << ",";
    os << "\"output_min\":" << a.output_min << ",";
    os << "\"output_max\":" << a.output_max << ",";
    os << "\"output_mean\":" << a.output_mean << ",";
    os << "\"edge_energy\":" << a.edge_energy << ",";
    os << "\"focus_score\":" << a.focus_score << ",";
    os << "\"noise_score\":" << a.noise_score << ",";
    os << "\"lighting_delta\":" << a.lighting_delta << ",";
    os << "\"saturation_pixels\":" << a.saturation_pixels << ",";
    os << "\"saturation_ratio\":" << a.saturation_ratio << ",";
    os << "\"facts_only\":true,";
    os << "\"note\":\"ISP filter metrics only; no object, identity, behavior, or intent claim is emitted.\",";
    os << "\"timing\":" << stage_timing_json(a.timing);
    if (include_output) {
        os << ",\"output\":" << uint8_vector_json(a.output);
    }
    os << "}";
    return os.str();
}

std::string isp_cpu_cuda_comparison_json(const IspFilterAnalysis& cpu, const IspFilterAnalysis& cuda) {
    std::ostringstream os;
    bool output_equal = false;
    int max_abs_diff = 0;
    std::size_t mismatch_count = 0;
    if (cpu.ok && cuda.ok && cpu.output.size() == cuda.output.size()) {
        output_equal = true;
        for (std::size_t i = 0; i < cpu.output.size(); ++i) {
            const int diff = std::abs(static_cast<int>(cpu.output[i]) - static_cast<int>(cuda.output[i]));
            max_abs_diff = std::max(max_abs_diff, diff);
            if (diff != 0) {
                output_equal = false;
                ++mismatch_count;
            }
        }
    }
    const double edge_energy_abs_diff = std::abs(cpu.edge_energy - cuda.edge_energy);
    const double focus_score_abs_diff = std::abs(cpu.focus_score - cuda.focus_score);
    const double noise_score_abs_diff = std::abs(cpu.noise_score - cuda.noise_score);
    const double lighting_delta_abs_diff = std::abs(cpu.lighting_delta - cuda.lighting_delta);
    const double saturation_ratio_abs_diff = std::abs(cpu.saturation_ratio - cuda.saturation_ratio);
    const bool metrics_close = edge_energy_abs_diff <= 1e-6
        && focus_score_abs_diff <= 1e-3
        && noise_score_abs_diff <= 1e-6
        && lighting_delta_abs_diff <= 1e-6
        && saturation_ratio_abs_diff <= 1e-9
        && cpu.saturation_pixels == cuda.saturation_pixels;
    os << "{";
    os << "\"ok\":" << bool_json(cpu.ok && cuda.ok && output_equal && metrics_close) << ",";
    os << "\"schema\":\"node1_non_llm_isp_cpu_cuda_compare.v0.1\",";
    os << "\"filter\":\"" << json_escape(cpu.filter) << "\",";
    os << "\"output_equal\":" << bool_json(output_equal) << ",";
    os << "\"mismatch_count\":" << mismatch_count << ",";
    os << "\"max_abs_diff\":" << max_abs_diff << ",";
    os << "\"metrics_close\":" << bool_json(metrics_close) << ",";
    os << "\"edge_energy_abs_diff\":" << edge_energy_abs_diff << ",";
    os << "\"focus_score_abs_diff\":" << focus_score_abs_diff << ",";
    os << "\"noise_score_abs_diff\":" << noise_score_abs_diff << ",";
    os << "\"lighting_delta_abs_diff\":" << lighting_delta_abs_diff << ",";
    os << "\"saturation_ratio_abs_diff\":" << saturation_ratio_abs_diff << ",";
    os << "\"saturation_pixels_equal\":" << bool_json(cpu.saturation_pixels == cuda.saturation_pixels) << ",";
    os << "\"facts_only\":true";
    os << "}";
    return os.str();
}

std::string sparse_roi_rect_json(const SparseRoiRect& r) {
    std::ostringstream os;
    os << "{";
    os << "\"tile_index\":" << r.tile_index << ",";
    os << "\"x\":" << r.x << ",";
    os << "\"y\":" << r.y << ",";
    os << "\"width\":" << r.width << ",";
    os << "\"height\":" << r.height;
    os << "}";
    return os.str();
}

std::string sparse_roi_rects_json(const std::vector<SparseRoiRect>& rois) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < rois.size(); ++i) {
        if (i) os << ",";
        os << sparse_roi_rect_json(rois[i]);
    }
    os << "]";
    return os.str();
}

std::string float_sample_vector_json(const std::vector<float>& values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) os << ",";
        os << values[i];
    }
    os << "]";
    return os.str();
}

std::string sparse_roi_analysis_json(const SparseRoiAnalysis& a, bool include_output) {
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(a.ok) << ",";
    os << "\"backend\":\"" << json_escape(a.backend) << "\",";
    os << "\"schema\":\"" << json_escape(a.schema) << "\",";
    os << "\"error\":\"" << json_escape(a.error) << "\",";
    os << "\"width\":" << a.width << ",";
    os << "\"height\":" << a.height << ",";
    os << "\"tile_cols\":" << a.tile_cols << ",";
    os << "\"tile_rows\":" << a.tile_rows << ",";
    os << "\"tile_mask\":" << a.tile_mask << ",";
    os << "\"tile_mask_hex\":\"" << hex32(a.tile_mask) << "\",";
    os << "\"active_tiles\":" << a.active_tiles << ",";
    os << "\"roi_count\":" << a.roi_count << ",";
    os << "\"target_width\":" << a.target_width << ",";
    os << "\"target_height\":" << a.target_height << ",";
    os << "\"source_pixels_covered\":" << a.source_pixels_covered << ",";
    os << "\"output_elements\":" << a.output_elements << ",";
    os << "\"bytes_read\":" << a.bytes_read << ",";
    os << "\"bytes_written\":" << a.bytes_written << ",";
    os << "\"output_min\":" << a.output_min << ",";
    os << "\"output_max\":" << a.output_max << ",";
    os << "\"output_mean\":" << a.output_mean << ",";
    os << "\"rois\":" << sparse_roi_rects_json(a.rois) << ",";
    os << "\"facts_only\":true,";
    os << "\"note\":\"Sparse ROI crop/resize/normalize workload metrics only; no object, identity, behavior, or intent claim is emitted.\",";
    os << "\"timing\":" << stage_timing_json(a.timing);
    if (include_output) {
        os << ",\"normalized\":" << float_sample_vector_json(a.normalized);
    }
    os << "}";
    return os.str();
}

std::string sparse_roi_cpu_cuda_comparison_json(const SparseRoiAnalysis& cpu, const SparseRoiAnalysis& cuda) {
    bool rois_equal = cpu.rois.size() == cuda.rois.size();
    if (rois_equal) {
        for (std::size_t i = 0; i < cpu.rois.size(); ++i) {
            const auto& a = cpu.rois[i];
            const auto& b = cuda.rois[i];
            if (a.tile_index != b.tile_index || a.x != b.x || a.y != b.y || a.width != b.width || a.height != b.height) {
                rois_equal = false;
                break;
            }
        }
    }
    bool output_close = false;
    double max_abs_diff = 0.0;
    std::size_t mismatch_count = 0;
    if (cpu.ok && cuda.ok && cpu.normalized.size() == cuda.normalized.size()) {
        output_close = true;
        for (std::size_t i = 0; i < cpu.normalized.size(); ++i) {
            const double diff = std::abs(static_cast<double>(cpu.normalized[i]) - static_cast<double>(cuda.normalized[i]));
            max_abs_diff = std::max(max_abs_diff, diff);
            if (diff > 1e-7) {
                output_close = false;
                ++mismatch_count;
            }
        }
    }
    const double output_mean_abs_diff = std::abs(cpu.output_mean - cuda.output_mean);
    const bool metrics_close = output_mean_abs_diff <= 1e-9
        && std::abs(static_cast<double>(cpu.output_min) - static_cast<double>(cuda.output_min)) <= 1e-7
        && std::abs(static_cast<double>(cpu.output_max) - static_cast<double>(cuda.output_max)) <= 1e-7;
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(cpu.ok && cuda.ok && rois_equal && output_close && metrics_close) << ",";
    os << "\"schema\":\"node1_non_llm_sparse_roi_cpu_cuda_compare.v0.1\",";
    os << "\"tile_mask_hex\":\"" << hex32(cpu.tile_mask) << "\",";
    os << "\"roi_count\":" << cpu.roi_count << ",";
    os << "\"rois_equal\":" << bool_json(rois_equal) << ",";
    os << "\"output_close\":" << bool_json(output_close) << ",";
    os << "\"mismatch_count\":" << mismatch_count << ",";
    os << "\"max_abs_diff\":" << max_abs_diff << ",";
    os << "\"metrics_close\":" << bool_json(metrics_close) << ",";
    os << "\"output_mean_abs_diff\":" << output_mean_abs_diff << ",";
    os << "\"facts_only\":true";
    os << "}";
    return os.str();
}


std::string int_vector_json(const std::vector<int>& values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) os << ",";
        os << values[i];
    }
    os << "]";
    return os.str();
}

std::string mixed_region_group_json(const MixedRegionGroup& g) {
    std::ostringstream os;
    os << "{";
    os << "\"component_index\":" << g.component_index << ",";
    os << "\"tile_mask\":" << g.tile_mask << ",";
    os << "\"tile_mask_hex\":\"" << hex32(g.tile_mask) << "\",";
    os << "\"tile_count\":" << g.tile_count << ",";
    os << "\"classification\":\"" << json_escape(g.classification) << "\",";
    os << "\"min_tile_col\":" << g.min_tile_col << ",";
    os << "\"min_tile_row\":" << g.min_tile_row << ",";
    os << "\"max_tile_col\":" << g.max_tile_col << ",";
    os << "\"max_tile_row\":" << g.max_tile_row << ",";
    os << "\"x\":" << g.x << ",";
    os << "\"y\":" << g.y << ",";
    os << "\"width\":" << g.width << ",";
    os << "\"height\":" << g.height << ",";
    os << "\"tile_indices\":" << int_vector_json(g.tile_indices);
    os << "}";
    return os.str();
}

std::string mixed_region_groups_json(const std::vector<MixedRegionGroup>& groups) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (i) os << ",";
        os << mixed_region_group_json(groups[i]);
    }
    os << "]";
    return os.str();
}

std::string mixed_region_analysis_json(const MixedRegionAnalysis& a, bool include_output) {
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(a.ok) << ",";
    os << "\"backend\":\"" << json_escape(a.backend) << "\",";
    os << "\"schema\":\"" << json_escape(a.schema) << "\",";
    os << "\"error\":\"" << json_escape(a.error) << "\",";
    os << "\"width\":" << a.width << ",";
    os << "\"height\":" << a.height << ",";
    os << "\"tile_cols\":" << a.tile_cols << ",";
    os << "\"tile_rows\":" << a.tile_rows << ",";
    os << "\"tile_mask\":" << a.tile_mask << ",";
    os << "\"tile_mask_hex\":\"" << hex32(a.tile_mask) << "\",";
    os << "\"active_tiles\":" << a.active_tiles << ",";
    os << "\"component_count\":" << a.component_count << ",";
    os << "\"contiguous_components\":" << a.contiguous_components << ",";
    os << "\"scattered_components\":" << a.scattered_components << ",";
    os << "\"classification\":\"" << json_escape(a.classification) << "\",";
    os << "\"group_count\":" << a.group_count << ",";
    os << "\"target_width\":" << a.target_width << ",";
    os << "\"target_height\":" << a.target_height << ",";
    os << "\"source_pixels_covered\":" << a.source_pixels_covered << ",";
    os << "\"output_elements\":" << a.output_elements << ",";
    os << "\"bytes_read\":" << a.bytes_read << ",";
    os << "\"bytes_written\":" << a.bytes_written << ",";
    os << "\"output_min\":" << a.output_min << ",";
    os << "\"output_max\":" << a.output_max << ",";
    os << "\"output_mean\":" << a.output_mean << ",";
    os << "\"groups\":" << mixed_region_groups_json(a.groups) << ",";
    os << "\"facts_only\":true,";
    os << "\"note\":\"Mixed region connected-component grouping and crop batching workload metrics only; no object, identity, behavior, or intent claim is emitted.\",";
    os << "\"timing\":" << stage_timing_json(a.timing);
    if (include_output) {
        os << ",\"normalized\":" << float_sample_vector_json(a.normalized);
    }
    os << "}";
    return os.str();
}

std::string mixed_region_cpu_cuda_comparison_json(const MixedRegionAnalysis& cpu, const MixedRegionAnalysis& cuda) {
    bool groups_equal = cpu.groups.size() == cuda.groups.size();
    if (groups_equal) {
        for (std::size_t i = 0; i < cpu.groups.size(); ++i) {
            const auto& a = cpu.groups[i];
            const auto& b = cuda.groups[i];
            if (a.component_index != b.component_index || a.tile_mask != b.tile_mask || a.tile_count != b.tile_count ||
                a.classification != b.classification || a.x != b.x || a.y != b.y || a.width != b.width ||
                a.height != b.height || a.tile_indices != b.tile_indices) {
                groups_equal = false;
                break;
            }
        }
    }

    bool output_close = false;
    double max_abs_diff = 0.0;
    std::size_t mismatch_count = 0;
    if (cpu.ok && cuda.ok && cpu.normalized.size() == cuda.normalized.size()) {
        output_close = true;
        for (std::size_t i = 0; i < cpu.normalized.size(); ++i) {
            const double diff = std::abs(static_cast<double>(cpu.normalized[i]) - static_cast<double>(cuda.normalized[i]));
            max_abs_diff = std::max(max_abs_diff, diff);
            if (diff > 1e-7) {
                output_close = false;
                ++mismatch_count;
            }
        }
    }
    const double output_mean_abs_diff = std::abs(cpu.output_mean - cuda.output_mean);
    const bool metrics_close = output_mean_abs_diff <= 1e-9
        && std::abs(static_cast<double>(cpu.output_min) - static_cast<double>(cuda.output_min)) <= 1e-7
        && std::abs(static_cast<double>(cpu.output_max) - static_cast<double>(cuda.output_max)) <= 1e-7
        && cpu.component_count == cuda.component_count
        && cpu.group_count == cuda.group_count
        && cpu.contiguous_components == cuda.contiguous_components
        && cpu.scattered_components == cuda.scattered_components
        && cpu.classification == cuda.classification;

    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(cpu.ok && cuda.ok && groups_equal && output_close && metrics_close) << ",";
    os << "\"schema\":\"node1_non_llm_mixed_region_cpu_cuda_compare.v0.1\",";
    os << "\"tile_mask_hex\":\"" << hex32(cpu.tile_mask) << "\",";
    os << "\"classification\":\"" << json_escape(cpu.classification) << "\",";
    os << "\"component_count\":" << cpu.component_count << ",";
    os << "\"group_count\":" << cpu.group_count << ",";
    os << "\"groups_equal\":" << bool_json(groups_equal) << ",";
    os << "\"output_close\":" << bool_json(output_close) << ",";
    os << "\"mismatch_count\":" << mismatch_count << ",";
    os << "\"max_abs_diff\":" << max_abs_diff << ",";
    os << "\"metrics_close\":" << bool_json(metrics_close) << ",";
    os << "\"output_mean_abs_diff\":" << output_mean_abs_diff << ",";
    os << "\"facts_only\":true";
    os << "}";
    return os.str();
}


std::string uint64_array_256_json(const std::array<std::uint64_t, 256>& values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) os << ",";
        os << values[i];
    }
    os << "]";
    return os.str();
}

std::string dense_full_frame_analysis_json(const DenseFullFrameAnalysis& a, bool include_output) {
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(a.ok) << ",";
    os << "\"backend\":\"" << json_escape(a.backend) << "\",";
    os << "\"schema\":\"" << json_escape(a.schema) << "\",";
    os << "\"error\":\"" << json_escape(a.error) << "\",";
    os << "\"width\":" << a.width << ",";
    os << "\"height\":" << a.height << ",";
    os << "\"pixel_threshold\":" << a.pixel_threshold << ",";
    os << "\"pixels_processed\":" << a.pixels_processed << ",";
    os << "\"changed_pixels\":" << a.changed_pixels << ",";
    os << "\"changed_ratio\":" << a.changed_ratio << ",";
    os << "\"bytes_read\":" << a.bytes_read << ",";
    os << "\"bytes_written\":" << a.bytes_written << ",";
    os << "\"diff_min\":" << a.diff_min << ",";
    os << "\"diff_max\":" << a.diff_max << ",";
    os << "\"diff_mean\":" << a.diff_mean << ",";
    os << "\"previous_mean\":" << a.previous_mean << ",";
    os << "\"current_mean\":" << a.current_mean << ",";
    os << "\"lighting_delta\":" << a.lighting_delta << ",";
    os << "\"histogram_total\":" << a.histogram_total << ",";
    os << "\"diff_histogram\":" << uint64_array_256_json(a.diff_histogram) << ",";
    os << "\"output_min\":" << a.output_min << ",";
    os << "\"output_max\":" << a.output_max << ",";
    os << "\"output_mean\":" << a.output_mean << ",";
    os << "\"facts_only\":true,";
    os << "\"note\":\"Dense full-frame diff/histogram/reduction/normalize workload metrics only; no object, identity, behavior, or intent claim is emitted.\",";
    os << "\"timing\":" << stage_timing_json(a.timing);
    if (include_output) {
        os << ",\"normalized\":" << float_sample_vector_json(a.normalized);
    }
    os << "}";
    return os.str();
}

std::string dense_full_frame_cpu_cuda_comparison_json(const DenseFullFrameAnalysis& cpu, const DenseFullFrameAnalysis& cuda) {
    bool histogram_equal = cpu.diff_histogram == cuda.diff_histogram;
    bool normalized_close = false;
    double max_abs_diff = 0.0;
    std::size_t mismatch_count = 0;
    if (cpu.ok && cuda.ok && cpu.normalized.size() == cuda.normalized.size()) {
        normalized_close = true;
        for (std::size_t i = 0; i < cpu.normalized.size(); ++i) {
            const double diff = std::abs(static_cast<double>(cpu.normalized[i]) - static_cast<double>(cuda.normalized[i]));
            max_abs_diff = std::max(max_abs_diff, diff);
            if (diff > 1e-7) {
                normalized_close = false;
                ++mismatch_count;
            }
        }
    }
    const double diff_mean_abs_diff = std::abs(cpu.diff_mean - cuda.diff_mean);
    const double previous_mean_abs_diff = std::abs(cpu.previous_mean - cuda.previous_mean);
    const double current_mean_abs_diff = std::abs(cpu.current_mean - cuda.current_mean);
    const double lighting_delta_abs_diff = std::abs(cpu.lighting_delta - cuda.lighting_delta);
    const double output_mean_abs_diff = std::abs(cpu.output_mean - cuda.output_mean);
    const bool reductions_close = diff_mean_abs_diff <= 1e-9
        && previous_mean_abs_diff <= 1e-9
        && current_mean_abs_diff <= 1e-9
        && lighting_delta_abs_diff <= 1e-9
        && output_mean_abs_diff <= 1e-9
        && std::abs(static_cast<double>(cpu.output_min) - static_cast<double>(cuda.output_min)) <= 1e-7
        && std::abs(static_cast<double>(cpu.output_max) - static_cast<double>(cuda.output_max)) <= 1e-7
        && cpu.diff_min == cuda.diff_min
        && cpu.diff_max == cuda.diff_max
        && cpu.changed_pixels == cuda.changed_pixels
        && cpu.histogram_total == cuda.histogram_total;
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(cpu.ok && cuda.ok && histogram_equal && normalized_close && reductions_close) << ",";
    os << "\"schema\":\"node1_non_llm_dense_full_frame_cpu_cuda_compare.v0.1\",";
    os << "\"histogram_equal\":" << bool_json(histogram_equal) << ",";
    os << "\"normalized_close\":" << bool_json(normalized_close) << ",";
    os << "\"mismatch_count\":" << mismatch_count << ",";
    os << "\"max_abs_diff\":" << max_abs_diff << ",";
    os << "\"reductions_close\":" << bool_json(reductions_close) << ",";
    os << "\"changed_pixels_equal\":" << bool_json(cpu.changed_pixels == cuda.changed_pixels) << ",";
    os << "\"histogram_total_equal\":" << bool_json(cpu.histogram_total == cuda.histogram_total) << ",";
    os << "\"diff_mean_abs_diff\":" << diff_mean_abs_diff << ",";
    os << "\"previous_mean_abs_diff\":" << previous_mean_abs_diff << ",";
    os << "\"current_mean_abs_diff\":" << current_mean_abs_diff << ",";
    os << "\"lighting_delta_abs_diff\":" << lighting_delta_abs_diff << ",";
    os << "\"output_mean_abs_diff\":" << output_mean_abs_diff << ",";
    os << "\"facts_only\":true";
    os << "}";
    return os.str();
}

} // namespace node1_non_llm

namespace node1_non_llm {

std::string overlay_heavy_analysis_json(const OverlayHeavyAnalysis& a, bool include_output) {
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(a.ok) << ",";
    os << "\"backend\":\"" << json_escape(a.backend) << "\",";
    os << "\"schema\":\"" << json_escape(a.schema) << "\",";
    os << "\"error\":\"" << json_escape(a.error) << "\",";
    os << "\"width\":" << a.width << ",";
    os << "\"height\":" << a.height << ",";
    os << "\"pixel_threshold\":" << a.pixel_threshold << ",";
    os << "\"alpha\":" << a.alpha << ",";
    os << "\"alpha_ratio\":" << a.alpha_ratio << ",";
    os << "\"thumbnail_width\":" << a.thumbnail_width << ",";
    os << "\"thumbnail_height\":" << a.thumbnail_height << ",";
    os << "\"pixels_processed\":" << a.pixels_processed << ",";
    os << "\"changed_pixels\":" << a.changed_pixels << ",";
    os << "\"changed_ratio\":" << a.changed_ratio << ",";
    os << "\"bytes_read\":" << a.bytes_read << ",";
    os << "\"bytes_written\":" << a.bytes_written << ",";
    os << "\"heatmap_min\":" << a.heatmap_min << ",";
    os << "\"heatmap_max\":" << a.heatmap_max << ",";
    os << "\"heatmap_mean\":" << a.heatmap_mean << ",";
    os << "\"before_after_max_diff\":" << a.before_after_max_diff << ",";
    os << "\"before_after_abs_mean\":" << a.before_after_abs_mean << ",";
    os << "\"previous_mean\":" << a.previous_mean << ",";
    os << "\"current_mean\":" << a.current_mean << ",";
    os << "\"lighting_delta\":" << a.lighting_delta << ",";
    os << "\"overlay_mean\":" << a.overlay_mean << ",";
    os << "\"thumbnail_mean\":" << a.thumbnail_mean << ",";
    os << "\"heatmap_elements\":" << a.heatmap.size() << ",";
    os << "\"overlay_rgb_elements\":" << a.overlay_rgb.size() << ",";
    os << "\"thumbnail_rgb_elements\":" << a.thumbnail_rgb.size() << ",";
    os << "\"facts_only\":true,";
    os << "\"note\":\"Overlay-heavy alpha blend, motion heatmap, thumbnail, and before/after comparison workload metrics only; no object, identity, behavior, or intent claim is emitted.\",";
    os << "\"timing\":" << stage_timing_json(a.timing);
    if (include_output) {
        os << ",\"heatmap\":" << uint8_vector_json(a.heatmap);
        os << ",\"overlay_rgb\":" << uint8_vector_json(a.overlay_rgb);
        os << ",\"thumbnail_rgb\":" << uint8_vector_json(a.thumbnail_rgb);
    }
    os << "}";
    return os.str();
}

std::string overlay_heavy_cpu_cuda_comparison_json(const OverlayHeavyAnalysis& cpu, const OverlayHeavyAnalysis& cuda) {
    const bool heatmap_equal = cpu.heatmap == cuda.heatmap;
    const bool overlay_equal = cpu.overlay_rgb == cuda.overlay_rgb;
    const bool thumbnail_equal = cpu.thumbnail_rgb == cuda.thumbnail_rgb;
    std::size_t mismatch_count = 0;
    int max_abs_diff = 0;
    auto compare_u8 = [&](const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
        if (a.size() != b.size()) {
            mismatch_count += std::max(a.size(), b.size());
            max_abs_diff = 255;
            return;
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            const int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
            if (diff != 0) {
                ++mismatch_count;
                max_abs_diff = std::max(max_abs_diff, diff);
            }
        }
    };
    compare_u8(cpu.heatmap, cuda.heatmap);
    compare_u8(cpu.overlay_rgb, cuda.overlay_rgb);
    compare_u8(cpu.thumbnail_rgb, cuda.thumbnail_rgb);

    const double heatmap_mean_abs_diff = std::abs(cpu.heatmap_mean - cuda.heatmap_mean);
    const double overlay_mean_abs_diff = std::abs(cpu.overlay_mean - cuda.overlay_mean);
    const double thumbnail_mean_abs_diff = std::abs(cpu.thumbnail_mean - cuda.thumbnail_mean);
    const double before_after_abs_mean_diff = std::abs(cpu.before_after_abs_mean - cuda.before_after_abs_mean);
    const double lighting_delta_abs_diff = std::abs(cpu.lighting_delta - cuda.lighting_delta);
    const bool metrics_close = cpu.changed_pixels == cuda.changed_pixels
        && cpu.before_after_max_diff == cuda.before_after_max_diff
        && cpu.heatmap_min == cuda.heatmap_min
        && cpu.heatmap_max == cuda.heatmap_max
        && heatmap_mean_abs_diff <= 1e-9
        && overlay_mean_abs_diff <= 1e-9
        && thumbnail_mean_abs_diff <= 1e-9
        && before_after_abs_mean_diff <= 1e-9
        && lighting_delta_abs_diff <= 1e-9;

    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(cpu.ok && cuda.ok && heatmap_equal && overlay_equal && thumbnail_equal && metrics_close) << ",";
    os << "\"schema\":\"node1_non_llm_overlay_heavy_cpu_cuda_compare.v0.1\",";
    os << "\"heatmap_equal\":" << bool_json(heatmap_equal) << ",";
    os << "\"overlay_equal\":" << bool_json(overlay_equal) << ",";
    os << "\"thumbnail_equal\":" << bool_json(thumbnail_equal) << ",";
    os << "\"mismatch_count\":" << mismatch_count << ",";
    os << "\"max_abs_diff\":" << max_abs_diff << ",";
    os << "\"metrics_close\":" << bool_json(metrics_close) << ",";
    os << "\"changed_pixels_equal\":" << bool_json(cpu.changed_pixels == cuda.changed_pixels) << ",";
    os << "\"before_after_max_diff_equal\":" << bool_json(cpu.before_after_max_diff == cuda.before_after_max_diff) << ",";
    os << "\"heatmap_mean_abs_diff\":" << heatmap_mean_abs_diff << ",";
    os << "\"overlay_mean_abs_diff\":" << overlay_mean_abs_diff << ",";
    os << "\"thumbnail_mean_abs_diff\":" << thumbnail_mean_abs_diff << ",";
    os << "\"before_after_abs_mean_diff\":" << before_after_abs_mean_diff << ",";
    os << "\"lighting_delta_abs_diff\":" << lighting_delta_abs_diff << ",";
    os << "\"facts_only\":true";
    os << "}";
    return os.str();
}

} // namespace node1_non_llm

namespace node1_non_llm {

std::string audiobox_analysis_json(const AudioBoxAnalysis& a, bool include_output) {
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(a.ok) << ",";
    os << "\"backend\":\"" << json_escape(a.backend) << "\",";
    os << "\"schema\":\"" << json_escape(a.schema) << "\",";
    os << "\"error\":\"" << json_escape(a.error) << "\",";
    os << "\"samples\":" << a.samples << ",";
    os << "\"sample_rate\":" << a.sample_rate << ",";
    os << "\"window_samples\":" << a.window_samples << ",";
    os << "\"windows\":" << a.windows << ",";
    os << "\"silence_threshold\":" << a.silence_threshold << ",";
    os << "\"onset_threshold\":" << a.onset_threshold << ",";
    os << "\"max_lag\":" << a.max_lag << ",";
    os << "\"silence_mask\":" << a.silence_mask << ",";
    os << "\"silence_mask_hex\":\"" << hex32(a.silence_mask) << "\",";
    os << "\"onset_mask\":" << a.onset_mask << ",";
    os << "\"onset_mask_hex\":\"" << hex32(a.onset_mask) << "\",";
    os << "\"silent_windows\":" << a.silent_windows << ",";
    os << "\"active_windows\":" << a.active_windows << ",";
    os << "\"onset_count\":" << a.onset_count << ",";
    os << "\"mean_rms\":" << a.mean_rms << ",";
    os << "\"max_rms\":" << a.max_rms << ",";
    os << "\"mean_peak\":" << a.mean_peak << ",";
    os << "\"max_peak\":" << a.max_peak << ",";
    os << "\"sync_drift_samples\":" << a.sync_drift_samples << ",";
    os << "\"sync_drift_ms\":" << a.sync_drift_ms << ",";
    os << "\"sync_correlation\":" << a.sync_correlation << ",";
    os << "\"sync_correlation_abs\":" << a.sync_correlation_abs << ",";
    os << "\"correlation_lag_count\":" << a.correlation_lag_count << ",";
    os << "\"bytes_read\":" << a.bytes_read << ",";
    os << "\"bytes_written\":" << a.bytes_written << ",";
    os << "\"facts_only\":true,";
    os << "\"note\":\"AudioBox RMS, peak, silence, onset, and cross-correlation sync-drift workload metrics only; no speech content, speaker identity, behavior, or intent claim is emitted.\",";
    os << "\"timing\":" << stage_timing_json(a.timing);
    os << ",\"rms\":" << float_vector_json(a.rms);
    os << ",\"peaks\":" << float_vector_json(a.peaks);
    if (include_output) {
        os << ",\"correlation_scores\":" << float_vector_json(a.correlation_scores);
    }
    os << "}";
    return os.str();
}

std::string audiobox_cpu_cuda_comparison_json(const AudioBoxAnalysis& cpu, const AudioBoxAnalysis& cuda) {
    std::size_t mismatch_count = 0;
    double max_abs_diff = 0.0;
    auto compare_float = [&](const std::vector<float>& a, const std::vector<float>& b, double tolerance) {
        bool close = a.size() == b.size();
        if (a.size() != b.size()) {
            mismatch_count += std::max(a.size(), b.size());
            max_abs_diff = std::max(max_abs_diff, 1.0);
            return false;
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            max_abs_diff = std::max(max_abs_diff, diff);
            if (diff > tolerance) {
                close = false;
                ++mismatch_count;
            }
        }
        return close;
    };
    const bool rms_close = compare_float(cpu.rms, cuda.rms, 1e-5);
    const bool peaks_close = compare_float(cpu.peaks, cuda.peaks, 1e-6);
    const bool correlation_close = compare_float(cpu.correlation_scores, cuda.correlation_scores, 1e-4);
    const double mean_rms_abs_diff = std::abs(static_cast<double>(cpu.mean_rms) - static_cast<double>(cuda.mean_rms));
    const double max_rms_abs_diff = std::abs(static_cast<double>(cpu.max_rms) - static_cast<double>(cuda.max_rms));
    const double mean_peak_abs_diff = std::abs(static_cast<double>(cpu.mean_peak) - static_cast<double>(cuda.mean_peak));
    const double max_peak_abs_diff = std::abs(static_cast<double>(cpu.max_peak) - static_cast<double>(cuda.max_peak));
    const double sync_correlation_abs_diff = std::abs(static_cast<double>(cpu.sync_correlation) - static_cast<double>(cuda.sync_correlation));
    const double sync_drift_ms_abs_diff = std::abs(cpu.sync_drift_ms - cuda.sync_drift_ms);
    const bool masks_equal = cpu.silence_mask == cuda.silence_mask && cpu.onset_mask == cuda.onset_mask;
    const bool drift_equal = cpu.sync_drift_samples == cuda.sync_drift_samples;
    const bool metrics_close = masks_equal
        && drift_equal
        && cpu.silent_windows == cuda.silent_windows
        && cpu.active_windows == cuda.active_windows
        && cpu.onset_count == cuda.onset_count
        && mean_rms_abs_diff <= 1e-5
        && max_rms_abs_diff <= 1e-5
        && mean_peak_abs_diff <= 1e-6
        && max_peak_abs_diff <= 1e-6
        && sync_correlation_abs_diff <= 1e-4
        && sync_drift_ms_abs_diff <= 1e-9;

    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(cpu.ok && cuda.ok && rms_close && peaks_close && correlation_close && metrics_close) << ",";
    os << "\"schema\":\"node1_non_llm_audiobox_cpu_cuda_compare.v0.1\",";
    os << "\"rms_close\":" << bool_json(rms_close) << ",";
    os << "\"peaks_close\":" << bool_json(peaks_close) << ",";
    os << "\"correlation_close\":" << bool_json(correlation_close) << ",";
    os << "\"masks_equal\":" << bool_json(masks_equal) << ",";
    os << "\"drift_equal\":" << bool_json(drift_equal) << ",";
    os << "\"metrics_close\":" << bool_json(metrics_close) << ",";
    os << "\"mismatch_count\":" << mismatch_count << ",";
    os << "\"max_abs_diff\":" << max_abs_diff << ",";
    os << "\"silence_mask_equal\":" << bool_json(cpu.silence_mask == cuda.silence_mask) << ",";
    os << "\"onset_mask_equal\":" << bool_json(cpu.onset_mask == cuda.onset_mask) << ",";
    os << "\"sync_drift_samples_cpu\":" << cpu.sync_drift_samples << ",";
    os << "\"sync_drift_samples_cuda\":" << cuda.sync_drift_samples << ",";
    os << "\"mean_rms_abs_diff\":" << mean_rms_abs_diff << ",";
    os << "\"max_rms_abs_diff\":" << max_rms_abs_diff << ",";
    os << "\"mean_peak_abs_diff\":" << mean_peak_abs_diff << ",";
    os << "\"max_peak_abs_diff\":" << max_peak_abs_diff << ",";
    os << "\"sync_correlation_abs_diff\":" << sync_correlation_abs_diff << ",";
    os << "\"sync_drift_ms_abs_diff\":" << sync_drift_ms_abs_diff << ",";
    os << "\"facts_only\":true";
    os << "}";
    return os.str();
}

} // namespace node1_non_llm

namespace node1_non_llm {

namespace {

std::string storage_manifest_entry_json(const StorageManifestEntry& e) {
    std::ostringstream os;
    os << "{";
    os << "\"clip_id\":\"" << json_escape(e.clip_id) << "\",";
    os << "\"path\":\"" << json_escape(e.path) << "\",";
    os << "\"start_ms\":" << e.start_ms << ",";
    os << "\"duration_ms\":" << e.duration_ms << ",";
    os << "\"bytes\":" << e.bytes << ",";
    os << "\"motion_score\":" << e.motion_score << ",";
    os << "\"audio_score\":" << e.audio_score << ",";
    os << "\"lighting_delta\":" << e.lighting_delta << ",";
    os << "\"changed_pixels\":" << e.changed_pixels << ",";
    os << "\"active_tiles\":" << e.active_tiles << ",";
    os << "\"priority_score\":" << e.priority_score;
    os << "}";
    return os.str();
}

std::string storage_manifest_json(const std::vector<StorageManifestEntry>& entries) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (i != 0) os << ",";
        os << storage_manifest_entry_json(entries[i]);
    }
    os << "]";
    return os.str();
}

std::string int_vector_json_local(const std::vector<int>& values) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) os << ",";
        os << values[i];
    }
    os << "]";
    return os.str();
}

std::string storage_batch_read_json(const StorageBatchReadPlan& b) {
    std::ostringstream os;
    os << "{";
    os << "\"batch_index\":" << b.batch_index << ",";
    os << "\"first_clip_index\":" << b.first_clip_index << ",";
    os << "\"clip_count\":" << b.clip_count << ",";
    os << "\"start_ms\":" << b.start_ms << ",";
    os << "\"end_ms\":" << b.end_ms << ",";
    os << "\"total_bytes\":" << b.total_bytes << ",";
    os << "\"clip_indices\":" << int_vector_json_local(b.clip_indices);
    os << "}";
    return os.str();
}

std::string storage_batches_json(const std::vector<StorageBatchReadPlan>& batches) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < batches.size(); ++i) {
        if (i != 0) os << ",";
        os << storage_batch_read_json(batches[i]);
    }
    os << "]";
    return os.str();
}

std::string storage_key_moment_json(const StorageKeyMoment& k) {
    std::ostringstream os;
    os << "{";
    os << "\"rank\":" << k.rank << ",";
    os << "\"clip_index\":" << k.clip_index << ",";
    os << "\"clip_id\":\"" << json_escape(k.clip_id) << "\",";
    os << "\"start_ms\":" << k.start_ms << ",";
    os << "\"duration_ms\":" << k.duration_ms << ",";
    os << "\"priority_score\":" << k.priority_score << ",";
    os << "\"motion_score\":" << k.motion_score << ",";
    os << "\"audio_score\":" << k.audio_score << ",";
    os << "\"lighting_delta\":" << k.lighting_delta << ",";
    os << "\"changed_pixels\":" << k.changed_pixels << ",";
    os << "\"reason\":\"" << json_escape(k.reason) << "\"";
    os << "}";
    return os.str();
}

std::string storage_key_moments_json(const std::vector<StorageKeyMoment>& key_moments) {
    std::ostringstream os;
    os << "[";
    for (std::size_t i = 0; i < key_moments.size(); ++i) {
        if (i != 0) os << ",";
        os << storage_key_moment_json(key_moments[i]);
    }
    os << "]";
    return os.str();
}

std::string storage_timeline_json(const StorageTimelineFeatures& t) {
    std::ostringstream os;
    os << "{";
    os << "\"clip_count\":" << t.clip_count << ",";
    os << "\"total_bytes\":" << t.total_bytes << ",";
    os << "\"timeline_start_ms\":" << t.timeline_start_ms << ",";
    os << "\"timeline_end_ms\":" << t.timeline_end_ms << ",";
    os << "\"timeline_span_ms\":" << t.timeline_span_ms << ",";
    os << "\"covered_duration_ms\":" << t.covered_duration_ms << ",";
    os << "\"max_gap_ms\":" << t.max_gap_ms << ",";
    os << "\"mean_gap_ms\":" << t.mean_gap_ms << ",";
    os << "\"mean_motion_score\":" << t.mean_motion_score << ",";
    os << "\"mean_audio_score\":" << t.mean_audio_score << ",";
    os << "\"mean_lighting_delta\":" << t.mean_lighting_delta << ",";
    os << "\"mean_priority_score\":" << t.mean_priority_score << ",";
    os << "\"max_priority_score\":" << t.max_priority_score;
    os << "}";
    return os.str();
}

} // namespace

std::string storage_batch_analysis_json(const StorageBatchAnalysis& a, bool include_manifest) {
    std::ostringstream os;
    os << "{";
    os << "\"ok\":" << bool_json(a.ok) << ",";
    os << "\"backend\":\"" << json_escape(a.backend) << "\",";
    os << "\"schema\":\"" << json_escape(a.schema) << "\",";
    os << "\"error\":\"" << json_escape(a.error) << "\",";
    os << "\"manifest_entries\":" << a.manifest_entries << ",";
    os << "\"clip_count\":" << a.clip_count << ",";
    os << "\"batch_count\":" << a.batch_count << ",";
    os << "\"key_moment_count\":" << a.key_moment_count << ",";
    os << "\"max_batch_bytes\":" << a.max_batch_bytes << ",";
    os << "\"max_batch_clips\":" << a.max_batch_clips << ",";
    os << "\"min_key_gap_ms\":" << a.min_key_gap_ms << ",";
    os << "\"total_manifest_bytes\":" << a.total_manifest_bytes << ",";
    os << "\"planned_read_bytes\":" << a.planned_read_bytes << ",";
    os << "\"manifest_sorted\":" << bool_json(a.manifest_sorted) << ",";
    os << "\"facts_only\":true,";
    os << "\"note\":\"Storage batch planning and clip timeline sampling workload metrics only; no visual, audio, identity, behavior, or intent claim is emitted.\",";
    os << "\"timeline\":" << storage_timeline_json(a.timeline) << ",";
    os << "\"batches\":" << storage_batches_json(a.batches) << ",";
    os << "\"key_moments\":" << storage_key_moments_json(a.key_moments) << ",";
    os << "\"timing\":" << stage_timing_json(a.timing);
    if (include_manifest) {
        os << ",\"manifest\":" << storage_manifest_json(a.manifest);
    }
    os << "}";
    return os.str();
}

} // namespace node1_non_llm
