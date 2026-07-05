#include "node1_non_llm/gpu_lab_json.hpp"

#include <algorithm>
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

} // namespace node1_non_llm
