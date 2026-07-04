#include "node1_non_llm/gpu_lab_json.hpp"

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

} // namespace node1_non_llm
