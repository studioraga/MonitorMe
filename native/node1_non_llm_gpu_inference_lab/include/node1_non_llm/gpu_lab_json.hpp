#pragma once

#include "node1_non_llm/gpu_lab_types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

std::string json_escape(const std::string& value);
const char* bool_json(bool value) noexcept;
std::string stage_timing_json(const StageTiming& timing);
std::string uint32_vector_json(const std::vector<std::uint32_t>& values);
std::string float_vector_json(const std::vector<float>& values);
std::string frame_analysis_json(const FrameAnalysis& analysis);
std::string audio_analysis_json(const AudioEnergyAnalysis& analysis);

} // namespace node1_non_llm
