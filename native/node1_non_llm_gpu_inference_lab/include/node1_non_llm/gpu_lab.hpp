#pragma once

#include "node1_non_llm/gpu_lab_types.hpp"
#include "node1_non_llm/isp_filters.hpp"
#include "node1_non_llm/sparse_roi.hpp"

namespace node1_non_llm {

FrameAnalysis analyze_gray_frames_cpu(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const TileAnalysisConfig& cfg);

AudioEnergyAnalysis analyze_audio_energy_cpu(
    const float* samples,
    int sample_count,
    const AudioEnergyConfig& cfg);

#ifdef NODE1_NON_LLM_WITH_CUDA
FrameAnalysis analyze_gray_frames_cuda(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const TileAnalysisConfig& cfg);

AudioEnergyAnalysis analyze_audio_energy_cuda(
    const float* samples,
    int sample_count,
    const AudioEnergyConfig& cfg);

IspFilterAnalysis analyze_isp_filter_cuda(
    const std::uint8_t* gray,
    const IspFilterConfig& cfg);

IspFilterAnalysis apply_isp_filter_cuda_tiled(
    const std::uint8_t* gray,
    const IspFilterConfig& cfg);

SparseRoiAnalysis analyze_sparse_roi_cuda(
    const std::uint8_t* gray,
    const SparseRoiConfig& cfg);
#endif

} // namespace node1_non_llm
