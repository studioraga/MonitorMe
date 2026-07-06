#pragma once

#include "node1_non_llm/gpu_lab_types.hpp"
#include "node1_non_llm/isp_filters.hpp"
#include "node1_non_llm/sparse_roi.hpp"
#include "node1_non_llm/mixed_region.hpp"
#include "node1_non_llm/dense_full_frame.hpp"
#include "node1_non_llm/overlay_heavy.hpp"
#include "node1_non_llm/audiobox.hpp"
#include "node1_non_llm/storage_batch.hpp"
#include "node1_non_llm/evidence_pipeline.hpp"

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
std::string isp_filter_analysis_json(const IspFilterAnalysis& analysis, bool include_output = false);
std::string isp_cpu_cuda_comparison_json(const IspFilterAnalysis& cpu, const IspFilterAnalysis& cuda);
std::string sparse_roi_analysis_json(const SparseRoiAnalysis& analysis, bool include_output = false);
std::string sparse_roi_cpu_cuda_comparison_json(const SparseRoiAnalysis& cpu, const SparseRoiAnalysis& cuda);
std::string mixed_region_analysis_json(const MixedRegionAnalysis& analysis, bool include_output = false);
std::string mixed_region_cpu_cuda_comparison_json(const MixedRegionAnalysis& cpu, const MixedRegionAnalysis& cuda);
std::string dense_full_frame_analysis_json(const DenseFullFrameAnalysis& analysis, bool include_output = false);
std::string dense_full_frame_cpu_cuda_comparison_json(const DenseFullFrameAnalysis& cpu, const DenseFullFrameAnalysis& cuda);
std::string overlay_heavy_analysis_json(const OverlayHeavyAnalysis& analysis, bool include_output = false);
std::string overlay_heavy_cpu_cuda_comparison_json(const OverlayHeavyAnalysis& cpu, const OverlayHeavyAnalysis& cuda);
std::string audiobox_analysis_json(const AudioBoxAnalysis& analysis, bool include_output = false);
std::string audiobox_cpu_cuda_comparison_json(const AudioBoxAnalysis& cpu, const AudioBoxAnalysis& cuda);
std::string storage_batch_analysis_json(const StorageBatchAnalysis& analysis, bool include_manifest = false);
std::string evidence_pipeline_analysis_json(const EvidencePipelineAnalysis& analysis, bool include_output = false);

} // namespace node1_non_llm
