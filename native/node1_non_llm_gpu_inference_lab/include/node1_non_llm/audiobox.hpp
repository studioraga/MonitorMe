#pragma once

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

struct AudioBoxConfig {
    int sample_count = 0;
    int sample_rate = 48000;
    int window_samples = 1024;
    float silence_threshold = 0.02f;
    float onset_threshold = 0.08f;
    int max_windows = 32;
    int max_lag = 128;
    bool collect_output = false;
};

struct AudioBoxAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    std::string schema = "node1_non_llm_audiobox.v0.1";
    std::string error;

    int samples = 0;
    int sample_rate = 48000;
    int window_samples = 1024;
    int windows = 0;
    float silence_threshold = 0.02f;
    float onset_threshold = 0.08f;
    int max_lag = 128;

    std::uint32_t silence_mask = 0;
    std::uint32_t onset_mask = 0;
    int silent_windows = 0;
    int active_windows = 0;
    int onset_count = 0;

    float mean_rms = 0.0f;
    float max_rms = 0.0f;
    float max_peak = 0.0f;
    float mean_peak = 0.0f;

    int sync_drift_samples = 0;
    double sync_drift_ms = 0.0;
    float sync_correlation = 0.0f;
    float sync_correlation_abs = 0.0f;
    int correlation_lag_count = 0;

    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_written = 0;

    std::vector<float> rms;
    std::vector<float> peaks;
    std::vector<float> correlation_scores;

    StageTiming timing;
};

bool validate_audiobox_config(const AudioBoxConfig& cfg, std::string& error) noexcept;
AudioBoxAnalysis analyze_audiobox_cpu(
    const float* primary_samples,
    const float* reference_samples,
    const AudioBoxConfig& cfg);

#ifdef NODE1_NON_LLM_WITH_CUDA
AudioBoxAnalysis analyze_audiobox_cuda(
    const float* primary_samples,
    const float* reference_samples,
    const AudioBoxConfig& cfg);
#endif

} // namespace node1_non_llm
