#include "node1_non_llm/audiobox.hpp"

#include "node1_non_llm/gpu_lab_types.hpp"
#include "node1_non_llm/gpu_lab_timing.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace node1_non_llm {

bool validate_audiobox_config(const AudioBoxConfig& cfg, std::string& error) noexcept {
    if (cfg.sample_count <= 0) {
        error = "sample_count must be positive";
        return false;
    }
    if (cfg.sample_rate <= 0) {
        error = "sample_rate must be positive";
        return false;
    }
    if (cfg.window_samples <= 0) {
        error = "window_samples must be positive";
        return false;
    }
    if (cfg.max_windows <= 0 || cfg.max_windows > 32) {
        error = "max_windows must be in [1, 32] because masks are uint32_t";
        return false;
    }
    if (cfg.silence_threshold < 0.0f) {
        error = "silence_threshold must be non-negative";
        return false;
    }
    if (cfg.onset_threshold < 0.0f) {
        error = "onset_threshold must be non-negative";
        return false;
    }
    if (cfg.max_lag < 0) {
        error = "max_lag must be non-negative";
        return false;
    }
    if (cfg.max_lag >= cfg.sample_count) {
        error = "max_lag must be smaller than sample_count";
        return false;
    }
    if ((2 * cfg.max_lag + 1) > 4097) {
        error = "2 * max_lag + 1 must be <= 4097 for bounded validation output";
        return false;
    }
    return true;
}

namespace {

void finalize_audiobox_analysis(AudioBoxAnalysis& out, const AudioBoxConfig& cfg) {
    out.windows = static_cast<int>(out.rms.size());
    out.silence_mask = 0U;
    out.onset_mask = 0U;
    out.silent_windows = 0;
    out.active_windows = 0;
    out.onset_count = 0;
    out.mean_rms = 0.0f;
    out.mean_peak = 0.0f;
    out.max_rms = 0.0f;
    out.max_peak = 0.0f;

    for (int w = 0; w < out.windows; ++w) {
        const float rms = out.rms[static_cast<std::size_t>(w)];
        const float peak = out.peaks[static_cast<std::size_t>(w)];
        out.mean_rms += rms;
        out.mean_peak += peak;
        out.max_rms = std::max(out.max_rms, rms);
        out.max_peak = std::max(out.max_peak, peak);
        if (rms <= cfg.silence_threshold) {
            out.silence_mask |= (1U << w);
            ++out.silent_windows;
        } else {
            ++out.active_windows;
        }
        const float previous = (w == 0) ? 0.0f : out.rms[static_cast<std::size_t>(w - 1)];
        if (rms > cfg.silence_threshold && (w == 0 ? rms >= cfg.onset_threshold : (rms - previous) >= cfg.onset_threshold)) {
            out.onset_mask |= (1U << w);
            ++out.onset_count;
        }
    }
    if (out.windows > 0) {
        out.mean_rms /= static_cast<float>(out.windows);
        out.mean_peak /= static_cast<float>(out.windows);
    }

    int best_lag = 0;
    float best_abs = -1.0f;
    float best_corr = 0.0f;
    for (std::size_t i = 0; i < out.correlation_scores.size(); ++i) {
        const float corr = out.correlation_scores[i];
        const float abs_corr = std::fabs(corr);
        const int lag = static_cast<int>(i) - cfg.max_lag;
        if (abs_corr > best_abs || (abs_corr == best_abs && std::abs(lag) < std::abs(best_lag))) {
            best_abs = abs_corr;
            best_corr = corr;
            best_lag = lag;
        }
    }
    out.sync_drift_samples = best_lag;
    out.sync_drift_ms = 1000.0 * static_cast<double>(best_lag) / static_cast<double>(std::max(1, cfg.sample_rate));
    out.sync_correlation = best_corr;
    out.sync_correlation_abs = std::fabs(best_corr);
    out.correlation_lag_count = static_cast<int>(out.correlation_scores.size());
}

} // namespace

AudioBoxAnalysis analyze_audiobox_cpu(
    const float* primary_samples,
    const float* reference_samples,
    const AudioBoxConfig& cfg) {

    HostStageTimer total_timer;

    AudioBoxAnalysis out;
    out.backend = "cpu";
    out.samples = cfg.sample_count;
    out.sample_rate = cfg.sample_rate;
    out.window_samples = cfg.window_samples;
    out.silence_threshold = cfg.silence_threshold;
    out.onset_threshold = cfg.onset_threshold;
    out.max_lag = cfg.max_lag;
    out.bytes_read = static_cast<std::uint64_t>(cfg.sample_count) * sizeof(float) * 2ULL;

    std::string error;
    if (!validate_audiobox_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (primary_samples == nullptr || reference_samples == nullptr) {
        out.error = "primary_samples and reference_samples must not be null";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    HostStageTimer kernel_timer;
    const int windows = std::min(cfg.max_windows, (cfg.sample_count + cfg.window_samples - 1) / cfg.window_samples);
    out.rms.assign(static_cast<std::size_t>(windows), 0.0f);
    out.peaks.assign(static_cast<std::size_t>(windows), 0.0f);

    for (int w = 0; w < windows; ++w) {
        const int start = w * cfg.window_samples;
        const int end = std::min(cfg.sample_count, start + cfg.window_samples);
        double sum_squares = 0.0;
        float peak = 0.0f;
        for (int i = start; i < end; ++i) {
            const float value = primary_samples[i];
            const float abs_value = std::fabs(value);
            sum_squares += static_cast<double>(value) * static_cast<double>(value);
            peak = std::max(peak, abs_value);
        }
        const int n = std::max(1, end - start);
        out.rms[static_cast<std::size_t>(w)] = static_cast<float>(std::sqrt(sum_squares / static_cast<double>(n)));
        out.peaks[static_cast<std::size_t>(w)] = peak;
    }

    out.correlation_scores.assign(static_cast<std::size_t>(2 * cfg.max_lag + 1), 0.0f);
    for (int lag = -cfg.max_lag; lag <= cfg.max_lag; ++lag) {
        double sum = 0.0;
        double primary_energy = 0.0;
        double reference_energy = 0.0;
        for (int i = 0; i < cfg.sample_count; ++i) {
            const int j = i + lag;
            if (j < 0 || j >= cfg.sample_count) {
                continue;
            }
            const double a = static_cast<double>(primary_samples[i]);
            const double b = static_cast<double>(reference_samples[j]);
            sum += a * b;
            primary_energy += a * a;
            reference_energy += b * b;
        }
        const double denom = std::sqrt(primary_energy * reference_energy);
        const float corr = denom > std::numeric_limits<double>::epsilon()
            ? static_cast<float>(sum / denom)
            : 0.0f;
        out.correlation_scores[static_cast<std::size_t>(lag + cfg.max_lag)] = corr;
    }
    out.timing.kernel_ms = kernel_timer.elapsed_ms();

    finalize_audiobox_analysis(out, cfg);
    out.bytes_written = static_cast<std::uint64_t>(out.rms.size() + out.peaks.size() + out.correlation_scores.size()) * sizeof(float);
    out.ok = true;
    out.timing.total_ms = total_timer.elapsed_ms();
    return out;
}

} // namespace node1_non_llm
