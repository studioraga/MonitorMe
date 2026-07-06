#pragma once

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

struct StorageManifestEntry {
    std::string clip_id;
    std::string path;
    std::uint64_t start_ms = 0;
    std::uint64_t duration_ms = 0;
    std::uint64_t bytes = 0;
    double motion_score = 0.0;
    double audio_score = 0.0;
    double lighting_delta = 0.0;
    std::uint64_t changed_pixels = 0;
    int active_tiles = 0;
    double priority_score = 0.0;

    // Phase 11: optional real media fingerprint facts. These are populated by
    // Python after decoded keyframes are explicitly routed into the evidence
    // pipeline. Native code consumes them as facts-only workload metadata and
    // falls back to deterministic synthetic fingerprints when absent.
    bool has_media_fingerprint = false;
    std::uint64_t media_ahash64 = 0;
    std::uint64_t media_dhash64 = 0;
    std::uint64_t media_fingerprint64 = 0;
    std::vector<std::uint32_t> media_histogram16;
    int decoded_width = 0;
    int decoded_height = 0;
    std::string fingerprint_source = "metadata_synthetic";
};

struct StorageBatchReadPlan {
    int batch_index = 0;
    int first_clip_index = 0;
    int clip_count = 0;
    std::uint64_t start_ms = 0;
    std::uint64_t end_ms = 0;
    std::uint64_t total_bytes = 0;
    std::vector<int> clip_indices;
};

struct StorageKeyMoment {
    int rank = 0;
    int clip_index = 0;
    std::string clip_id;
    std::uint64_t start_ms = 0;
    std::uint64_t duration_ms = 0;
    double priority_score = 0.0;
    double motion_score = 0.0;
    double audio_score = 0.0;
    double lighting_delta = 0.0;
    std::uint64_t changed_pixels = 0;
    std::string reason;
};

struct StorageTimelineFeatures {
    int clip_count = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t timeline_start_ms = 0;
    std::uint64_t timeline_end_ms = 0;
    std::uint64_t timeline_span_ms = 0;
    std::uint64_t covered_duration_ms = 0;
    std::uint64_t max_gap_ms = 0;
    double mean_gap_ms = 0.0;
    double mean_motion_score = 0.0;
    double mean_audio_score = 0.0;
    double mean_lighting_delta = 0.0;
    double mean_priority_score = 0.0;
    double max_priority_score = 0.0;
};

struct StorageBatchConfig {
    std::uint64_t max_batch_bytes = 4ULL * 1024ULL * 1024ULL;
    int max_batch_clips = 4;
    int key_moments = 5;
    std::uint64_t min_key_gap_ms = 1000;
    bool collect_manifest = false;
};

struct StorageBatchAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    std::string schema = "node1_non_llm_storage_batch.v0.1";
    std::string error;

    int manifest_entries = 0;
    int clip_count = 0;
    int batch_count = 0;
    int key_moment_count = 0;
    std::uint64_t max_batch_bytes = 0;
    int max_batch_clips = 0;
    std::uint64_t min_key_gap_ms = 0;
    std::uint64_t total_manifest_bytes = 0;
    std::uint64_t planned_read_bytes = 0;
    bool manifest_sorted = false;
    bool facts_only = true;

    std::vector<StorageManifestEntry> manifest;
    std::vector<StorageBatchReadPlan> batches;
    std::vector<StorageKeyMoment> key_moments;
    StorageTimelineFeatures timeline;
    StageTiming timing;
};

bool validate_storage_batch_config(const StorageBatchConfig& cfg, std::string& error) noexcept;
std::vector<StorageManifestEntry> make_synthetic_storage_manifest(int clips = 12);
std::vector<StorageManifestEntry> scan_storage_manifest_csv(const std::string& path);
StorageBatchAnalysis analyze_storage_batch_cpu(
    const std::vector<StorageManifestEntry>& manifest,
    const StorageBatchConfig& cfg);

} // namespace node1_non_llm
