#pragma once

#include "node1_non_llm/gpu_lab_timing.hpp"
#include "node1_non_llm/storage_batch.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

struct EvidenceFingerprint {
    int clip_index = 0;
    std::string clip_id;
    std::uint64_t start_ms = 0;
    std::uint64_t duration_ms = 0;
    std::uint64_t ahash64 = 0;
    std::uint64_t dhash64 = 0;
    std::uint64_t fingerprint64 = 0;
    std::string fingerprint_hex;
    int histogram_bins = 16;
    std::vector<std::uint32_t> histogram16;
    int duplicate_group = -1;
    int duplicate_of = -1;
    int nearest_hamming = 64;
    double fingerprint_score = 0.0;
};

struct EvidenceDuplicateGroup {
    int group_id = 0;
    int representative_clip_index = 0;
    std::string representative_clip_id;
    int group_size = 0;
    int duplicate_count = 0;
    int min_hamming = 0;
    int max_hamming = 0;
    std::vector<int> clip_indices;
    std::vector<std::string> clip_ids;
};

struct EvidencePipelineLatencyThroughput {
    double manifest_scan_ms = 0.0;
    double batch_plan_ms = 0.0;
    double fingerprint_ms = 0.0;
    double dedup_ms = 0.0;
    double key_selection_ms = 0.0;
    double safety_validation_ms = 0.0;
    double total_ms = 0.0;
    double clips_per_ms = 0.0;
    double planned_read_mb = 0.0;
    double planned_read_mb_per_s = 0.0;
};

struct EvidenceSafetyValidation {
    bool ok = false;
    std::string schema = "node1_non_llm_evidence_safety_validator.v0.1";
    bool facts_only = true;
    bool no_semantic_claims = true;
    bool manifest_ok = false;
    bool batch_plan_ok = false;
    bool key_moments_ok = false;
    bool fingerprint_ok = false;
    bool dedup_ok = false;
    bool timeline_ok = false;
    int checks_count = 0;
    int violation_count = 0;
    std::vector<std::string> violations;
};

struct EvidencePipelineConfig {
    StorageBatchConfig storage;
    int fingerprint_width = 16;
    int fingerprint_height = 16;
    int fingerprint_cycle = 6;
    int dedup_hamming_threshold = 0;
    bool collect_output = false;
};

struct EvidencePipelineAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    std::string schema = "node1_non_llm_evidence_pipeline.v0.1";
    std::string error;

    int manifest_entries = 0;
    int fingerprint_count = 0;
    int duplicate_group_count = 0;
    int duplicate_clip_count = 0;
    int unique_clip_count = 0;
    int key_moment_count = 0;
    int batch_count = 0;
    std::uint64_t planned_read_bytes = 0;
    std::uint64_t total_manifest_bytes = 0;
    int dedup_hamming_threshold = 0;
    int fingerprint_width = 0;
    int fingerprint_height = 0;
    int fingerprint_cycle = 0;
    bool facts_only = true;

    StorageBatchAnalysis storage_batch;
    std::vector<EvidenceFingerprint> fingerprints;
    std::vector<EvidenceDuplicateGroup> duplicate_groups;
    std::vector<StorageKeyMoment> key_moments;
    StorageTimelineFeatures timeline;
    EvidencePipelineLatencyThroughput latency;
    EvidenceSafetyValidation safety;
    StageTiming timing;
};

bool validate_evidence_pipeline_config(const EvidencePipelineConfig& cfg, std::string& error) noexcept;
EvidencePipelineAnalysis analyze_evidence_pipeline_cpu(
    const std::vector<StorageManifestEntry>& manifest,
    const EvidencePipelineConfig& cfg);

} // namespace node1_non_llm
