#include "node1_non_llm/evidence_pipeline.hpp"

#include "node1_non_llm/gpu_lab_timing.hpp"
#include "node1_non_llm/gpu_lab_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace node1_non_llm {

namespace {

std::uint64_t rotl64(std::uint64_t value, int shift) noexcept {
    const unsigned int s = static_cast<unsigned int>(shift & 63);
    return (value << s) | (value >> ((64U - s) & 63U));
}

std::string hex64(std::uint64_t value) {
    const char* hex = "0123456789ABCDEF";
    std::string out(18, '0');
    out[0] = '0';
    out[1] = 'x';
    for (int i = 0; i < 16; ++i) {
        const int shift = (15 - i) * 4;
        out[static_cast<std::size_t>(2 + i)] = hex[(value >> shift) & 0xFULL];
    }
    return out;
}

int hamming64(std::uint64_t a, std::uint64_t b) noexcept {
    return popcount32(static_cast<std::uint32_t>((a ^ b) & 0xFFFFFFFFULL))
        + popcount32(static_cast<std::uint32_t>(((a ^ b) >> 32U) & 0xFFFFFFFFULL));
}

int clamp_int(int value, int lo, int hi) noexcept {
    return std::max(lo, std::min(hi, value));
}

std::vector<std::uint8_t> make_visual_workload_tile(
    const StorageManifestEntry& entry,
    int clip_index,
    const EvidencePipelineConfig& cfg) {

    const int width = cfg.fingerprint_width;
    const int height = cfg.fingerprint_height;
    std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width * height), 0U);

    // The lab does not decode media in Phase 9.  It generates a deterministic
    // thumbnail-like workload vector from manifest/timeline facts.  The
    // fingerprint contract is still useful for CPU-side dedup planning and is
    // explicitly facts-only in JSON output.
    const int profile = cfg.fingerprint_cycle > 0
        ? (clip_index % cfg.fingerprint_cycle)
        : clip_index;
    const int motion_q = clamp_int(static_cast<int>(std::lround(entry.motion_score * 10.0)), 0, 10);
    const int audio_q = clamp_int(static_cast<int>(std::lround(entry.audio_score * 10.0)), 0, 10);
    const int light_q = clamp_int(static_cast<int>(std::lround(entry.lighting_delta / 8.0)), 0, 31);
    const int base = (profile * 37 + motion_q * 11 + audio_q * 7 + light_q * 3) & 0xFF;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int wave = ((x * 17 + y * 29 + (x * y + profile * 13)) & 0x7F);
            const int stripe = ((x + profile) % 5 == 0 || (y + profile) % 7 == 0) ? 41 : 0;
            pixels[static_cast<std::size_t>(y * width + x)] = static_cast<std::uint8_t>((base + wave + stripe) & 0xFF);
        }
    }
    return pixels;
}

std::uint64_t average_hash64(const std::vector<std::uint8_t>& pixels, int width, int height) {
    std::uint32_t samples[64]{};
    std::uint64_t sum = 0;
    for (int sy = 0; sy < 8; ++sy) {
        for (int sx = 0; sx < 8; ++sx) {
            const int x = std::min(width - 1, (sx * width) / 8);
            const int y = std::min(height - 1, (sy * height) / 8);
            const std::uint32_t v = pixels[static_cast<std::size_t>(y * width + x)];
            samples[sy * 8 + sx] = v;
            sum += v;
        }
    }
    const double mean = static_cast<double>(sum) / 64.0;
    std::uint64_t bits = 0;
    for (int i = 0; i < 64; ++i) {
        if (static_cast<double>(samples[i]) >= mean) {
            bits |= (1ULL << static_cast<unsigned int>(i));
        }
    }
    return bits;
}

std::uint64_t difference_hash64(const std::vector<std::uint8_t>& pixels, int width, int height) {
    std::uint64_t bits = 0;
    int bit = 0;
    for (int sy = 0; sy < 8; ++sy) {
        const int y = std::min(height - 1, (sy * height) / 8);
        for (int sx = 0; sx < 8; ++sx) {
            const int x0 = std::min(width - 1, (sx * width) / 9);
            const int x1 = std::min(width - 1, ((sx + 1) * width) / 9);
            const auto a = pixels[static_cast<std::size_t>(y * width + x0)];
            const auto b = pixels[static_cast<std::size_t>(y * width + x1)];
            if (a < b) {
                bits |= (1ULL << static_cast<unsigned int>(bit));
            }
            ++bit;
        }
    }
    return bits;
}

std::vector<std::uint32_t> histogram16(const std::vector<std::uint8_t>& pixels) {
    std::vector<std::uint32_t> hist(16, 0U);
    for (const auto v : pixels) {
        ++hist[static_cast<std::size_t>(v >> 4U)];
    }
    return hist;
}

EvidenceFingerprint make_fingerprint(
    const StorageManifestEntry& entry,
    int clip_index,
    const EvidencePipelineConfig& cfg) {

    EvidenceFingerprint fp;
    fp.clip_index = clip_index;
    fp.clip_id = entry.clip_id;
    fp.start_ms = entry.start_ms;
    fp.duration_ms = entry.duration_ms;
    fp.histogram_bins = 16;

    if (entry.has_media_fingerprint) {
        fp.from_media = true;
        fp.fingerprint_source = entry.fingerprint_source;
        fp.decoded_width = entry.decoded_width;
        fp.decoded_height = entry.decoded_height;
        fp.histogram16 = entry.media_histogram16;
        fp.ahash64 = entry.media_ahash64;
        fp.dhash64 = entry.media_dhash64;
        fp.fingerprint64 = entry.media_fingerprint64;
        fp.fingerprint_hex = hex64(fp.fingerprint64);
        const std::uint64_t hist_energy = std::accumulate(fp.histogram16.begin(), fp.histogram16.end(), std::uint64_t{0});
        fp.fingerprint_score = static_cast<double>(hist_energy) / static_cast<double>(std::max<std::uint64_t>(1ULL, hist_energy));
        return fp;
    }

    const auto pixels = make_visual_workload_tile(entry, clip_index, cfg);
    fp.from_media = false;
    fp.fingerprint_source = "metadata_synthetic";
    fp.decoded_width = cfg.fingerprint_width;
    fp.decoded_height = cfg.fingerprint_height;
    fp.histogram16 = histogram16(pixels);
    fp.ahash64 = average_hash64(pixels, cfg.fingerprint_width, cfg.fingerprint_height);
    fp.dhash64 = difference_hash64(pixels, cfg.fingerprint_width, cfg.fingerprint_height);
    fp.fingerprint64 = fp.ahash64 ^ rotl64(fp.dhash64, 17) ^ rotl64(static_cast<std::uint64_t>(clip_index % std::max(1, cfg.fingerprint_cycle)), 41);
    if (cfg.fingerprint_cycle > 0) {
        // Keep repeated synthetic visual profiles identical so the dedup path
        // can be deterministically validated without requiring real media IO.
        const int profile = clip_index % cfg.fingerprint_cycle;
        fp.fingerprint64 = fp.ahash64 ^ rotl64(fp.dhash64, 17) ^ rotl64(static_cast<std::uint64_t>(profile), 41);
    }
    fp.fingerprint_hex = hex64(fp.fingerprint64);
    const std::uint64_t hist_energy = std::accumulate(fp.histogram16.begin(), fp.histogram16.end(), std::uint64_t{0});
    fp.fingerprint_score = static_cast<double>(hist_energy) / static_cast<double>(std::max(1, cfg.fingerprint_width * cfg.fingerprint_height));
    return fp;
}

struct UnionFind {
    std::vector<int> parent;
    explicit UnionFind(int n) : parent(static_cast<std::size_t>(n), 0) {
        for (int i = 0; i < n; ++i) parent[static_cast<std::size_t>(i)] = i;
    }
    int find(int x) {
        int& p = parent[static_cast<std::size_t>(x)];
        if (p != x) p = find(p);
        return p;
    }
    void unite(int a, int b) {
        int ra = find(a);
        int rb = find(b);
        if (ra == rb) return;
        if (rb < ra) std::swap(ra, rb);
        parent[static_cast<std::size_t>(rb)] = ra;
    }
};

std::vector<EvidenceDuplicateGroup> build_dedup_groups(
    std::vector<EvidenceFingerprint>& fingerprints,
    int threshold,
    int& duplicate_clip_count) {

    const int n = static_cast<int>(fingerprints.size());
    duplicate_clip_count = 0;
    if (n == 0) return {};
    UnionFind uf(n);
    for (int i = 0; i < n; ++i) {
        int nearest = 64;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            nearest = std::min(nearest, hamming64(fingerprints[static_cast<std::size_t>(i)].fingerprint64,
                                                  fingerprints[static_cast<std::size_t>(j)].fingerprint64));
        }
        fingerprints[static_cast<std::size_t>(i)].nearest_hamming = nearest == 64 && n == 1 ? 64 : nearest;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const int distance = hamming64(fingerprints[static_cast<std::size_t>(i)].fingerprint64,
                                           fingerprints[static_cast<std::size_t>(j)].fingerprint64);
            if (distance <= threshold) {
                uf.unite(i, j);
            }
        }
    }

    std::vector<int> roots;
    for (int i = 0; i < n; ++i) {
        const int r = uf.find(i);
        if (std::find(roots.begin(), roots.end(), r) == roots.end()) {
            roots.push_back(r);
        }
    }
    std::stable_sort(roots.begin(), roots.end());

    std::vector<EvidenceDuplicateGroup> groups;
    int gid = 0;
    for (const int root : roots) {
        std::vector<int> members;
        for (int i = 0; i < n; ++i) {
            if (uf.find(i) == root) {
                members.push_back(i);
            }
        }
        if (members.size() <= 1) {
            fingerprints[static_cast<std::size_t>(members.front())].duplicate_group = -1;
            fingerprints[static_cast<std::size_t>(members.front())].duplicate_of = -1;
            continue;
        }
        EvidenceDuplicateGroup g;
        g.group_id = gid++;
        g.representative_clip_index = members.front();
        g.representative_clip_id = fingerprints[static_cast<std::size_t>(members.front())].clip_id;
        g.group_size = static_cast<int>(members.size());
        g.duplicate_count = g.group_size - 1;
        g.min_hamming = std::numeric_limits<int>::max();
        g.max_hamming = 0;
        for (const int idx : members) {
            g.clip_indices.push_back(idx);
            g.clip_ids.push_back(fingerprints[static_cast<std::size_t>(idx)].clip_id);
            fingerprints[static_cast<std::size_t>(idx)].duplicate_group = g.group_id;
            fingerprints[static_cast<std::size_t>(idx)].duplicate_of = idx == members.front() ? -1 : members.front();
        }
        for (std::size_t a = 0; a < members.size(); ++a) {
            for (std::size_t b = a + 1; b < members.size(); ++b) {
                const int d = hamming64(fingerprints[static_cast<std::size_t>(members[a])].fingerprint64,
                                        fingerprints[static_cast<std::size_t>(members[b])].fingerprint64);
                g.min_hamming = std::min(g.min_hamming, d);
                g.max_hamming = std::max(g.max_hamming, d);
            }
        }
        if (g.min_hamming == std::numeric_limits<int>::max()) g.min_hamming = 0;
        duplicate_clip_count += g.duplicate_count;
        groups.push_back(g);
    }
    return groups;
}

std::vector<StorageKeyMoment> select_dedup_key_moments(
    const StorageBatchAnalysis& storage,
    const std::vector<EvidenceFingerprint>& fingerprints,
    const EvidencePipelineConfig& cfg) {

    std::vector<int> order(storage.manifest.size());
    for (std::size_t i = 0; i < storage.manifest.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const auto& ea = storage.manifest[static_cast<std::size_t>(a)];
        const auto& eb = storage.manifest[static_cast<std::size_t>(b)];
        if (ea.priority_score != eb.priority_score) return ea.priority_score > eb.priority_score;
        if (ea.start_ms != eb.start_ms) return ea.start_ms < eb.start_ms;
        return ea.clip_id < eb.clip_id;
    });

    std::vector<StorageKeyMoment> out;
    std::vector<int> selected_groups;
    for (const int idx : order) {
        if (static_cast<int>(out.size()) >= cfg.storage.key_moments) break;
        const auto& candidate = storage.manifest[static_cast<std::size_t>(idx)];
        const int group_id = fingerprints[static_cast<std::size_t>(idx)].duplicate_group;
        if (group_id >= 0 && std::find(selected_groups.begin(), selected_groups.end(), group_id) != selected_groups.end()) {
            continue;
        }
        bool far_enough = true;
        for (const auto& selected : out) {
            const auto a = static_cast<std::int64_t>(candidate.start_ms);
            const auto b = static_cast<std::int64_t>(selected.start_ms);
            if (std::llabs(a - b) < static_cast<std::int64_t>(cfg.storage.min_key_gap_ms)) {
                far_enough = false;
                break;
            }
        }
        if (!far_enough) continue;
        StorageKeyMoment m;
        m.rank = static_cast<int>(out.size()) + 1;
        m.clip_index = idx;
        m.clip_id = candidate.clip_id;
        m.start_ms = candidate.start_ms;
        m.duration_ms = candidate.duration_ms;
        m.priority_score = candidate.priority_score;
        m.motion_score = candidate.motion_score;
        m.audio_score = candidate.audio_score;
        m.lighting_delta = candidate.lighting_delta;
        m.changed_pixels = candidate.changed_pixels;
        if (group_id >= 0) {
            m.reason = "dedup_representative";
            selected_groups.push_back(group_id);
        } else if (candidate.motion_score >= 0.8 && candidate.audio_score >= 0.5) {
            m.reason = "motion_audio_peak";
        } else if (candidate.motion_score >= 0.8) {
            m.reason = "motion_peak";
        } else if (candidate.audio_score >= 0.5) {
            m.reason = "audio_peak";
        } else if (candidate.lighting_delta >= 40.0) {
            m.reason = "lighting_delta";
        } else {
            m.reason = "priority_score";
        }
        out.push_back(m);
    }
    return out;
}

void add_violation(EvidenceSafetyValidation& s, const std::string& violation) {
    s.violations.push_back(violation);
}

EvidenceSafetyValidation validate_evidence_safety(
    const EvidencePipelineAnalysis& out,
    const EvidencePipelineConfig& cfg) {

    EvidenceSafetyValidation s;
    s.facts_only = true;
    s.no_semantic_claims = true;

    ++s.checks_count;
    s.manifest_ok = out.manifest_entries > 0 && out.storage_batch.manifest_entries == out.manifest_entries;
    if (!s.manifest_ok) add_violation(s, "manifest_count_mismatch_or_empty");

    ++s.checks_count;
    s.batch_plan_ok = out.storage_batch.ok && out.storage_batch.planned_read_bytes == out.storage_batch.total_manifest_bytes;
    for (const auto& b : out.storage_batch.batches) {
        if (b.clip_count <= 0 || b.clip_count > cfg.storage.max_batch_clips) {
            s.batch_plan_ok = false;
        }
        if (b.total_bytes > cfg.storage.max_batch_bytes && b.clip_count > 1) {
            s.batch_plan_ok = false;
        }
    }
    if (!s.batch_plan_ok) add_violation(s, "batch_plan_constraints_failed");

    ++s.checks_count;
    s.fingerprint_ok = out.fingerprint_count == out.manifest_entries
        && out.media_fingerprint_count + out.synthetic_fingerprint_count == out.fingerprint_count;
    for (const auto& fp : out.fingerprints) {
        if (fp.histogram16.size() != 16 || fp.fingerprint_hex.size() != 18) {
            s.fingerprint_ok = false;
        }
        if (fp.from_media && (fp.fingerprint_source != "decoded_keyframe" || fp.decoded_width <= 0 || fp.decoded_height <= 0)) {
            s.fingerprint_ok = false;
        }
    }
    if (!s.fingerprint_ok) add_violation(s, "fingerprint_contract_failed");

    ++s.checks_count;
    s.dedup_ok = true;
    int counted_duplicates = 0;
    for (const auto& g : out.duplicate_groups) {
        if (g.group_size <= 1 || g.duplicate_count != g.group_size - 1 || g.clip_indices.empty()) {
            s.dedup_ok = false;
        }
        counted_duplicates += g.duplicate_count;
    }
    if (counted_duplicates != out.duplicate_clip_count) {
        s.dedup_ok = false;
    }
    if (!s.dedup_ok) add_violation(s, "dedup_group_contract_failed");

    ++s.checks_count;
    s.key_moments_ok = out.key_moment_count <= cfg.storage.key_moments;
    for (std::size_t i = 1; i < out.key_moments.size(); ++i) {
        const auto a = static_cast<std::int64_t>(out.key_moments[i].start_ms);
        for (std::size_t j = 0; j < i; ++j) {
            const auto b = static_cast<std::int64_t>(out.key_moments[j].start_ms);
            if (std::llabs(a - b) < static_cast<std::int64_t>(cfg.storage.min_key_gap_ms)) {
                s.key_moments_ok = false;
            }
        }
    }
    if (!s.key_moments_ok) add_violation(s, "key_moment_gap_or_count_failed");

    ++s.checks_count;
    s.timeline_ok = out.timeline.clip_count == out.manifest_entries && out.timeline.total_bytes == out.total_manifest_bytes;
    if (!s.timeline_ok) add_violation(s, "timeline_summary_mismatch");

    s.violation_count = static_cast<int>(s.violations.size());
    s.ok = s.violation_count == 0;
    return s;
}

} // namespace

bool validate_evidence_pipeline_config(const EvidencePipelineConfig& cfg, std::string& error) noexcept {
    if (!validate_storage_batch_config(cfg.storage, error)) {
        return false;
    }
    if (cfg.fingerprint_width < 8 || cfg.fingerprint_width > 128 || cfg.fingerprint_height < 8 || cfg.fingerprint_height > 128) {
        error = "fingerprint dimensions must be in [8, 128]";
        return false;
    }
    if (cfg.fingerprint_cycle < 0 || cfg.fingerprint_cycle > 1024) {
        error = "fingerprint_cycle must be in [0, 1024]";
        return false;
    }
    if (cfg.dedup_hamming_threshold < 0 || cfg.dedup_hamming_threshold > 64) {
        error = "dedup_hamming_threshold must be in [0, 64]";
        return false;
    }
    return true;
}

EvidencePipelineAnalysis analyze_evidence_pipeline_cpu(
    const std::vector<StorageManifestEntry>& manifest,
    const EvidencePipelineConfig& cfg) {

    HostStageTimer total_timer;
    EvidencePipelineAnalysis out;
    out.backend = "cpu";
    out.dedup_hamming_threshold = cfg.dedup_hamming_threshold;
    out.fingerprint_width = cfg.fingerprint_width;
    out.fingerprint_height = cfg.fingerprint_height;
    out.fingerprint_cycle = cfg.fingerprint_cycle;

    std::string error;
    if (!validate_evidence_pipeline_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (manifest.empty()) {
        out.error = "manifest must contain at least one evidence clip";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    HostStageTimer batch_timer;
    out.storage_batch = analyze_storage_batch_cpu(manifest, cfg.storage);
    out.latency.batch_plan_ms = batch_timer.elapsed_ms();
    if (!out.storage_batch.ok) {
        out.error = out.storage_batch.error;
        out.timing.total_ms = total_timer.elapsed_ms();
        out.latency.total_ms = out.timing.total_ms;
        return out;
    }

    out.manifest_entries = out.storage_batch.manifest_entries;
    out.batch_count = out.storage_batch.batch_count;
    out.planned_read_bytes = out.storage_batch.planned_read_bytes;
    out.total_manifest_bytes = out.storage_batch.total_manifest_bytes;
    out.timeline = out.storage_batch.timeline;

    HostStageTimer fingerprint_timer;
    out.fingerprints.reserve(out.storage_batch.manifest.size());
    for (std::size_t i = 0; i < out.storage_batch.manifest.size(); ++i) {
        out.fingerprints.push_back(make_fingerprint(out.storage_batch.manifest[i], static_cast<int>(i), cfg));
    }
    out.fingerprint_count = static_cast<int>(out.fingerprints.size());
    for (const auto& fp : out.fingerprints) {
        if (fp.from_media) {
            ++out.media_fingerprint_count;
        } else {
            ++out.synthetic_fingerprint_count;
        }
    }
    out.latency.fingerprint_ms = fingerprint_timer.elapsed_ms();

    HostStageTimer dedup_timer;
    out.duplicate_groups = build_dedup_groups(out.fingerprints, cfg.dedup_hamming_threshold, out.duplicate_clip_count);
    out.duplicate_group_count = static_cast<int>(out.duplicate_groups.size());
    out.unique_clip_count = out.fingerprint_count - out.duplicate_clip_count;
    out.latency.dedup_ms = dedup_timer.elapsed_ms();

    HostStageTimer key_timer;
    out.key_moments = select_dedup_key_moments(out.storage_batch, out.fingerprints, cfg);
    out.key_moment_count = static_cast<int>(out.key_moments.size());
    out.latency.key_selection_ms = key_timer.elapsed_ms();

    HostStageTimer safety_timer;
    out.safety = validate_evidence_safety(out, cfg);
    out.latency.safety_validation_ms = safety_timer.elapsed_ms();

    out.latency.total_ms = total_timer.elapsed_ms();
    out.timing.kernel_ms = out.latency.batch_plan_ms + out.latency.fingerprint_ms + out.latency.dedup_ms + out.latency.key_selection_ms + out.latency.safety_validation_ms;
    out.timing.total_ms = out.latency.total_ms;
    out.latency.planned_read_mb = static_cast<double>(out.planned_read_bytes) / (1024.0 * 1024.0);
    if (out.latency.total_ms > 0.0) {
        out.latency.clips_per_ms = static_cast<double>(out.manifest_entries) / out.latency.total_ms;
        out.latency.planned_read_mb_per_s = out.latency.planned_read_mb / (out.latency.total_ms / 1000.0);
    }
    out.ok = out.safety.ok;
    if (!out.ok) {
        out.error = "evidence safety validation failed";
    }
    return out;
}

} // namespace node1_non_llm
