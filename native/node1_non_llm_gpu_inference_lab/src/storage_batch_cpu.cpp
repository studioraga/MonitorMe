#include "node1_non_llm/storage_batch.hpp"

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace node1_non_llm {

namespace {

std::string trim_copy(const std::string& value) {
    std::size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first])) != 0) {
        ++first;
    }
    std::size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1])) != 0) {
        --last;
    }
    return value.substr(first, last - first);
}

std::vector<std::string> split_csv_simple(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, ',')) {
        out.push_back(trim_copy(field));
    }
    return out;
}

bool is_header_line(const std::vector<std::string>& fields) {
    return !fields.empty() && fields[0] == "clip_id";
}

std::string lower_copy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::map<std::string, std::size_t> header_index(const std::vector<std::string>& header) {
    std::map<std::string, std::size_t> out;
    for (std::size_t i = 0; i < header.size(); ++i) {
        out[lower_copy(trim_copy(header[i]))] = i;
    }
    return out;
}

std::string get_field(
    const std::vector<std::string>& fields,
    const std::map<std::string, std::size_t>& index,
    const std::string& name,
    const std::string& fallback = "") {
    const auto it = index.find(name);
    if (it == index.end() || it->second >= fields.size()) {
        return fallback;
    }
    return fields[it->second];
}

bool has_field(
    const std::vector<std::string>& fields,
    const std::map<std::string, std::size_t>& index,
    const std::string& name) {
    const auto it = index.find(name);
    return it != index.end() && it->second < fields.size() && !trim_copy(fields[it->second]).empty();
}

std::uint64_t parse_u64_flexible(const std::string& text) {
    const std::string value = trim_copy(text);
    if (value.empty()) return 0;
    std::size_t consumed = 0;
    const int base = (value.rfind("0x", 0) == 0 || value.rfind("0X", 0) == 0) ? 16 : 10;
    const auto parsed = std::stoull(value, &consumed, base);
    if (consumed != value.size()) {
        throw std::runtime_error("invalid uint64 value: " + value);
    }
    return static_cast<std::uint64_t>(parsed);
}

std::vector<std::uint32_t> parse_histogram16(const std::string& text) {
    std::vector<std::uint32_t> hist;
    std::string cleaned = text;
    for (char& c : cleaned) {
        if (c == '|' || c == ';') c = ',';
    }
    std::stringstream ss(cleaned);
    std::string field;
    while (std::getline(ss, field, ',')) {
        const auto trimmed = trim_copy(field);
        if (trimmed.empty()) continue;
        hist.push_back(static_cast<std::uint32_t>(std::stoul(trimmed)));
    }
    return hist;
}

double clamp01(double value) {
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

double compute_priority_score(const StorageManifestEntry& e) {
    const double changed_component = clamp01(static_cast<double>(e.changed_pixels) / 100000.0) * 0.25;
    const double tile_component = clamp01(static_cast<double>(std::max(0, e.active_tiles)) / 32.0) * 0.15;
    const double lighting_component = clamp01(e.lighting_delta / 255.0) * 0.15;
    return e.motion_score + 0.5 * e.audio_score + changed_component + tile_component + lighting_component;
}

std::string key_reason(const StorageManifestEntry& e) {
    if (e.motion_score >= 0.8 && e.audio_score >= 0.5) {
        return "motion_audio_peak";
    }
    if (e.motion_score >= 0.8) {
        return "motion_peak";
    }
    if (e.audio_score >= 0.5) {
        return "audio_peak";
    }
    if (e.lighting_delta >= 40.0) {
        return "lighting_delta";
    }
    return "priority_score";
}

std::vector<int> timeline_order(const std::vector<StorageManifestEntry>& entries) {
    std::vector<int> order(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        order[i] = static_cast<int>(i);
    }
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const auto& ea = entries[static_cast<std::size_t>(a)];
        const auto& eb = entries[static_cast<std::size_t>(b)];
        if (ea.start_ms != eb.start_ms) return ea.start_ms < eb.start_ms;
        return ea.clip_id < eb.clip_id;
    });
    return order;
}

bool order_is_sorted(const std::vector<StorageManifestEntry>& entries) {
    for (std::size_t i = 1; i < entries.size(); ++i) {
        if (entries[i - 1].start_ms > entries[i].start_ms) {
            return false;
        }
    }
    return true;
}

void finalize_batch(
    StorageBatchAnalysis& out,
    StorageBatchReadPlan& batch,
    const std::vector<StorageManifestEntry>& entries) {

    if (batch.clip_indices.empty()) {
        return;
    }
    batch.batch_index = static_cast<int>(out.batches.size());
    batch.first_clip_index = batch.clip_indices.front();
    batch.clip_count = static_cast<int>(batch.clip_indices.size());
    batch.start_ms = std::numeric_limits<std::uint64_t>::max();
    batch.end_ms = 0;
    batch.total_bytes = 0;
    for (const int idx : batch.clip_indices) {
        const auto& e = entries[static_cast<std::size_t>(idx)];
        batch.start_ms = std::min(batch.start_ms, e.start_ms);
        batch.end_ms = std::max(batch.end_ms, e.start_ms + e.duration_ms);
        batch.total_bytes += e.bytes;
    }
    out.planned_read_bytes += batch.total_bytes;
    out.batches.push_back(batch);
    batch = StorageBatchReadPlan{};
}

} // namespace

bool validate_storage_batch_config(const StorageBatchConfig& cfg, std::string& error) noexcept {
    if (cfg.max_batch_bytes == 0) {
        error = "max_batch_bytes must be positive";
        return false;
    }
    if (cfg.max_batch_clips <= 0 || cfg.max_batch_clips > 1024) {
        error = "max_batch_clips must be in [1, 1024]";
        return false;
    }
    if (cfg.key_moments < 0 || cfg.key_moments > 1024) {
        error = "key_moments must be in [0, 1024]";
        return false;
    }
    return true;
}

std::vector<StorageManifestEntry> make_synthetic_storage_manifest(int clips) {
    if (clips <= 0) {
        throw std::runtime_error("synthetic storage clip count must be positive");
    }
    std::vector<StorageManifestEntry> out;
    out.reserve(static_cast<std::size_t>(clips));
    for (int i = 0; i < clips; ++i) {
        StorageManifestEntry e;
        e.clip_id = "clip_" + std::to_string(i);
        e.path = "clips/session_a/clip_" + std::to_string(i) + ".mkv";
        e.start_ms = static_cast<std::uint64_t>(i) * 1250ULL;
        e.duration_ms = 1000ULL + static_cast<std::uint64_t>((i % 3) * 250);
        e.bytes = 700000ULL + static_cast<std::uint64_t>((i % 5) * 180000ULL);
        e.motion_score = 0.10 + 0.05 * static_cast<double>(i % 4);
        e.audio_score = 0.03 * static_cast<double>((i + 1) % 5);
        e.lighting_delta = 4.0 * static_cast<double>(i % 6);
        e.changed_pixels = static_cast<std::uint64_t>(400 + (i % 7) * 850);
        e.active_tiles = 1 + (i % 9);
        if (i == 3) {
            e.motion_score = 0.94;
            e.audio_score = 0.62;
            e.lighting_delta = 58.0;
            e.changed_pixels = 42000;
            e.active_tiles = 18;
        } else if (i == 7) {
            e.motion_score = 0.87;
            e.audio_score = 0.18;
            e.lighting_delta = 22.0;
            e.changed_pixels = 35000;
            e.active_tiles = 14;
        } else if (i == 10) {
            e.motion_score = 0.48;
            e.audio_score = 0.77;
            e.lighting_delta = 12.0;
            e.changed_pixels = 11000;
            e.active_tiles = 6;
        }
        e.priority_score = compute_priority_score(e);
        out.push_back(e);
    }
    return out;
}

std::vector<StorageManifestEntry> scan_storage_manifest_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open storage manifest: " + path);
    }
    std::vector<StorageManifestEntry> out;
    std::string line;
    int line_no = 0;
    std::map<std::string, std::size_t> header;
    bool saw_header = false;
    while (std::getline(f, line)) {
        ++line_no;
        const std::string trimmed = trim_copy(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        const auto fields = split_csv_simple(trimmed);
        if (is_header_line(fields)) {
            header = header_index(fields);
            saw_header = true;
            continue;
        }

        StorageManifestEntry e;
        if (saw_header) {
            const auto required = {"clip_id", "path", "start_ms", "duration_ms", "bytes", "motion_score", "audio_score", "lighting_delta", "changed_pixels"};
            for (const auto* name : required) {
                if (!has_field(fields, header, name)) {
                    throw std::runtime_error("storage manifest line " + std::to_string(line_no) + " is missing required column: " + std::string(name));
                }
            }
            e.clip_id = get_field(fields, header, "clip_id");
            e.path = get_field(fields, header, "path");
            e.start_ms = parse_u64_flexible(get_field(fields, header, "start_ms"));
            e.duration_ms = parse_u64_flexible(get_field(fields, header, "duration_ms"));
            e.bytes = parse_u64_flexible(get_field(fields, header, "bytes"));
            e.motion_score = std::stod(get_field(fields, header, "motion_score"));
            e.audio_score = std::stod(get_field(fields, header, "audio_score"));
            e.lighting_delta = std::stod(get_field(fields, header, "lighting_delta"));
            e.changed_pixels = parse_u64_flexible(get_field(fields, header, "changed_pixels"));

            e.fingerprint_source = get_field(fields, header, "fingerprint_source", "metadata_synthetic");
            if (has_field(fields, header, "decoded_width")) {
                e.decoded_width = static_cast<int>(parse_u64_flexible(get_field(fields, header, "decoded_width")));
            }
            if (has_field(fields, header, "decoded_height")) {
                e.decoded_height = static_cast<int>(parse_u64_flexible(get_field(fields, header, "decoded_height")));
            }
            const bool has_real_fingerprint_cols = has_field(fields, header, "ahash64")
                && has_field(fields, header, "dhash64")
                && has_field(fields, header, "fingerprint64")
                && has_field(fields, header, "histogram16");
            if (has_real_fingerprint_cols) {
                e.media_ahash64 = parse_u64_flexible(get_field(fields, header, "ahash64"));
                e.media_dhash64 = parse_u64_flexible(get_field(fields, header, "dhash64"));
                e.media_fingerprint64 = parse_u64_flexible(get_field(fields, header, "fingerprint64"));
                e.media_histogram16 = parse_histogram16(get_field(fields, header, "histogram16"));
                e.has_media_fingerprint = e.media_histogram16.size() == 16
                    && e.decoded_width > 0
                    && e.decoded_height > 0
                    && e.fingerprint_source == "decoded_keyframe";
            }
        } else {
            if (fields.size() != 9) {
                throw std::runtime_error("storage manifest line " + std::to_string(line_no) + " must have 9 CSV fields");
            }
            e.clip_id = fields[0];
            e.path = fields[1];
            e.start_ms = static_cast<std::uint64_t>(std::stoull(fields[2]));
            e.duration_ms = static_cast<std::uint64_t>(std::stoull(fields[3]));
            e.bytes = static_cast<std::uint64_t>(std::stoull(fields[4]));
            e.motion_score = std::stod(fields[5]);
            e.audio_score = std::stod(fields[6]);
            e.lighting_delta = std::stod(fields[7]);
            e.changed_pixels = static_cast<std::uint64_t>(std::stoull(fields[8]));
        }
        e.active_tiles = static_cast<int>(std::min<std::uint64_t>(32ULL, e.changed_pixels / 2400ULL + 1ULL));
        e.priority_score = compute_priority_score(e);
        out.push_back(e);
    }
    return out;
}

StorageBatchAnalysis analyze_storage_batch_cpu(
    const std::vector<StorageManifestEntry>& manifest,
    const StorageBatchConfig& cfg) {

    HostStageTimer total_timer;
    StorageBatchAnalysis out;
    out.backend = "cpu";
    out.max_batch_bytes = cfg.max_batch_bytes;
    out.max_batch_clips = cfg.max_batch_clips;
    out.min_key_gap_ms = cfg.min_key_gap_ms;

    std::string error;
    if (!validate_storage_batch_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (manifest.empty()) {
        out.error = "manifest must contain at least one clip";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    HostStageTimer kernel_timer;
    out.manifest = manifest;
    for (auto& e : out.manifest) {
        e.priority_score = compute_priority_score(e);
    }
    out.manifest_entries = static_cast<int>(out.manifest.size());
    out.clip_count = out.manifest_entries;
    out.manifest_sorted = order_is_sorted(out.manifest);

    for (const auto& e : out.manifest) {
        out.total_manifest_bytes += e.bytes;
    }

    const auto order = timeline_order(out.manifest);
    StorageBatchReadPlan current_batch;
    for (const int idx : order) {
        const auto& e = out.manifest[static_cast<std::size_t>(idx)];
        const bool batch_full_by_count = static_cast<int>(current_batch.clip_indices.size()) >= cfg.max_batch_clips;
        const bool batch_full_by_bytes = !current_batch.clip_indices.empty()
            && (current_batch.total_bytes + e.bytes > cfg.max_batch_bytes);
        if (batch_full_by_count || batch_full_by_bytes) {
            finalize_batch(out, current_batch, out.manifest);
        }
        current_batch.clip_indices.push_back(idx);
        current_batch.total_bytes += e.bytes;
    }
    finalize_batch(out, current_batch, out.manifest);
    out.batch_count = static_cast<int>(out.batches.size());

    std::vector<int> score_order(out.manifest.size());
    for (std::size_t i = 0; i < out.manifest.size(); ++i) {
        score_order[i] = static_cast<int>(i);
    }
    std::stable_sort(score_order.begin(), score_order.end(), [&](int a, int b) {
        const auto& ea = out.manifest[static_cast<std::size_t>(a)];
        const auto& eb = out.manifest[static_cast<std::size_t>(b)];
        if (ea.priority_score != eb.priority_score) return ea.priority_score > eb.priority_score;
        if (ea.start_ms != eb.start_ms) return ea.start_ms < eb.start_ms;
        return ea.clip_id < eb.clip_id;
    });
    for (const int idx : score_order) {
        if (static_cast<int>(out.key_moments.size()) >= cfg.key_moments) {
            break;
        }
        const auto& candidate = out.manifest[static_cast<std::size_t>(idx)];
        bool far_enough = true;
        for (const auto& selected : out.key_moments) {
            const auto a = static_cast<std::int64_t>(candidate.start_ms);
            const auto b = static_cast<std::int64_t>(selected.start_ms);
            if (std::llabs(a - b) < static_cast<std::int64_t>(cfg.min_key_gap_ms)) {
                far_enough = false;
                break;
            }
        }
        if (!far_enough) {
            continue;
        }
        StorageKeyMoment m;
        m.rank = static_cast<int>(out.key_moments.size()) + 1;
        m.clip_index = idx;
        m.clip_id = candidate.clip_id;
        m.start_ms = candidate.start_ms;
        m.duration_ms = candidate.duration_ms;
        m.priority_score = candidate.priority_score;
        m.motion_score = candidate.motion_score;
        m.audio_score = candidate.audio_score;
        m.lighting_delta = candidate.lighting_delta;
        m.changed_pixels = candidate.changed_pixels;
        m.reason = key_reason(candidate);
        out.key_moments.push_back(m);
    }
    out.key_moment_count = static_cast<int>(out.key_moments.size());

    out.timeline.clip_count = out.clip_count;
    out.timeline.total_bytes = out.total_manifest_bytes;
    out.timeline.timeline_start_ms = out.manifest[static_cast<std::size_t>(order.front())].start_ms;
    out.timeline.timeline_end_ms = out.timeline.timeline_start_ms;
    double gap_sum = 0.0;
    int gap_count = 0;
    std::uint64_t previous_end = out.timeline.timeline_start_ms;
    for (int oi = 0; oi < static_cast<int>(order.size()); ++oi) {
        const auto& e = out.manifest[static_cast<std::size_t>(order[static_cast<std::size_t>(oi)])];
        const std::uint64_t end_ms = e.start_ms + e.duration_ms;
        out.timeline.timeline_start_ms = std::min(out.timeline.timeline_start_ms, e.start_ms);
        out.timeline.timeline_end_ms = std::max(out.timeline.timeline_end_ms, end_ms);
        out.timeline.covered_duration_ms += e.duration_ms;
        out.timeline.mean_motion_score += e.motion_score;
        out.timeline.mean_audio_score += e.audio_score;
        out.timeline.mean_lighting_delta += e.lighting_delta;
        out.timeline.mean_priority_score += e.priority_score;
        out.timeline.max_priority_score = std::max(out.timeline.max_priority_score, e.priority_score);
        if (oi > 0 && e.start_ms > previous_end) {
            const std::uint64_t gap = e.start_ms - previous_end;
            out.timeline.max_gap_ms = std::max(out.timeline.max_gap_ms, gap);
            gap_sum += static_cast<double>(gap);
            ++gap_count;
        }
        previous_end = std::max(previous_end, end_ms);
    }
    out.timeline.timeline_span_ms = out.timeline.timeline_end_ms - out.timeline.timeline_start_ms;
    if (out.clip_count > 0) {
        out.timeline.mean_motion_score /= static_cast<double>(out.clip_count);
        out.timeline.mean_audio_score /= static_cast<double>(out.clip_count);
        out.timeline.mean_lighting_delta /= static_cast<double>(out.clip_count);
        out.timeline.mean_priority_score /= static_cast<double>(out.clip_count);
    }
    if (gap_count > 0) {
        out.timeline.mean_gap_ms = gap_sum / static_cast<double>(gap_count);
    }

    out.timing.kernel_ms = kernel_timer.elapsed_ms();
    out.ok = true;
    out.timing.total_ms = total_timer.elapsed_ms();
    return out;
}

} // namespace node1_non_llm
