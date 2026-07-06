#include "node1_non_llm/overlay_heavy.hpp"

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace node1_non_llm {
namespace {

std::uint8_t blend_channel(std::uint8_t base, std::uint8_t overlay, int alpha) noexcept {
    const int inv = 255 - alpha;
    return static_cast<std::uint8_t>((static_cast<int>(base) * inv + static_cast<int>(overlay) * alpha + 127) / 255);
}

void heat_color(std::uint8_t base, int diff, std::uint8_t& r, std::uint8_t& g, std::uint8_t& b) noexcept {
    if (diff <= 0) {
        r = base;
        g = base;
        b = base;
        return;
    }
    r = 255U;
    g = static_cast<std::uint8_t>(std::max(0, 255 - diff));
    b = 0U;
}

} // namespace

bool validate_overlay_heavy_config(const OverlayHeavyConfig& cfg, std::string& error) noexcept {
    if (cfg.width <= 0 || cfg.height <= 0) {
        error = "width and height must be positive";
        return false;
    }
    if (cfg.pixel_threshold < 0 || cfg.pixel_threshold > 255) {
        error = "pixel_threshold must be in [0, 255]";
        return false;
    }
    if (cfg.alpha < 0 || cfg.alpha > 255) {
        error = "alpha must be in [0, 255]";
        return false;
    }
    if (cfg.thumbnail_width <= 0 || cfg.thumbnail_height <= 0) {
        error = "thumbnail dimensions must be positive";
        return false;
    }
    error.clear();
    return true;
}

OverlayHeavyAnalysis analyze_overlay_heavy_cpu(
    const std::uint8_t* previous_gray,
    const std::uint8_t* current_gray,
    const OverlayHeavyConfig& cfg) {

    HostStageTimer total_timer;
    OverlayHeavyAnalysis out;
    out.backend = "cpu";
    out.width = cfg.width;
    out.height = cfg.height;
    out.pixel_threshold = cfg.pixel_threshold;
    out.alpha = cfg.alpha;
    out.alpha_ratio = static_cast<double>(cfg.alpha) / 255.0;
    out.thumbnail_width = cfg.thumbnail_width;
    out.thumbnail_height = cfg.thumbnail_height;
    out.pixels_processed = static_cast<std::uint64_t>(cfg.width) * static_cast<std::uint64_t>(cfg.height);
    const std::uint64_t thumbnail_pixels = static_cast<std::uint64_t>(std::max(0, cfg.thumbnail_width)) * static_cast<std::uint64_t>(std::max(0, cfg.thumbnail_height));
    out.bytes_read = out.pixels_processed * 2U;
    out.bytes_written = out.pixels_processed + out.pixels_processed * 3U + thumbnail_pixels * 3U;

    std::string error;
    if (!previous_gray || !current_gray) {
        out.error = "previous_gray and current_gray must not be null";
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }
    if (!validate_overlay_heavy_config(cfg, error)) {
        out.error = error;
        out.timing.total_ms = total_timer.elapsed_ms();
        return out;
    }

    const std::size_t total_pixels = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);
    const std::size_t thumb_pixels = static_cast<std::size_t>(cfg.thumbnail_width) * static_cast<std::size_t>(cfg.thumbnail_height);
    if (cfg.collect_output) {
        out.heatmap.assign(total_pixels, 0U);
        out.overlay_rgb.assign(total_pixels * 3U, 0U);
        out.thumbnail_rgb.assign(thumb_pixels * 3U, 0U);
    }

    HostStageTimer kernel_timer;
    std::uint64_t sum_diff = 0;
    std::uint64_t sum_prev = 0;
    std::uint64_t sum_curr = 0;
    std::uint64_t sum_overlay = 0;
    int min_diff = 255;
    int max_diff = 0;

    for (std::size_t i = 0; i < total_pixels; ++i) {
        const int prev = static_cast<int>(previous_gray[i]);
        const int curr = static_cast<int>(current_gray[i]);
        const int diff = std::abs(curr - prev);
        if (diff > cfg.pixel_threshold) {
            ++out.changed_pixels;
        }
        min_diff = std::min(min_diff, diff);
        max_diff = std::max(max_diff, diff);
        sum_diff += static_cast<std::uint64_t>(diff);
        sum_prev += static_cast<std::uint64_t>(prev);
        sum_curr += static_cast<std::uint64_t>(curr);

        if (cfg.collect_output) {
            out.heatmap[i] = static_cast<std::uint8_t>(diff);
            std::uint8_t heat_r = 0U;
            std::uint8_t heat_g = 0U;
            std::uint8_t heat_b = 0U;
            const auto base = static_cast<std::uint8_t>(curr);
            heat_color(base, diff, heat_r, heat_g, heat_b);
            const std::size_t j = i * 3U;
            out.overlay_rgb[j + 0U] = blend_channel(base, heat_r, cfg.alpha);
            out.overlay_rgb[j + 1U] = blend_channel(base, heat_g, cfg.alpha);
            out.overlay_rgb[j + 2U] = blend_channel(base, heat_b, cfg.alpha);
            sum_overlay += static_cast<std::uint64_t>(out.overlay_rgb[j + 0U]);
            sum_overlay += static_cast<std::uint64_t>(out.overlay_rgb[j + 1U]);
            sum_overlay += static_cast<std::uint64_t>(out.overlay_rgb[j + 2U]);
        }
    }

    std::uint64_t sum_thumbnail = 0;
    if (cfg.collect_output) {
        for (int ty = 0; ty < cfg.thumbnail_height; ++ty) {
            const int sy = std::min((ty * cfg.height) / cfg.thumbnail_height, cfg.height - 1);
            for (int tx = 0; tx < cfg.thumbnail_width; ++tx) {
                const int sx = std::min((tx * cfg.width) / cfg.thumbnail_width, cfg.width - 1);
                const std::size_t src = static_cast<std::size_t>(sy * cfg.width + sx) * 3U;
                const std::size_t dst = static_cast<std::size_t>(ty * cfg.thumbnail_width + tx) * 3U;
                out.thumbnail_rgb[dst + 0U] = out.overlay_rgb[src + 0U];
                out.thumbnail_rgb[dst + 1U] = out.overlay_rgb[src + 1U];
                out.thumbnail_rgb[dst + 2U] = out.overlay_rgb[src + 2U];
                sum_thumbnail += static_cast<std::uint64_t>(out.thumbnail_rgb[dst + 0U]);
                sum_thumbnail += static_cast<std::uint64_t>(out.thumbnail_rgb[dst + 1U]);
                sum_thumbnail += static_cast<std::uint64_t>(out.thumbnail_rgb[dst + 2U]);
            }
        }
    }
    out.timing.kernel_ms = kernel_timer.elapsed_ms();

    const double n = static_cast<double>(std::max<std::size_t>(1U, total_pixels));
    out.heatmap_min = min_diff;
    out.heatmap_max = max_diff;
    out.heatmap_mean = static_cast<double>(sum_diff) / n;
    out.before_after_max_diff = max_diff;
    out.before_after_abs_mean = out.heatmap_mean;
    out.previous_mean = static_cast<double>(sum_prev) / n;
    out.current_mean = static_cast<double>(sum_curr) / n;
    out.lighting_delta = std::abs(out.current_mean - out.previous_mean);
    out.changed_ratio = static_cast<double>(out.changed_pixels) / n;
    out.overlay_mean = cfg.collect_output ? static_cast<double>(sum_overlay) / (n * 3.0) : 0.0;
    const double tn = static_cast<double>(std::max<std::size_t>(1U, thumb_pixels));
    out.thumbnail_mean = cfg.collect_output ? static_cast<double>(sum_thumbnail) / (tn * 3.0) : 0.0;
    out.ok = true;
    out.timing.total_ms = total_timer.elapsed_ms();
    return out;
}

} // namespace node1_non_llm
