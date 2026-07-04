#include "node1_non_llm/isp_filters.hpp"

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace node1_non_llm {
namespace {

std::uint8_t clamp_u8_int(int value) noexcept {
    return static_cast<std::uint8_t>(std::max(0, std::min(255, value)));
}

int clamp_index(int value, int upper_exclusive) noexcept {
    if (value < 0) return 0;
    if (value >= upper_exclusive) return upper_exclusive - 1;
    return value;
}

void load_clamped_row(const std::uint8_t* src, int width, int height, int row, std::vector<std::uint8_t>& out) {
    const int y = clamp_index(row, height);
    const auto* base = src + static_cast<std::size_t>(y * width);
    std::copy(base, base + width, out.begin());
}

int row_at(const std::vector<std::uint8_t>& row, int x, int width) noexcept {
    return static_cast<int>(row[static_cast<std::size_t>(clamp_index(x, width))]);
}

int src_at(const std::uint8_t* src, int width, int height, int x, int y) noexcept {
    const int cx = clamp_index(x, width);
    const int cy = clamp_index(y, height);
    return static_cast<int>(src[static_cast<std::size_t>(cy * width + cx)]);
}

std::uint8_t apply_filter_window(IspFilterKind filter, const int px[3][3]) noexcept {
    switch (filter) {
        case IspFilterKind::Blur3x3: {
            int sum = 0;
            for (int y = 0; y < 3; ++y) {
                for (int x = 0; x < 3; ++x) sum += px[y][x];
            }
            return clamp_u8_int((sum + 4) / 9);
        }
        case IspFilterKind::Sharpen3x3: {
            const int v = 5 * px[1][1] - px[0][1] - px[1][0] - px[1][2] - px[2][1];
            return clamp_u8_int(v);
        }
        case IspFilterKind::Edge3x3: {
            const int v = 8 * px[1][1]
                - px[0][0] - px[0][1] - px[0][2]
                - px[1][0]            - px[1][2]
                - px[2][0] - px[2][1] - px[2][2];
            return clamp_u8_int(std::abs(v));
        }
        case IspFilterKind::SobelX: {
            const int sx = -px[0][0] + px[0][2]
                         - 2 * px[1][0] + 2 * px[1][2]
                         - px[2][0] + px[2][2];
            return clamp_u8_int(std::abs(sx));
        }
        case IspFilterKind::SobelY: {
            const int sy = -px[0][0] - 2 * px[0][1] - px[0][2]
                         + px[2][0] + 2 * px[2][1] + px[2][2];
            return clamp_u8_int(std::abs(sy));
        }
        case IspFilterKind::SobelMag: {
            const int sx = -px[0][0] + px[0][2]
                         - 2 * px[1][0] + 2 * px[1][2]
                         - px[2][0] + px[2][2];
            const int sy = -px[0][0] - 2 * px[0][1] - px[0][2]
                         + px[2][0] + 2 * px[2][1] + px[2][2];
            const int mag = static_cast<int>(std::lround(std::sqrt(static_cast<double>(sx * sx + sy * sy))));
            return clamp_u8_int(mag);
        }
    }
    return 0;
}

void finalize_metrics(IspFilterAnalysis& out, const std::uint8_t* input, const std::vector<std::uint8_t>& values) {
    if (values.empty()) return;

    std::uint64_t sum = 0;
    std::uint64_t diff_sum = 0;
    std::uint64_t saturation = 0;
    int min_v = 255;
    int max_v = 0;
    double input_sum = 0.0;

    for (std::size_t i = 0; i < values.size(); ++i) {
        const int v = static_cast<int>(values[i]);
        const int in = static_cast<int>(input[i]);
        min_v = std::min(min_v, v);
        max_v = std::max(max_v, v);
        sum += static_cast<std::uint64_t>(v);
        input_sum += static_cast<double>(in);
        diff_sum += static_cast<std::uint64_t>(std::abs(v - in));
        if (v == 0 || v == 255) ++saturation;
    }

    const double n = static_cast<double>(values.size());
    const double mean = static_cast<double>(sum) / n;
    double variance_acc = 0.0;
    for (std::uint8_t value : values) {
        const double delta = static_cast<double>(value) - mean;
        variance_acc += delta * delta;
    }

    out.output_min = min_v;
    out.output_max = max_v;
    out.output_mean = mean;
    out.edge_energy = mean;
    out.focus_score = variance_acc / n;
    out.noise_score = static_cast<double>(diff_sum) / n;
    out.lighting_delta = std::abs(mean - (input_sum / n));
    out.saturation_pixels = saturation;
    out.saturation_ratio = static_cast<double>(saturation) / n;
}

} // namespace

const char* isp_filter_name(IspFilterKind filter) noexcept {
    switch (filter) {
        case IspFilterKind::Blur3x3: return "blur";
        case IspFilterKind::Sharpen3x3: return "sharpen";
        case IspFilterKind::Edge3x3: return "edge";
        case IspFilterKind::SobelX: return "sobel-x";
        case IspFilterKind::SobelY: return "sobel-y";
        case IspFilterKind::SobelMag: return "sobel-mag";
    }
    return "unknown";
}

bool parse_isp_filter_kind(const std::string& value, IspFilterKind& out) noexcept {
    if (value == "blur" || value == "blur3x3") {
        out = IspFilterKind::Blur3x3;
    } else if (value == "sharpen" || value == "sharpen3x3") {
        out = IspFilterKind::Sharpen3x3;
    } else if (value == "edge" || value == "conv-edge" || value == "edge3x3") {
        out = IspFilterKind::Edge3x3;
    } else if (value == "sobel-x" || value == "sobelx") {
        out = IspFilterKind::SobelX;
    } else if (value == "sobel-y" || value == "sobely") {
        out = IspFilterKind::SobelY;
    } else if (value == "sobel-mag" || value == "sobel" || value == "sobel-magnitude") {
        out = IspFilterKind::SobelMag;
    } else {
        return false;
    }
    return true;
}

bool validate_isp_filter_config(const IspFilterConfig& cfg, std::string& error) noexcept {
    if (cfg.width <= 0 || cfg.height <= 0) {
        error = "width and height must be positive";
        return false;
    }
    const auto pixels = static_cast<std::uint64_t>(cfg.width) * static_cast<std::uint64_t>(cfg.height);
    if (pixels > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        error = "image is too large for Phase 1 ISP CPU lab";
        return false;
    }
    error.clear();
    return true;
}

ImageU8 make_synthetic_isp_image(int width, int height) {
    ImageU8 image;
    image.width = width;
    image.height = height;
    image.channels = 1;
    image.data.assign(static_cast<std::size_t>(width * height), 0U);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int value = (x * 255) / std::max(width - 1, 1);
            if (x >= width / 3 && x < (width * 2) / 3 && y >= height / 3 && y < (height * 2) / 3) {
                value = 220;
            }
            if ((x + y) % 17 == 0) {
                value = 255;
            }
            image.data[static_cast<std::size_t>(y * width + x)] = static_cast<std::uint8_t>(value);
        }
    }
    return image;
}

IspFilterAnalysis apply_isp_filter_cpu_reference(const std::uint8_t* gray, const IspFilterConfig& cfg) {
    HostStageTimer timer;
    IspFilterAnalysis out;
    out.backend = "cpu_reference";
    out.filter = isp_filter_name(cfg.filter);
    out.width = cfg.width;
    out.height = cfg.height;
    out.channels = 1;
    out.pixels_processed = static_cast<std::uint64_t>(cfg.width) * static_cast<std::uint64_t>(cfg.height);
    out.bytes_read = out.pixels_processed;
    out.bytes_written = out.pixels_processed;

    std::string error;
    if (!gray) {
        out.error = "gray input pointer is null";
        return out;
    }
    if (!validate_isp_filter_config(cfg, error)) {
        out.error = error;
        return out;
    }

    std::vector<std::uint8_t> values(static_cast<std::size_t>(cfg.width * cfg.height));
    for (int y = 0; y < cfg.height; ++y) {
        for (int x = 0; x < cfg.width; ++x) {
            int px[3][3];
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    px[ky + 1][kx + 1] = src_at(gray, cfg.width, cfg.height, x + kx, y + ky);
                }
            }
            values[static_cast<std::size_t>(y * cfg.width + x)] = apply_filter_window(cfg.filter, px);
        }
    }

    out.ok = true;
    if (cfg.collect_output) out.output = values;
    finalize_metrics(out, gray, values);
    out.timing.kernel_ms = timer.elapsed_ms();
    out.timing.total_ms = out.timing.kernel_ms;
    return out;
}

IspFilterAnalysis apply_isp_filter_cpu_rolling(const std::uint8_t* gray, const IspFilterConfig& cfg) {
    HostStageTimer timer;
    IspFilterAnalysis out;
    out.backend = "cpu";
    out.filter = isp_filter_name(cfg.filter);
    out.width = cfg.width;
    out.height = cfg.height;
    out.channels = 1;
    out.pixels_processed = static_cast<std::uint64_t>(cfg.width) * static_cast<std::uint64_t>(cfg.height);
    out.bytes_read = out.pixels_processed;
    out.bytes_written = out.pixels_processed;

    std::string error;
    if (!gray) {
        out.error = "gray input pointer is null";
        return out;
    }
    if (!validate_isp_filter_config(cfg, error)) {
        out.error = error;
        return out;
    }

    std::vector<std::uint8_t> values(static_cast<std::size_t>(cfg.width * cfg.height));
    std::vector<std::uint8_t> line0(static_cast<std::size_t>(cfg.width));
    std::vector<std::uint8_t> line1(static_cast<std::size_t>(cfg.width));
    std::vector<std::uint8_t> line2(static_cast<std::size_t>(cfg.width));

    load_clamped_row(gray, cfg.width, cfg.height, -1, line0);
    load_clamped_row(gray, cfg.width, cfg.height, 0, line1);
    load_clamped_row(gray, cfg.width, cfg.height, 1, line2);

    for (int y = 0; y < cfg.height; ++y) {
        for (int x = 0; x < cfg.width; ++x) {
            int px[3][3];
            px[0][0] = row_at(line0, x - 1, cfg.width);
            px[0][1] = row_at(line0, x,     cfg.width);
            px[0][2] = row_at(line0, x + 1, cfg.width);
            px[1][0] = row_at(line1, x - 1, cfg.width);
            px[1][1] = row_at(line1, x,     cfg.width);
            px[1][2] = row_at(line1, x + 1, cfg.width);
            px[2][0] = row_at(line2, x - 1, cfg.width);
            px[2][1] = row_at(line2, x,     cfg.width);
            px[2][2] = row_at(line2, x + 1, cfg.width);
            values[static_cast<std::size_t>(y * cfg.width + x)] = apply_filter_window(cfg.filter, px);
        }
        // True rolling line buffer: rotate row ownership and load only the new
        // bottom row for the next output line. This avoids reloading y-1/y/y+1
        // from scratch for every row.
        if (y + 1 < cfg.height) {
            line0.swap(line1);
            line1.swap(line2);
            load_clamped_row(gray, cfg.width, cfg.height, y + 2, line2);
        }
    }

    out.ok = true;
    if (cfg.collect_output) out.output = values;
    finalize_metrics(out, gray, values);
    out.timing.kernel_ms = timer.elapsed_ms();
    out.timing.total_ms = out.timing.kernel_ms;
    return out;
}

} // namespace node1_non_llm
