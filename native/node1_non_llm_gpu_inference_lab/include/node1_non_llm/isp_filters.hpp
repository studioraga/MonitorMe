#pragma once

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace node1_non_llm {

enum class IspFilterKind {
    Blur3x3 = 0,
    Sharpen3x3 = 1,
    Edge3x3 = 2,
    SobelX = 3,
    SobelY = 4,
    SobelMag = 5,
};

struct ImageU8 {
    int width = 0;
    int height = 0;
    int channels = 1;
    std::vector<std::uint8_t> data;
};

struct IspFilterConfig {
    int width = 0;
    int height = 0;
    IspFilterKind filter = IspFilterKind::SobelMag;
    bool collect_output = true;
};

struct IspFilterAnalysis {
    bool ok = false;
    std::string backend = "cpu";
    std::string schema = "node1_non_llm_isp_filters.v0.1";
    std::string profile = "edge_feature";
    std::string filter;
    int width = 0;
    int height = 0;
    int channels = 1;
    std::uint64_t pixels_processed = 0;
    std::uint64_t bytes_read = 0;
    std::uint64_t bytes_written = 0;
    int output_min = 0;
    int output_max = 0;
    double output_mean = 0.0;
    double edge_energy = 0.0;
    double focus_score = 0.0;
    double noise_score = 0.0;
    double lighting_delta = 0.0;
    std::uint64_t saturation_pixels = 0;
    double saturation_ratio = 0.0;
    StageTiming timing;
    std::vector<std::uint8_t> output;
    std::string error;
};

const char* isp_filter_name(IspFilterKind filter) noexcept;
bool parse_isp_filter_kind(const std::string& value, IspFilterKind& out) noexcept;
bool validate_isp_filter_config(const IspFilterConfig& cfg, std::string& error) noexcept;

ImageU8 make_synthetic_isp_image(int width, int height);
ImageU8 image_to_gray_u8(const ImageU8& image);
ImageU8 read_pnm(const std::string& path);
void write_pgm(const std::string& path, const ImageU8& image);
void write_ppm(const std::string& path, const ImageU8& image);

IspFilterAnalysis apply_isp_filter_cpu_rolling(const std::uint8_t* gray, const IspFilterConfig& cfg);
IspFilterAnalysis apply_isp_filter_cpu_reference(const std::uint8_t* gray, const IspFilterConfig& cfg);

#ifdef NODE1_NON_LLM_WITH_CUDA
IspFilterAnalysis analyze_isp_filter_cuda(const std::uint8_t* gray, const IspFilterConfig& cfg);
IspFilterAnalysis apply_isp_filter_cuda_tiled(const std::uint8_t* gray, const IspFilterConfig& cfg);
#endif

} // namespace node1_non_llm
