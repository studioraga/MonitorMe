#include "node1_non_llm/gpu_lab.hpp"
#include "node1_non_llm/gpu_lab_json.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "node1_non_llm/isp_filters.hpp"

using namespace node1_non_llm;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<std::uint8_t> make_frame(int width, int height, std::uint8_t value) {
    return std::vector<std::uint8_t>(static_cast<std::size_t>(width * height), value);
}

void paint(std::vector<std::uint8_t>& frame, int width, int height, int x0, int y0, int x1, int y1, std::uint8_t value) {
    x0 = std::max(0, std::min(width, x0));
    x1 = std::max(0, std::min(width, x1));
    y0 = std::max(0, std::min(height, y0));
    y1 = std::max(0, std::min(height, y1));
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            frame[static_cast<std::size_t>(y * width + x)] = value;
        }
    }
}

void test_shared_type_helpers() {
    require(std::string(workload_path_name(WorkloadPath::Sparse)) == "sparse", "sparse name mismatch");
    require(std::string(workload_path_name(WorkloadPath::Mixed)) == "mixed", "mixed name mismatch");
    require(std::string(workload_path_name(WorkloadPath::Dense)) == "dense", "dense name mismatch");
    require(choose_workload_path(8, 8, 24) == WorkloadPath::Sparse, "sparse boundary mismatch");
    require(choose_workload_path(9, 8, 24) == WorkloadPath::Mixed, "mixed lower boundary mismatch");
    require(choose_workload_path(24, 8, 24) == WorkloadPath::Dense, "dense boundary mismatch");
    require(hex32(0x3C3C3C3CU) == "0x3C3C3C3C", "hex32 mismatch");
    require(popcount32(0xF0F0U) == 8, "popcount mismatch");
}

void test_config_validation() {
    std::string error;
    TileAnalysisConfig cfg{320, 240, 8, 4, 30, 8, 24};
    require(validate_tile_config(cfg, error), "valid tile config rejected");
    cfg.tile_cols = 9;
    cfg.tile_rows = 4;
    require(!validate_tile_config(cfg, error), "invalid >32 tile config accepted");
    cfg = TileAnalysisConfig{320, 240, 8, 4, 30, 24, 8};
    require(!validate_tile_config(cfg, error), "invalid threshold config accepted");

    AudioEnergyConfig acfg;
    require(validate_audio_config(32768, acfg, error), "valid audio config rejected");
    acfg.max_windows = 33;
    require(!validate_audio_config(32768, acfg, error), "invalid max_windows accepted");
    acfg.max_windows = 32;
    acfg.threshold = -0.1f;
    require(!validate_audio_config(32768, acfg, error), "invalid negative threshold accepted");
}

void test_cpu_frame_routes() {
    const int width = 320;
    const int height = 240;
    TileAnalysisConfig cfg{width, height, 8, 4, 30, 8, 24};

    auto prev = make_frame(width, height, 10U);
    auto curr = prev;
    paint(curr, width, height, width / 16, height / 16, width / 8, height / 8, 220U);
    paint(curr, width, height, width / 2, height / 2, width / 2 + width / 16, height / 2 + height / 16, 200U);
    FrameAnalysis sparse = analyze_gray_frames_cpu(prev.data(), curr.data(), cfg);
    require(sparse.ok, "sparse CPU analysis failed: " + sparse.error);
    require(sparse.path == WorkloadPath::Sparse, "sparse CPU route mismatch");
    require(sparse.tile_mask == 0x00100001U, "sparse tile mask mismatch");
    require(sparse.timing.total_ms >= 0.0, "sparse timing missing");

    curr = prev;
    paint(curr, width, height, width / 4, 0, (width * 3) / 4, height, 200U);
    FrameAnalysis mixed = analyze_gray_frames_cpu(prev.data(), curr.data(), cfg);
    require(mixed.ok, "mixed CPU analysis failed: " + mixed.error);
    require(mixed.path == WorkloadPath::Mixed, "mixed CPU route mismatch");
    require(mixed.tile_mask == 0x3C3C3C3CU, "mixed tile mask mismatch");

    curr = prev;
    paint(curr, width, height, 0, 0, width, height, 210U);
    FrameAnalysis dense = analyze_gray_frames_cpu(prev.data(), curr.data(), cfg);
    require(dense.ok, "dense CPU analysis failed: " + dense.error);
    require(dense.path == WorkloadPath::Dense, "dense CPU route mismatch");
    require(dense.tile_mask == 0xFFFFFFFFU, "dense tile mask mismatch");
}

void test_cpu_audio() {
    std::vector<float> samples(4096, 0.0f);
    for (int i = 1024; i < 2048; ++i) {
        samples[static_cast<std::size_t>(i)] = 0.25f;
    }
    AudioEnergyConfig cfg;
    cfg.window_samples = 1024;
    cfg.threshold = 0.05f;
    cfg.max_windows = 4;
    AudioEnergyAnalysis audio = analyze_audio_energy_cpu(samples.data(), static_cast<int>(samples.size()), cfg);
    require(audio.ok, "audio CPU analysis failed: " + audio.error);
    require(audio.active_windows == 1, "audio active window mismatch");
    require(audio.event_mask == 0x00000002U, "audio mask mismatch");
    require(audio.timing.total_ms >= 0.0, "audio timing missing");
}

void test_json_helpers() {
    require(json_escape("a\"b\\c\n") == "a\\\"b\\\\c\\n", "json escape mismatch");
    StageTiming t;
    t.h2d_ms = 1.0;
    t.kernel_ms = 2.0;
    t.d2h_ms = 3.0;
    t.total_ms = 6.0;
    const std::string timing_json = stage_timing_json(t);
    require(timing_json.find("\"kernel_ms\":2") != std::string::npos, "timing JSON missing kernel_ms");

    FrameAnalysis frame;
    frame.ok = true;
    frame.backend = "cpu";
    frame.tile_mask = 1;
    frame.low_half_mask = 1;
    frame.path = WorkloadPath::Sparse;
    const std::string frame_json = frame_analysis_json(frame);
    require(frame_json.find("\"timing\":") != std::string::npos, "frame JSON missing timing");

    AudioEnergyAnalysis audio;
    audio.ok = true;
    audio.backend = "cpu";
    audio.event_mask = 1;
    const std::string audio_json = audio_analysis_json(audio);
    require(audio_json.find("\"timing\":") != std::string::npos, "audio JSON missing timing");
}

void test_isp_filter_reference_equivalence() {
    const int width = 9;
    const int height = 7;
    ImageU8 image = make_synthetic_isp_image(width, height);
    for (IspFilterKind filter : {
            IspFilterKind::Blur3x3,
            IspFilterKind::Sharpen3x3,
            IspFilterKind::Edge3x3,
            IspFilterKind::SobelX,
            IspFilterKind::SobelY,
            IspFilterKind::SobelMag}) {
        IspFilterConfig cfg;
        cfg.width = width;
        cfg.height = height;
        cfg.filter = filter;
        cfg.collect_output = true;
        const auto rolling = apply_isp_filter_cpu_rolling(image.data.data(), cfg);
        const auto ref = apply_isp_filter_cpu_reference(image.data.data(), cfg);
        require(rolling.ok, std::string("rolling ISP failed for ") + isp_filter_name(filter) + ": " + rolling.error);
        require(ref.ok, std::string("reference ISP failed for ") + isp_filter_name(filter) + ": " + ref.error);
        require(rolling.output == ref.output, std::string("rolling/reference mismatch for ") + isp_filter_name(filter));
        require(rolling.pixels_processed == static_cast<std::uint64_t>(width * height), "ISP pixels_processed mismatch");
        require(rolling.bytes_read == static_cast<std::uint64_t>(width * height), "ISP bytes_read mismatch");
        require(rolling.bytes_written == static_cast<std::uint64_t>(width * height), "ISP bytes_written mismatch");
        require(rolling.timing.total_ms >= 0.0, "ISP timing missing");
    }
}

void test_isp_known_values() {
    ImageU8 image;
    image.width = 5;
    image.height = 5;
    image.channels = 1;
    image.data.assign(25, 0U);
    image.data[12] = 255U;

    IspFilterConfig cfg;
    cfg.width = image.width;
    cfg.height = image.height;
    cfg.collect_output = true;

    cfg.filter = IspFilterKind::Blur3x3;
    auto blur = apply_isp_filter_cpu_rolling(image.data.data(), cfg);
    require(blur.ok, "blur failed");
    require(static_cast<int>(blur.output[12]) == 28, "blur center expected rounded 255/9 = 28");

    cfg.filter = IspFilterKind::Sharpen3x3;
    auto sharpen = apply_isp_filter_cpu_rolling(image.data.data(), cfg);
    require(sharpen.ok, "sharpen failed");
    require(static_cast<int>(sharpen.output[12]) == 255, "sharpen center expected clamp to 255");

    cfg.filter = IspFilterKind::Edge3x3;
    auto edge = apply_isp_filter_cpu_rolling(image.data.data(), cfg);
    require(edge.ok, "edge failed");
    require(static_cast<int>(edge.output[12]) == 255, "edge center expected clamp to 255");

    cfg.filter = IspFilterKind::SobelMag;
    auto sobel = apply_isp_filter_cpu_rolling(image.data.data(), cfg);
    require(sobel.ok, "sobel-mag failed");
    require(sobel.output.size() == 25, "sobel output size mismatch");
}

void test_pgm_ppm_io() {
    const auto base = std::filesystem::temp_directory_path() / "node1_isp_phase1_selftest";
    const auto pgm_path = base.string() + ".pgm";
    const auto ppm_path = base.string() + ".ppm";

    ImageU8 image = make_synthetic_isp_image(8, 6);
    write_pgm(pgm_path, image);
    ImageU8 pgm = read_pnm(pgm_path);
    require(pgm.width == 8 && pgm.height == 6 && pgm.channels == 1, "PGM roundtrip shape mismatch");
    require(pgm.data == image.data, "PGM roundtrip data mismatch");

    write_ppm(ppm_path, image);
    ImageU8 ppm = read_pnm(ppm_path);
    require(ppm.width == 8 && ppm.height == 6 && ppm.channels == 3, "PPM roundtrip shape mismatch");
    ImageU8 gray = image_to_gray_u8(ppm);
    require(gray.data == image.data, "PPM gray conversion mismatch");

    std::filesystem::remove(pgm_path);
    std::filesystem::remove(ppm_path);
}

void test_isp_json() {
    ImageU8 image = make_synthetic_isp_image(8, 6);
    IspFilterConfig cfg;
    cfg.width = image.width;
    cfg.height = image.height;
    cfg.filter = IspFilterKind::SobelMag;
    auto analysis = apply_isp_filter_cpu_rolling(image.data.data(), cfg);
    const std::string json = isp_filter_analysis_json(analysis, false);
    require(json.find("\"schema\":\"node1_non_llm_isp_filters.v0.1\"") != std::string::npos, "ISP JSON missing schema");
    require(json.find("\"facts_only\":true") != std::string::npos, "ISP JSON missing facts_only");
    require(json.find("\"edge_energy\":") != std::string::npos, "ISP JSON missing edge_energy");
}

} // namespace

int main() {
    try {
        test_shared_type_helpers();
        test_config_validation();
        test_cpu_frame_routes();
        test_cpu_audio();
        test_json_helpers();
        test_isp_filter_reference_equivalence();
        test_isp_known_values();
        test_pgm_ppm_io();
        test_isp_json();
        std::cout << "node1_non_llm_gpu_lab_selftest PASS\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "node1_non_llm_gpu_lab_selftest FAIL: " << exc.what() << "\n";
        return 1;
    }
}
