#include "node1_non_llm/gpu_lab.hpp"
#include "node1_non_llm/gpu_lab_json.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace node1_non_llm;

namespace {

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key.rfind("--", 0) == 0) {
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                args[key.substr(2)] = argv[++i];
            } else {
                args[key.substr(2)] = "1";
            }
        }
    }
    return args;
}

int arg_int(const std::map<std::string, std::string>& args, const std::string& key, int fallback) {
    auto it = args.find(key);
    if (it == args.end()) return fallback;
    return std::stoi(it->second);
}

float arg_float(const std::map<std::string, std::string>& args, const std::string& key, float fallback) {
    auto it = args.find(key);
    if (it == args.end()) return fallback;
    return std::stof(it->second);
}

std::string arg_string(const std::map<std::string, std::string>& args, const std::string& key, const std::string& fallback) {
    auto it = args.find(key);
    if (it == args.end()) return fallback;
    return it->second;
}

bool has_flag(const std::map<std::string, std::string>& args, const std::string& key) {
    return args.find(key) != args.end();
}

std::vector<std::uint8_t> read_binary_u8(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open file: " + path);
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(f), {});
}

std::vector<float> read_binary_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open file: " + path);
    const auto bytes = f.tellg();
    if (bytes < 0 || static_cast<std::size_t>(bytes) % sizeof(float) != 0) {
        throw std::runtime_error("audio f32 raw file size must be a multiple of 4 bytes");
    }
    f.seekg(0);
    std::vector<float> out(static_cast<std::size_t>(bytes) / sizeof(float));
    f.read(reinterpret_cast<char*>(out.data()), bytes);
    return out;
}

void make_synthetic_frames(
    int width,
    int height,
    const std::string& scenario,
    std::vector<std::uint8_t>& prev,
    std::vector<std::uint8_t>& curr) {

    prev.assign(static_cast<std::size_t>(width * height), 10U);
    curr = prev;

    auto paint = [&](int x0, int y0, int x1, int y1, std::uint8_t value) {
        x0 = std::max(0, std::min(width, x0));
        y0 = std::max(0, std::min(height, y0));
        x1 = std::max(0, std::min(width, x1));
        y1 = std::max(0, std::min(height, y1));
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                curr[static_cast<std::size_t>(y * width + x)] = value;
            }
        }
    };

    if (scenario == "sparse") {
        paint(width / 16, height / 16, width / 8, height / 8, 220U);
        paint(width / 2, height / 2, width / 2 + width / 16, height / 2 + height / 16, 200U);
    } else if (scenario == "dense") {
        paint(0, 0, width, height, 210U);
    } else if (scenario == "mixed" || scenario == "contiguous") {
        // Cover four middle columns across all four tile rows: 16 active
        // tiles with the default 8x4 grid, so the route is mixed and the
        // Phase 4 component grouping is one contiguous component.
        paint(width / 4, 0, (width * 3) / 4, height, 200U);
    } else if (scenario == "scattered") {
        // Paint a checkerboard of active default 8x4 tiles. Diagonal touches
        // do not count as connected for the Phase 4 4-neighbor component walk,
        // so this creates many scattered components while still using the
        // mixed route by active tile count.
        const int tile_cols = 8;
        const int tile_rows = 4;
        for (int ty = 0; ty < tile_rows; ++ty) {
            for (int tx = 0; tx < tile_cols; ++tx) {
                if (((tx + ty) & 1) != 0) {
                    continue;
                }
                const int x0 = (tx * width) / tile_cols;
                const int x1 = ((tx + 1) * width) / tile_cols;
                const int y0 = (ty * height) / tile_rows;
                const int y1 = ((ty + 1) * height) / tile_rows;
                paint(x0, y0, x1, y1, 200U);
            }
        }
    } else {
        throw std::runtime_error("unknown synthetic scenario: " + scenario);
    }
}


void make_synthetic_audiobox(
    int sample_count,
    int drift_samples,
    std::vector<float>& primary,
    std::vector<float>& reference) {

    if (sample_count <= 0) {
        throw std::runtime_error("audio-samples must be positive");
    }
    primary.assign(static_cast<std::size_t>(sample_count), 0.0f);
    reference.assign(static_cast<std::size_t>(sample_count), 0.0f);

    auto paint_burst = [&](int start, int length, float amplitude) {
        const int end = std::min(sample_count, start + length);
        for (int i = std::max(0, start); i < end; ++i) {
            const int phase = (i - start) % 17;
            const float shaped = amplitude * (0.65f + 0.02f * static_cast<float>(phase));
            primary[static_cast<std::size_t>(i)] = (phase & 1) ? shaped : -shaped;
        }
    };

    paint_burst(sample_count / 5, sample_count / 20, 0.28f);
    paint_burst(sample_count / 2, sample_count / 16, 0.42f);
    paint_burst((sample_count * 3) / 4, sample_count / 24, 0.36f);

    for (int i = 0; i < sample_count; ++i) {
        const int j = i + drift_samples;
        if (j >= 0 && j < sample_count) {
            reference[static_cast<std::size_t>(j)] = primary[static_cast<std::size_t>(i)];
        }
    }
}

std::vector<float> make_synthetic_audio(int sample_count) {
    if (sample_count < 0) {
        throw std::runtime_error("audio-samples must be non-negative");
    }
    std::vector<float> samples(static_cast<std::size_t>(sample_count), 0.0f);
    for (int i = 0; i < sample_count; ++i) {
        if (i > sample_count / 3 && i < sample_count / 3 + sample_count / 12) {
            samples[static_cast<std::size_t>(i)] = 0.25f;
        }
        if (i > (sample_count * 2) / 3 && i < (sample_count * 2) / 3 + sample_count / 16) {
            samples[static_cast<std::size_t>(i)] = -0.35f;
        }
    }
    return samples;
}

void print_usage() {
    std::cerr << "node1_non_llm_gpu_lab --mode synthetic|analyze-raw-gray|audio-raw-f32|isp-synthetic|isp-pgm|sparse-roi-synthetic|mixed-region-synthetic|dense-full-frame-synthetic|overlay-heavy-synthetic|audiobox-synthetic [options]\n"
              << "  --scenario sparse|mixed|dense|contiguous|scattered synthetic frame pattern\n"
              << "  --prev previous.gray --curr current.gray --width W --height H\n"
              << "  --audio samples.f32 --audio-samples N --audio-window-samples N\n"
              << "  --sample-rate N --silence-threshold F --onset-threshold F --max-lag N --sync-drift-samples N\n"
              << "  --isp-filter blur|sharpen|edge|sobel-x|sobel-y|sobel-mag\n"
              << "  --input frame.pgm|frame.ppm --output filtered.pgm|filtered.ppm\n"
              << "  --target-width N --target-height N    ROI/mixed-region resize target dimensions\n"
              << "  --max-rois N                       sparse ROI maximum active tile crops\n"
              << "  --max-groups N                     mixed-region maximum connected components\n"
              << "  --include-output                   include ISP/ROI/mixed-region output values in JSON, useful for tiny tests\n"
              << "  --gpu                              also run CUDA backend when compiled with CUDA\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        if (has_flag(args, "help")) {
            print_usage();
            return 0;
        }

        const std::string mode = arg_string(args, "mode", "synthetic");
        const int width = arg_int(args, "width", 320);
        const int height = arg_int(args, "height", 240);
        TileAnalysisConfig tile_cfg;
        tile_cfg.width = width;
        tile_cfg.height = height;
        tile_cfg.tile_cols = arg_int(args, "tile-cols", 8);
        tile_cfg.tile_rows = arg_int(args, "tile-rows", 4);
        tile_cfg.pixel_threshold = arg_int(args, "pixel-threshold", 30);
        tile_cfg.sparse_threshold = arg_int(args, "sparse-threshold", 8);
        tile_cfg.dense_threshold = arg_int(args, "dense-threshold", 24);

        std::vector<std::uint8_t> prev;
        std::vector<std::uint8_t> curr;
        FrameAnalysis cpu_frame;
        FrameAnalysis gpu_frame;
        bool ran_frame = false;

        if (mode == "synthetic") {
            make_synthetic_frames(width, height, arg_string(args, "scenario", "mixed"), prev, curr);
            cpu_frame = analyze_gray_frames_cpu(prev.data(), curr.data(), tile_cfg);
            ran_frame = true;
        } else if (mode == "analyze-raw-gray") {
            prev = read_binary_u8(arg_string(args, "prev", ""));
            curr = read_binary_u8(arg_string(args, "curr", ""));
            const std::size_t expected = static_cast<std::size_t>(width * height);
            if (prev.size() != expected || curr.size() != expected) {
                throw std::runtime_error("raw gray files must match width * height bytes");
            }
            cpu_frame = analyze_gray_frames_cpu(prev.data(), curr.data(), tile_cfg);
            ran_frame = true;
        } else if (mode == "sparse-roi-synthetic") {
            make_synthetic_frames(width, height, arg_string(args, "scenario", "sparse"), prev, curr);
            cpu_frame = analyze_gray_frames_cpu(prev.data(), curr.data(), tile_cfg);
            ran_frame = true;
        } else if (mode == "mixed-region-synthetic") {
            make_synthetic_frames(width, height, arg_string(args, "scenario", "contiguous"), prev, curr);
            cpu_frame = analyze_gray_frames_cpu(prev.data(), curr.data(), tile_cfg);
            ran_frame = true;
        } else if (mode == "dense-full-frame-synthetic") {
            make_synthetic_frames(width, height, arg_string(args, "scenario", "dense"), prev, curr);
            cpu_frame = analyze_gray_frames_cpu(prev.data(), curr.data(), tile_cfg);
            ran_frame = true;
        } else if (mode == "overlay-heavy-synthetic") {
            make_synthetic_frames(width, height, arg_string(args, "scenario", "mixed"), prev, curr);
            cpu_frame = analyze_gray_frames_cpu(prev.data(), curr.data(), tile_cfg);
            ran_frame = true;
        } else if (mode != "audio-raw-f32" && mode != "audiobox-synthetic" && mode != "isp-synthetic" && mode != "isp-pgm") {
            throw std::runtime_error("unknown mode: " + mode);
        }

#ifdef NODE1_NON_LLM_WITH_CUDA
        if (ran_frame && has_flag(args, "gpu")) {
            gpu_frame = analyze_gray_frames_cuda(prev.data(), curr.data(), tile_cfg);
        }
#endif

        AudioEnergyAnalysis cpu_audio;
        AudioEnergyAnalysis gpu_audio;
        bool ran_audio = false;
        if (mode == "synthetic" || mode == "audio-raw-f32") {
            AudioEnergyConfig audio_cfg;
            audio_cfg.window_samples = arg_int(args, "audio-window-samples", 1024);
            audio_cfg.threshold = arg_float(args, "audio-threshold", 0.05f);
            audio_cfg.max_windows = arg_int(args, "audio-max-windows", 32);
            std::vector<float> samples;
            if (mode == "audio-raw-f32") {
                samples = read_binary_f32(arg_string(args, "audio", ""));
            } else {
                samples = make_synthetic_audio(arg_int(args, "audio-samples", 32768));
            }
            cpu_audio = analyze_audio_energy_cpu(samples.data(), static_cast<int>(samples.size()), audio_cfg);
#ifdef NODE1_NON_LLM_WITH_CUDA
            if (has_flag(args, "gpu")) {
                gpu_audio = analyze_audio_energy_cuda(samples.data(), static_cast<int>(samples.size()), audio_cfg);
            }
#endif
            ran_audio = true;
        }

        AudioBoxAnalysis cpu_audiobox;
        AudioBoxAnalysis gpu_audiobox;
        bool ran_audiobox = false;
        bool ran_audiobox_cuda = false;
        if (mode == "audiobox-synthetic") {
            AudioBoxConfig audiobox_cfg;
            audiobox_cfg.sample_count = arg_int(args, "audio-samples", 32768);
            audiobox_cfg.sample_rate = arg_int(args, "sample-rate", 48000);
            audiobox_cfg.window_samples = arg_int(args, "audio-window-samples", 1024);
            audiobox_cfg.silence_threshold = arg_float(args, "silence-threshold", 0.02f);
            audiobox_cfg.onset_threshold = arg_float(args, "onset-threshold", 0.08f);
            audiobox_cfg.max_windows = arg_int(args, "audio-max-windows", 32);
            audiobox_cfg.max_lag = arg_int(args, "max-lag", 128);
            audiobox_cfg.collect_output = true;

            std::vector<float> primary_samples;
            std::vector<float> reference_samples;
            make_synthetic_audiobox(
                audiobox_cfg.sample_count,
                arg_int(args, "sync-drift-samples", 64),
                primary_samples,
                reference_samples);
            cpu_audiobox = analyze_audiobox_cpu(primary_samples.data(), reference_samples.data(), audiobox_cfg);
            ran_audiobox = true;
#ifdef NODE1_NON_LLM_WITH_CUDA
            if (has_flag(args, "gpu")) {
                gpu_audiobox = analyze_audiobox_cuda(primary_samples.data(), reference_samples.data(), audiobox_cfg);
                ran_audiobox_cuda = true;
            }
#endif
        }


        IspFilterAnalysis cpu_isp;
        IspFilterAnalysis gpu_isp;
        bool ran_isp = false;
        bool ran_isp_cuda = false;
        std::string isp_input_path;
        std::string isp_output_path;
        if (mode == "isp-synthetic" || mode == "isp-pgm") {
            IspFilterKind filter_kind;
            const std::string filter_name = arg_string(args, "isp-filter", "sobel-mag");
            if (!parse_isp_filter_kind(filter_name, filter_kind)) {
                throw std::runtime_error("unknown ISP filter: " + filter_name);
            }

            ImageU8 gray_image;
            if (mode == "isp-synthetic") {
                gray_image = make_synthetic_isp_image(width, height);
            } else {
                isp_input_path = arg_string(args, "input", "");
                if (isp_input_path.empty()) {
                    throw std::runtime_error("--input is required for --mode isp-pgm");
                }
                gray_image = image_to_gray_u8(read_pnm(isp_input_path));
            }

            IspFilterConfig isp_cfg;
            isp_cfg.width = gray_image.width;
            isp_cfg.height = gray_image.height;
            isp_cfg.filter = filter_kind;
            isp_cfg.collect_output = true;
            cpu_isp = apply_isp_filter_cpu_rolling(gray_image.data.data(), isp_cfg);
            ran_isp = true;
#ifdef NODE1_NON_LLM_WITH_CUDA
            if (has_flag(args, "gpu")) {
                gpu_isp = analyze_isp_filter_cuda(gray_image.data.data(), isp_cfg);
                ran_isp_cuda = true;
            }
#endif

            isp_output_path = arg_string(args, "output", "");
            if (!isp_output_path.empty()) {
                ImageU8 out_image;
                out_image.width = cpu_isp.width;
                out_image.height = cpu_isp.height;
                out_image.channels = 1;
                out_image.data = cpu_isp.output;
                if (isp_output_path.size() >= 4 && isp_output_path.substr(isp_output_path.size() - 4) == ".ppm") {
                    write_ppm(isp_output_path, out_image);
                } else {
                    write_pgm(isp_output_path, out_image);
                }
            }
        }

        SparseRoiAnalysis cpu_sparse_roi;
        SparseRoiAnalysis gpu_sparse_roi;
        bool ran_sparse_roi = false;
        bool ran_sparse_roi_cuda = false;
        if (mode == "sparse-roi-synthetic") {
            SparseRoiConfig roi_cfg;
            roi_cfg.width = width;
            roi_cfg.height = height;
            roi_cfg.tile_cols = tile_cfg.tile_cols;
            roi_cfg.tile_rows = tile_cfg.tile_rows;
            roi_cfg.tile_mask = cpu_frame.tile_mask;
            roi_cfg.target_width = arg_int(args, "target-width", 16);
            roi_cfg.target_height = arg_int(args, "target-height", 16);
            roi_cfg.max_rois = arg_int(args, "max-rois", 32);
            roi_cfg.collect_output = true;
            cpu_sparse_roi = analyze_sparse_roi_cpu(curr.data(), roi_cfg);
            ran_sparse_roi = true;
#ifdef NODE1_NON_LLM_WITH_CUDA
            if (has_flag(args, "gpu")) {
                gpu_sparse_roi = analyze_sparse_roi_cuda(curr.data(), roi_cfg);
                ran_sparse_roi_cuda = true;
            }
#endif
        }

        MixedRegionAnalysis cpu_mixed_region;
        MixedRegionAnalysis gpu_mixed_region;
        bool ran_mixed_region = false;
        bool ran_mixed_region_cuda = false;
        if (mode == "mixed-region-synthetic") {
            MixedRegionConfig mixed_cfg;
            mixed_cfg.width = width;
            mixed_cfg.height = height;
            mixed_cfg.tile_cols = tile_cfg.tile_cols;
            mixed_cfg.tile_rows = tile_cfg.tile_rows;
            mixed_cfg.tile_mask = cpu_frame.tile_mask;
            mixed_cfg.target_width = arg_int(args, "target-width", 32);
            mixed_cfg.target_height = arg_int(args, "target-height", 32);
            mixed_cfg.max_groups = arg_int(args, "max-groups", 32);
            mixed_cfg.collect_output = true;
            cpu_mixed_region = analyze_mixed_region_cpu(curr.data(), mixed_cfg);
            ran_mixed_region = true;
#ifdef NODE1_NON_LLM_WITH_CUDA
            if (has_flag(args, "gpu")) {
                gpu_mixed_region = analyze_mixed_region_cuda(curr.data(), mixed_cfg);
                ran_mixed_region_cuda = true;
            }
#endif
        }

        DenseFullFrameAnalysis cpu_dense_full_frame;
        DenseFullFrameAnalysis gpu_dense_full_frame;
        bool ran_dense_full_frame = false;
        bool ran_dense_full_frame_cuda = false;
        if (mode == "dense-full-frame-synthetic") {
            DenseFullFrameConfig dense_cfg;
            dense_cfg.width = width;
            dense_cfg.height = height;
            dense_cfg.pixel_threshold = tile_cfg.pixel_threshold;
            dense_cfg.collect_output = true;
            cpu_dense_full_frame = analyze_dense_full_frame_cpu(prev.data(), curr.data(), dense_cfg);
            ran_dense_full_frame = true;
#ifdef NODE1_NON_LLM_WITH_CUDA
            if (has_flag(args, "gpu")) {
                gpu_dense_full_frame = analyze_dense_full_frame_cuda(prev.data(), curr.data(), dense_cfg);
                ran_dense_full_frame_cuda = true;
            }
#endif
        }

        OverlayHeavyAnalysis cpu_overlay_heavy;
        OverlayHeavyAnalysis gpu_overlay_heavy;
        bool ran_overlay_heavy = false;
        bool ran_overlay_heavy_cuda = false;
        if (mode == "overlay-heavy-synthetic") {
            OverlayHeavyConfig overlay_cfg;
            overlay_cfg.width = width;
            overlay_cfg.height = height;
            overlay_cfg.pixel_threshold = tile_cfg.pixel_threshold;
            overlay_cfg.alpha = arg_int(args, "alpha", 128);
            overlay_cfg.thumbnail_width = arg_int(args, "thumbnail-width", 64);
            overlay_cfg.thumbnail_height = arg_int(args, "thumbnail-height", 48);
            overlay_cfg.collect_output = true;
            cpu_overlay_heavy = analyze_overlay_heavy_cpu(prev.data(), curr.data(), overlay_cfg);
            ran_overlay_heavy = true;
#ifdef NODE1_NON_LLM_WITH_CUDA
            if (has_flag(args, "gpu")) {
                gpu_overlay_heavy = analyze_overlay_heavy_cuda(prev.data(), curr.data(), overlay_cfg);
                ran_overlay_heavy_cuda = true;
            }
#endif
        }

#ifndef NODE1_NON_LLM_WITH_CUDA
        (void)ran_isp_cuda;
        (void)ran_sparse_roi_cuda;
        (void)ran_mixed_region_cuda;
        (void)ran_dense_full_frame_cuda;
        (void)ran_overlay_heavy_cuda;
        (void)ran_audiobox_cuda;
#endif
        std::ostringstream os;
        os << "{";
        os << "\"ok\":true,";
        os << "\"schema\":\"node1_non_llm_gpu_inference_lab.v0.1\",";
        os << "\"mode\":\"" << json_escape(mode) << "\",";
        os << "\"cuda_compiled\":";
#ifdef NODE1_NON_LLM_WITH_CUDA
        os << "true";
#else
        os << "false";
#endif
        os << ",\"frame\":" << (ran_frame ? frame_analysis_json(cpu_frame) : "null");
#ifdef NODE1_NON_LLM_WITH_CUDA
        os << ",\"frame_cuda\":" << (gpu_frame.backend == "cuda" ? frame_analysis_json(gpu_frame) : "null");
#else
        os << ",\"frame_cuda\":null";
#endif
        os << ",\"audio\":" << (ran_audio ? audio_analysis_json(cpu_audio) : "null");
#ifdef NODE1_NON_LLM_WITH_CUDA
        os << ",\"audio_cuda\":" << (gpu_audio.backend == "cuda" ? audio_analysis_json(gpu_audio) : "null");
#else
        os << ",\"audio_cuda\":null";
#endif
        const bool include_isp_output = has_flag(args, "include-output");
        os << ",\"isp\":" << (ran_isp ? isp_filter_analysis_json(cpu_isp, include_isp_output) : "null");
#ifdef NODE1_NON_LLM_WITH_CUDA
        os << ",\"isp_cuda\":" << (ran_isp_cuda ? isp_filter_analysis_json(gpu_isp, include_isp_output) : "null");
        os << ",\"isp_cpu_cuda_comparison\":" << (ran_isp_cuda ? isp_cpu_cuda_comparison_json(cpu_isp, gpu_isp) : "null");
#else
        os << ",\"isp_cuda\":null";
        os << ",\"isp_cpu_cuda_comparison\":null";
#endif
        const bool include_sparse_roi_output = has_flag(args, "include-output");
        os << ",\"sparse_roi\":" << (ran_sparse_roi ? sparse_roi_analysis_json(cpu_sparse_roi, include_sparse_roi_output) : "null");
#ifdef NODE1_NON_LLM_WITH_CUDA
        os << ",\"sparse_roi_cuda\":" << (ran_sparse_roi_cuda ? sparse_roi_analysis_json(gpu_sparse_roi, include_sparse_roi_output) : "null");
        os << ",\"sparse_roi_cpu_cuda_comparison\":" << (ran_sparse_roi_cuda ? sparse_roi_cpu_cuda_comparison_json(cpu_sparse_roi, gpu_sparse_roi) : "null");
#else
        os << ",\"sparse_roi_cuda\":null";
        os << ",\"sparse_roi_cpu_cuda_comparison\":null";
#endif
        const bool include_mixed_region_output = has_flag(args, "include-output");
        os << ",\"mixed_region\":" << (ran_mixed_region ? mixed_region_analysis_json(cpu_mixed_region, include_mixed_region_output) : "null");
#ifdef NODE1_NON_LLM_WITH_CUDA
        os << ",\"mixed_region_cuda\":" << (ran_mixed_region_cuda ? mixed_region_analysis_json(gpu_mixed_region, include_mixed_region_output) : "null");
        os << ",\"mixed_region_cpu_cuda_comparison\":" << (ran_mixed_region_cuda ? mixed_region_cpu_cuda_comparison_json(cpu_mixed_region, gpu_mixed_region) : "null");
#else
        os << ",\"mixed_region_cuda\":null";
        os << ",\"mixed_region_cpu_cuda_comparison\":null";
#endif
        const bool include_dense_full_frame_output = has_flag(args, "include-output");
        os << ",\"dense_full_frame\":" << (ran_dense_full_frame ? dense_full_frame_analysis_json(cpu_dense_full_frame, include_dense_full_frame_output) : "null");
#ifdef NODE1_NON_LLM_WITH_CUDA
        os << ",\"dense_full_frame_cuda\":" << (ran_dense_full_frame_cuda ? dense_full_frame_analysis_json(gpu_dense_full_frame, include_dense_full_frame_output) : "null");
        os << ",\"dense_full_frame_cpu_cuda_comparison\":" << (ran_dense_full_frame_cuda ? dense_full_frame_cpu_cuda_comparison_json(cpu_dense_full_frame, gpu_dense_full_frame) : "null");
#else
        os << ",\"dense_full_frame_cuda\":null";
        os << ",\"dense_full_frame_cpu_cuda_comparison\":null";
#endif
        const bool include_overlay_heavy_output = has_flag(args, "include-output");
        os << ",\"overlay_heavy\":" << (ran_overlay_heavy ? overlay_heavy_analysis_json(cpu_overlay_heavy, include_overlay_heavy_output) : "null");
#ifdef NODE1_NON_LLM_WITH_CUDA
        os << ",\"overlay_heavy_cuda\":" << (ran_overlay_heavy_cuda ? overlay_heavy_analysis_json(gpu_overlay_heavy, include_overlay_heavy_output) : "null");
        os << ",\"overlay_heavy_cpu_cuda_comparison\":" << (ran_overlay_heavy_cuda ? overlay_heavy_cpu_cuda_comparison_json(cpu_overlay_heavy, gpu_overlay_heavy) : "null");
#else
        os << ",\"overlay_heavy_cuda\":null";
        os << ",\"overlay_heavy_cpu_cuda_comparison\":null";
#endif
        const bool include_audiobox_output = has_flag(args, "include-output");
        os << ",\"audiobox\":" << (ran_audiobox ? audiobox_analysis_json(cpu_audiobox, include_audiobox_output) : "null");
#ifdef NODE1_NON_LLM_WITH_CUDA
        os << ",\"audiobox_cuda\":" << (ran_audiobox_cuda ? audiobox_analysis_json(gpu_audiobox, include_audiobox_output) : "null");
        os << ",\"audiobox_cpu_cuda_comparison\":" << (ran_audiobox_cuda ? audiobox_cpu_cuda_comparison_json(cpu_audiobox, gpu_audiobox) : "null");
#else
        os << ",\"audiobox_cuda\":null";
        os << ",\"audiobox_cpu_cuda_comparison\":null";
#endif
        if (ran_isp) {

            os << ",\"isp_input\":\"" << json_escape(isp_input_path) << "\"";
            os << ",\"isp_output\":\"" << json_escape(isp_output_path) << "\"";
        }
        os << "}";
        std::cout << os.str() << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cout << "{\"ok\":false,\"error\":\"" << json_escape(exc.what()) << "\"}\n";
        return 2;
    }
}
