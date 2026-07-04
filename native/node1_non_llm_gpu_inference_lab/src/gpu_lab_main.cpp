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
    } else if (scenario == "mixed") {
        // Cover four middle columns across all four tile rows: 16 active
        // tiles with the default 8x4 grid, so the route is mixed.
        paint(width / 4, 0, (width * 3) / 4, height, 200U);
    } else {
        throw std::runtime_error("unknown synthetic scenario: " + scenario);
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
    std::cerr << "node1_non_llm_gpu_lab --mode synthetic|analyze-raw-gray|audio-raw-f32 [options]\n"
              << "  --scenario sparse|mixed|dense       synthetic frame pattern\n"
              << "  --prev previous.gray --curr current.gray --width W --height H\n"
              << "  --audio samples.f32 --audio-samples N --audio-window-samples N\n"
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
        } else if (mode != "audio-raw-f32") {
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
        os << "}";
        std::cout << os.str() << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cout << "{\"ok\":false,\"error\":\"" << json_escape(exc.what()) << "\"}\n";
        return 2;
    }
}
