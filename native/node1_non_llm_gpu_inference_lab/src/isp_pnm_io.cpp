#include "node1_non_llm/isp_filters.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace node1_non_llm {
namespace {

std::string next_pnm_token(std::istream& in) {
    std::string token;
    char c = 0;
    while (in.get(c)) {
        if (std::isspace(static_cast<unsigned char>(c))) continue;
        if (c == '#') {
            std::string ignored;
            std::getline(in, ignored);
            continue;
        }
        token.push_back(c);
        break;
    }
    while (in.get(c)) {
        if (std::isspace(static_cast<unsigned char>(c))) break;
        token.push_back(c);
    }
    return token;
}

void expect_image_contract(const ImageU8& image) {
    if (image.width <= 0 || image.height <= 0) {
        throw std::runtime_error("image width/height must be positive");
    }
    if (image.channels != 1 && image.channels != 3) {
        throw std::runtime_error("image channels must be 1 or 3");
    }
    const auto expected = static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height) * static_cast<std::size_t>(image.channels);
    if (image.data.size() != expected) {
        throw std::runtime_error("image data size does not match dimensions");
    }
}

} // namespace

ImageU8 image_to_gray_u8(const ImageU8& image) {
    expect_image_contract(image);
    if (image.channels == 1) return image;

    ImageU8 gray;
    gray.width = image.width;
    gray.height = image.height;
    gray.channels = 1;
    gray.data.resize(static_cast<std::size_t>(image.width * image.height));
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            const auto idx = static_cast<std::size_t>((y * image.width + x) * 3);
            const int r = static_cast<int>(image.data[idx + 0]);
            const int g = static_cast<int>(image.data[idx + 1]);
            const int b = static_cast<int>(image.data[idx + 2]);
            const int v = std::max(0, std::min(255, static_cast<int>(0.299 * r + 0.587 * g + 0.114 * b + 0.5)));
            gray.data[static_cast<std::size_t>(y * image.width + x)] = static_cast<std::uint8_t>(v);
        }
    }
    return gray;
}

ImageU8 read_pnm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open PGM/PPM input: " + path);

    const std::string magic = next_pnm_token(in);
    if (magic != "P5" && magic != "P6") {
        throw std::runtime_error("unsupported PNM format; expected binary P5 PGM or P6 PPM");
    }
    const int width = std::stoi(next_pnm_token(in));
    const int height = std::stoi(next_pnm_token(in));
    const int max_value = std::stoi(next_pnm_token(in));
    if (width <= 0 || height <= 0 || max_value != 255) {
        throw std::runtime_error("PNM input must have positive dimensions and max value 255");
    }

    ImageU8 image;
    image.width = width;
    image.height = height;
    image.channels = (magic == "P6") ? 3 : 1;
    const auto bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(image.channels);
    image.data.resize(bytes);
    in.read(reinterpret_cast<char*>(image.data.data()), static_cast<std::streamsize>(bytes));
    if (in.gcount() != static_cast<std::streamsize>(bytes)) {
        throw std::runtime_error("PNM input ended before expected pixel payload");
    }
    return image;
}

void write_pgm(const std::string& path, const ImageU8& image) {
    const ImageU8 gray = image_to_gray_u8(image);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open PGM output: " + path);
    out << "P5\n" << gray.width << " " << gray.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(gray.data.data()), static_cast<std::streamsize>(gray.data.size()));
}

void write_ppm(const std::string& path, const ImageU8& image) {
    expect_image_contract(image);
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot open PPM output: " + path);
    out << "P6\n" << image.width << " " << image.height << "\n255\n";
    if (image.channels == 3) {
        out.write(reinterpret_cast<const char*>(image.data.data()), static_cast<std::streamsize>(image.data.size()));
        return;
    }
    std::vector<std::uint8_t> rgb(static_cast<std::size_t>(image.width * image.height * 3));
    for (int i = 0; i < image.width * image.height; ++i) {
        const auto v = image.data[static_cast<std::size_t>(i)];
        rgb[static_cast<std::size_t>(i * 3 + 0)] = v;
        rgb[static_cast<std::size_t>(i * 3 + 1)] = v;
        rgb[static_cast<std::size_t>(i * 3 + 2)] = v;
    }
    out.write(reinterpret_cast<const char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
}

} // namespace node1_non_llm
