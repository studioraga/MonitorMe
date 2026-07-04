#include "node1_non_llm/gpu_lab_timing.hpp"

namespace node1_non_llm {

HostStageTimer::HostStageTimer() noexcept : start_(clock::now()) {}

void HostStageTimer::reset() noexcept {
    start_ = clock::now();
}

double HostStageTimer::elapsed_ms() const noexcept {
    const auto end = clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
    return static_cast<double>(ns) / 1.0e6;
}

} // namespace node1_non_llm
