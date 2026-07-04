#pragma once

#include <chrono>

namespace node1_non_llm {

struct StageTiming {
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
};

class HostStageTimer {
public:
    HostStageTimer() noexcept;
    void reset() noexcept;
    double elapsed_ms() const noexcept;

private:
    using clock = std::chrono::steady_clock;
    clock::time_point start_;
};

} // namespace node1_non_llm
