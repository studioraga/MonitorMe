#pragma once

#ifdef NODE1_NON_LLM_WITH_CUDA

#include "node1_non_llm/gpu_lab_timing.hpp"

#include <cuda_runtime.h>

#include <string>

namespace node1_non_llm {

std::string cuda_error_message(cudaError_t err, const char* where);

class CudaEventTimer {
public:
    CudaEventTimer() noexcept;
    ~CudaEventTimer() noexcept;
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    cudaError_t start() noexcept;
    cudaError_t stop(double& elapsed_ms) noexcept;
    bool ok() const noexcept { return created_; }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
    bool created_ = false;
};

} // namespace node1_non_llm

#endif // NODE1_NON_LLM_WITH_CUDA
