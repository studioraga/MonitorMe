#include "node1_non_llm/gpu_lab_cuda_utils.cuh"

#ifdef NODE1_NON_LLM_WITH_CUDA

namespace node1_non_llm {

std::string cuda_error_message(cudaError_t err, const char* where) {
    std::string out(where ? where : "cuda");
    out += ": ";
    out += cudaGetErrorString(err);
    return out;
}

CudaEventTimer::CudaEventTimer() noexcept {
    if (cudaEventCreate(&start_) == cudaSuccess && cudaEventCreate(&stop_) == cudaSuccess) {
        created_ = true;
    } else {
        if (start_) {
            cudaEventDestroy(start_);
            start_ = nullptr;
        }
        if (stop_) {
            cudaEventDestroy(stop_);
            stop_ = nullptr;
        }
        created_ = false;
    }
}

CudaEventTimer::~CudaEventTimer() noexcept {
    if (start_) {
        cudaEventDestroy(start_);
    }
    if (stop_) {
        cudaEventDestroy(stop_);
    }
}

cudaError_t CudaEventTimer::start() noexcept {
    if (!created_) {
        return cudaErrorInitializationError;
    }
    return cudaEventRecord(start_, 0);
}

cudaError_t CudaEventTimer::stop(double& elapsed_ms) noexcept {
    elapsed_ms = 0.0;
    if (!created_) {
        return cudaErrorInitializationError;
    }
    cudaError_t err = cudaEventRecord(stop_, 0);
    if (err != cudaSuccess) {
        return err;
    }
    err = cudaEventSynchronize(stop_);
    if (err != cudaSuccess) {
        return err;
    }
    float ms = 0.0f;
    err = cudaEventElapsedTime(&ms, start_, stop_);
    if (err != cudaSuccess) {
        return err;
    }
    elapsed_ms = static_cast<double>(ms);
    return cudaSuccess;
}

} // namespace node1_non_llm

#endif // NODE1_NON_LLM_WITH_CUDA
