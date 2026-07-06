cmake_minimum_required(VERSION 3.22)
project(node1_non_llm_gpu_inference_lab LANGUAGES CXX)

option(NODE1_NON_LLM_ENABLE_CUDA "Build CUDA kernels for Node1 GPU path" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(node1_non_llm_gpu_lab_core
    src/gpu_lab_types.cpp
    src/gpu_lab_timing.cpp
    src/gpu_lab_json.cpp
    src/gpu_lab_cpu.cpp
    src/isp_filters_cpu.cpp
    src/isp_pnm_io.cpp
    src/sparse_roi_cpu.cpp
    src/mixed_region_cpu.cpp
    src/dense_full_frame_cpu.cpp
    src/overlay_heavy_cpu.cpp
    src/audiobox_cpu.cpp
)
target_include_directories(node1_non_llm_gpu_lab_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(node1_non_llm_gpu_lab_core PRIVATE -Wall -Wextra -Wpedantic)

# Backward-compatible library alias for existing local build expectations.
add_library(node1_non_llm_gpu_lab_cpu ALIAS node1_non_llm_gpu_lab_core)

add_executable(node1_non_llm_gpu_lab
    src/gpu_lab_main.cpp
)
target_link_libraries(node1_non_llm_gpu_lab PRIVATE node1_non_llm_gpu_lab_core)
target_compile_options(node1_non_llm_gpu_lab PRIVATE -Wall -Wextra -Wpedantic)

add_executable(node1_non_llm_gpu_lab_selftest
    src/gpu_lab_selftest.cpp
)
target_link_libraries(node1_non_llm_gpu_lab_selftest PRIVATE node1_non_llm_gpu_lab_core)
target_compile_options(node1_non_llm_gpu_lab_selftest PRIVATE -Wall -Wextra -Wpedantic)

if(NODE1_NON_LLM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
    add_library(node1_non_llm_gpu_lab_cuda
        src/gpu_lab_cuda_utils.cu
        src/gpu_lab_cuda.cu
    )
    target_include_directories(node1_non_llm_gpu_lab_cuda
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(node1_non_llm_gpu_lab_cuda PUBLIC cxx_std_17)
    target_link_libraries(node1_non_llm_gpu_lab_cuda PUBLIC CUDA::cudart node1_non_llm_gpu_lab_core)
    target_compile_definitions(node1_non_llm_gpu_lab_cuda PUBLIC NODE1_NON_LLM_WITH_CUDA=1)
    target_compile_definitions(node1_non_llm_gpu_lab PRIVATE NODE1_NON_LLM_WITH_CUDA=1)
    target_link_libraries(node1_non_llm_gpu_lab PRIVATE node1_non_llm_gpu_lab_cuda CUDA::cudart)
endif()

install(TARGETS node1_non_llm_gpu_lab RUNTIME DESTINATION bin)
