cmake_minimum_required(VERSION 3.22)
project(node1_non_llm_gpu_inference_lab LANGUAGES CXX)

option(NODE1_NON_LLM_ENABLE_CUDA "Build CUDA kernels for Node1 GPU path" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(node1_non_llm_gpu_lab_cpu
    src/gpu_lab_cpu.cpp
)
target_include_directories(node1_non_llm_gpu_lab_cpu
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_compile_options(node1_non_llm_gpu_lab_cpu PRIVATE -Wall -Wextra -Wpedantic)

add_executable(node1_non_llm_gpu_lab
    src/gpu_lab_main.cpp
)
target_link_libraries(node1_non_llm_gpu_lab PRIVATE node1_non_llm_gpu_lab_cpu)

target_compile_options(node1_non_llm_gpu_lab PRIVATE -Wall -Wextra -Wpedantic)

if(NODE1_NON_LLM_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 120)
    endif()
    add_library(node1_non_llm_gpu_lab_cuda
        src/gpu_lab_cuda.cu
    )
    target_include_directories(node1_non_llm_gpu_lab_cuda
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_compile_features(node1_non_llm_gpu_lab_cuda PUBLIC cxx_std_17)
    target_link_libraries(node1_non_llm_gpu_lab_cuda PUBLIC CUDA::cudart node1_non_llm_gpu_lab_cpu)
    target_compile_definitions(node1_non_llm_gpu_lab_cuda PUBLIC NODE1_NON_LLM_WITH_CUDA=1)
    target_compile_definitions(node1_non_llm_gpu_lab PRIVATE NODE1_NON_LLM_WITH_CUDA=1)
    target_link_libraries(node1_non_llm_gpu_lab PRIVATE node1_non_llm_gpu_lab_cuda CUDA::cudart)
endif()

install(TARGETS node1_non_llm_gpu_lab RUNTIME DESTINATION bin)
