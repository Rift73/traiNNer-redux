cmake_minimum_required(VERSION 3.25)

project(DrftTensorRTPlugins LANGUAGES CXX CUDA)

if(NOT DEFINED TENSORRT_ROOT)
    message(FATAL_ERROR "TENSORRT_ROOT must point to the TensorRT SDK directory")
endif()

find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS
        "${TENSORRT_ROOT}/include"
        "${TENSORRT_ROOT}/include/x86_64-linux-gnu"
    NO_DEFAULT_PATH
    REQUIRED
)

find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer_11 nvinfer
    HINTS
        "${TENSORRT_ROOT}/lib"
        "${TENSORRT_ROOT}/lib64"
        "${TENSORRT_ROOT}/lib/x86_64-linux-gnu"
    NO_DEFAULT_PATH
    REQUIRED
)

message(STATUS "TensorRT include directory: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT runtime library: ${TENSORRT_NVINFER_LIBRARY}")

set(
    DRFT_CUDA_ARCHITECTURES
    "80-real;86-real;89-real;90-real;120-real;120-virtual"
    CACHE STRING
    "CUDA architectures for the DRFT TensorRT plugin pack"
)
set(CMAKE_CUDA_ARCHITECTURES "${DRFT_CUDA_ARCHITECTURES}")

add_library(DrftTensorRTPlugins SHARED
    DrftOCABAttentionPlugin.cu
)

target_compile_features(DrftTensorRTPlugins PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(DrftTensorRTPlugins PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPlugins PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPlugins PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPlugins PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPlugins PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPlugins PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPlugins PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsV3"
    CUDA_SEPARABLE_COMPILATION OFF
)

add_library(DrftTensorRTPluginsSafe SHARED
    DrftOCABAttentionSafePlugin.cu
)

target_compile_features(DrftTensorRTPluginsSafe PRIVATE cxx_std_17 cuda_std_17)
target_compile_options(DrftTensorRTPluginsSafe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPluginsSafe PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPluginsSafe PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPluginsSafe PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPluginsSafe PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPluginsSafe PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPluginsSafe PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsSafeV1"
    CUDA_SEPARABLE_COMPILATION OFF
)

add_library(DrftTensorRTPluginsOptimized SHARED
    DrftOCABAttentionSafePlugin.cu
    DrftImageLayerNormPlugin.cu
)

target_compile_features(DrftTensorRTPluginsOptimized PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(DrftTensorRTPluginsOptimized PRIVATE
    DRFT_ENABLE_IMAGE_LN_PLUGIN=1
)
target_compile_options(DrftTensorRTPluginsOptimized PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPluginsOptimized PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPluginsOptimized PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPluginsOptimized PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPluginsOptimized PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPluginsOptimized PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPluginsOptimized PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsOptimizedV1"
    CUDA_SEPARABLE_COMPILATION OFF
)

add_library(DrftTensorRTPluginsOptimizedV2 SHARED
    DrftOCABAttentionSafePlugin.cu
    DrftImageLayerNormPlugin.cu
)

target_compile_features(DrftTensorRTPluginsOptimizedV2 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(DrftTensorRTPluginsOptimizedV2 PRIVATE
    DRFT_ENABLE_IMAGE_LN_PLUGIN=1
    DRFT_ENABLE_IMAGE_LN_FAST_PLUGIN=1
)
target_compile_options(DrftTensorRTPluginsOptimizedV2 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPluginsOptimizedV2 PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPluginsOptimizedV2 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPluginsOptimizedV2 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPluginsOptimizedV2 PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPluginsOptimizedV2 PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPluginsOptimizedV2 PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsOptimizedV2"
    CUDA_SEPARABLE_COMPILATION OFF
)

add_library(DrftTensorRTPluginsOptimizedV3 SHARED
    DrftOCABAttentionSafePlugin.cu
    DrftImageLayerNormPlugin.cu
    DrftOCABQKVPackPlugin.cu
)

target_compile_features(DrftTensorRTPluginsOptimizedV3 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(DrftTensorRTPluginsOptimizedV3 PRIVATE
    DRFT_ENABLE_IMAGE_LN_PLUGIN=1
    DRFT_ENABLE_OCAB_QKV_PACK_PLUGIN=1
)
target_compile_options(DrftTensorRTPluginsOptimizedV3 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPluginsOptimizedV3 PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPluginsOptimizedV3 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPluginsOptimizedV3 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPluginsOptimizedV3 PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPluginsOptimizedV3 PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPluginsOptimizedV3 PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsOptimizedV3"
    CUDA_SEPARABLE_COMPILATION OFF
)

add_library(DrftTensorRTPluginsOptimizedV4 SHARED
    DrftOCABAttentionSafePlugin.cu
    DrftImageLayerNormPlugin.cu
    DrftOCABQKVPackPlugin.cu
)

target_compile_features(DrftTensorRTPluginsOptimizedV4 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(DrftTensorRTPluginsOptimizedV4 PRIVATE
    DRFT_ENABLE_IMAGE_LN_COOPERATIVE_PLUGIN=1
    DRFT_ENABLE_OCAB_QKV_PACK_PLUGIN=1
)
target_compile_options(DrftTensorRTPluginsOptimizedV4 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPluginsOptimizedV4 PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPluginsOptimizedV4 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPluginsOptimizedV4 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPluginsOptimizedV4 PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPluginsOptimizedV4 PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPluginsOptimizedV4 PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsOptimizedV4"
    CUDA_SEPARABLE_COMPILATION OFF
)

add_library(DrftTensorRTPluginsOptimizedV5 SHARED
    DrftOCABAttentionSafePlugin.cu
    DrftImageLayerNormPlugin.cu
    DrftOCABQKVPackPlugin.cu
    DrftFFNSwiGLUDepthwisePlugin.cu
)

target_compile_features(DrftTensorRTPluginsOptimizedV5 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(DrftTensorRTPluginsOptimizedV5 PRIVATE
    DRFT_ENABLE_IMAGE_LN_PLUGIN=1
    DRFT_ENABLE_OCAB_QKV_PACK_PLUGIN=1
    DRFT_ENABLE_FFN_SWIGLU_DEPTHWISE_PLUGIN=1
)
target_compile_options(DrftTensorRTPluginsOptimizedV5 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPluginsOptimizedV5 PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPluginsOptimizedV5 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPluginsOptimizedV5 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPluginsOptimizedV5 PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPluginsOptimizedV5 PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPluginsOptimizedV5 PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsOptimizedV5"
    CUDA_SEPARABLE_COMPILATION OFF
)

add_library(DrftTensorRTPluginsOptimizedV6 SHARED
    DrftOCABAttentionSafePlugin.cu
    DrftImageLayerNormPlugin.cu
    DrftOCABQKVPackPlugin.cu
    DrftResidualImageLayerNormPlugin.cu
)

target_compile_features(DrftTensorRTPluginsOptimizedV6 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(DrftTensorRTPluginsOptimizedV6 PRIVATE
    DRFT_ENABLE_IMAGE_LN_PLUGIN=1
    DRFT_ENABLE_OCAB_QKV_PACK_PLUGIN=1
    DRFT_ENABLE_RESIDUAL_IMAGE_LN_PLUGIN=1
)
target_compile_options(DrftTensorRTPluginsOptimizedV6 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPluginsOptimizedV6 PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPluginsOptimizedV6 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPluginsOptimizedV6 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPluginsOptimizedV6 PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPluginsOptimizedV6 PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPluginsOptimizedV6 PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsOptimizedV6"
    CUDA_SEPARABLE_COMPILATION OFF
)

add_library(DrftTensorRTPluginsOptimizedV7 SHARED
    DrftOCABAttentionSafePlugin.cu
    DrftImageLayerNormPlugin.cu
    DrftOCABQKVPackPlugin.cu
)

target_compile_features(DrftTensorRTPluginsOptimizedV7 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(DrftTensorRTPluginsOptimizedV7 PRIVATE
    DRFT_ENABLE_IMAGE_LN_PLUGIN=1
    DRFT_ENABLE_OCAB_QKV_PACK_PLUGIN=1
    DRFT_ENABLE_OCAB_QKV_PACK_VECTORIZED=1
)
target_compile_options(DrftTensorRTPluginsOptimizedV7 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPluginsOptimizedV7 PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPluginsOptimizedV7 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPluginsOptimizedV7 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPluginsOptimizedV7 PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPluginsOptimizedV7 PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPluginsOptimizedV7 PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsOptimizedV7"
    CUDA_SEPARABLE_COMPILATION OFF
)

add_library(DrftTensorRTPluginsOptimizedV8 SHARED
    DrftOCABAttentionSafePlugin.cu
    DrftImageLayerNormPlugin.cu
    DrftOCABQKVPackPlugin.cu
)

target_compile_features(DrftTensorRTPluginsOptimizedV8 PRIVATE cxx_std_17 cuda_std_17)
target_compile_definitions(DrftTensorRTPluginsOptimizedV8 PRIVATE
    DRFT_ENABLE_IMAGE_LN_PLUGIN=1
    DRFT_ENABLE_OCAB_QKV_PACK_PLUGIN=1
    DRFT_ENABLE_OCAB_QKV_PACK_TOKEN_MAJOR=1
)
target_compile_options(DrftTensorRTPluginsOptimizedV8 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(WIN32)
    target_compile_definitions(DrftTensorRTPluginsOptimizedV8 PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(DrftTensorRTPluginsOptimizedV8 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>
    )
else()
    target_compile_options(DrftTensorRTPluginsOptimizedV8 PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

target_include_directories(DrftTensorRTPluginsOptimizedV8 PRIVATE
    "${TENSORRT_INCLUDE_DIR}"
)
target_link_libraries(DrftTensorRTPluginsOptimizedV8 PRIVATE
    "${TENSORRT_NVINFER_LIBRARY}"
    CUDA::cudart
)

set_target_properties(DrftTensorRTPluginsOptimizedV8 PROPERTIES
    OUTPUT_NAME "DrftTensorRTPluginsOptimizedV8"
    CUDA_SEPARABLE_COMPILATION OFF
)
