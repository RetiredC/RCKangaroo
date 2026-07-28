cmake_minimum_required(VERSION 3.17)

# Must be set before project(), because this project compiles a .cu file.
set(CUDAToolkit_ROOT "/usr/local/cuda-12.8" CACHE PATH "CUDA Toolkit root")
set(CMAKE_CUDA_COMPILER "${CUDAToolkit_ROOT}/bin/nvcc" CACHE FILEPATH "CUDA compiler")

project(rckangaroo
    VERSION 1.0
    LANGUAGES C CXX CUDA
)

set(TARGET_NAME rckangaroo)

# Make compile_commands.json for VS Code IntelliSense and clang tooling.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# The old Makefile always used -O3, so Release is the sensible default.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(PROJECT_SOURCES
    RCKangaroo.cpp
    GpuKang.cpp
    Ec.cpp
    utils.cpp
    CallCubin.cpp
    RCGpuCore.cu
)

add_executable(${TARGET_NAME} ${PROJECT_SOURCES})

target_include_directories(${TARGET_NAME} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Preserve the PTX targets from the old Makefile exactly.
target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_89,code=compute_89>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=compute_86>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_75,code=compute_75>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_61,code=compute_61>
)

# Match the working project: static CUDA runtime plus NVIDIA driver API.
if(NOT TARGET CUDA::cudart_static)
    message(FATAL_ERROR "CUDA::cudart_static was not found. Check CUDAToolkit_ROOT and CUDA Toolkit installation.")
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    CUDA::cudart_static
    CUDA::cuda_driver
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET_NAME} PRIVATE rt)
endif()

if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /W4)
else()
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unknown-pragmas>
    )
endif()

set_target_properties(${TARGET_NAME} PROPERTIES
    CUDA_RUNTIME_LIBRARY Static
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
