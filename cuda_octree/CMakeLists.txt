cmake_minimum_required(VERSION 3.18)

# Enable PIC globally to ensure all objects are position-independent
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile commands and disable NVCC response files
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE OFF CACHE BOOL "Disable NVCC response file mode for compile_commands.json" FORCE)

# Force your CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 86 CACHE STRING "CUDA architectures" FORCE)

# Project declaration
project(CudaOctree
        VERSION 0.1
        LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Locate CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Define static library target
add_library(${PROJECT_NAME} STATIC
    src/cuda_test.cu
    src/interop.cu
)

# Include directories
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRECTORIES}
)

# Compile options
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_options(${PROJECT_NAME}
    PRIVATE
        # For C++ files
        $<$<COMPILE_LANGUAGE:CXX>:-fPIC>
        # For CUDA files
        $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            -Xcompiler=-Wall
            -Xcompiler=-Wextra
            -Xcompiler=-fPIC
        >
)

# Link against CUDA runtime
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        CUDA::cudart
)

# Optionally set rpath for CUDA libraries at runtime
if(UNIX)
    target_link_options(${PROJECT_NAME} PRIVATE "-Wl,-rpath,/usr/local/cuda/lib64")
endif()

# Notes:
# * Clean build directory to pick up PIC changes:
#     rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Debug ..
# * If CUDA is in a custom location, pass:
#     -DCUDAToolkit_ROOT=/path/to/cuda
