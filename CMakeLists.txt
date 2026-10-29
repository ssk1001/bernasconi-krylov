cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Optimize for performance: set CUDA architectures before project() enables CUDA language
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

project(FixedG_Subspace_Solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Add CUDA library path if present (e.g. on Colab or standard Linux)
if(EXISTS "/usr/local/cuda/lib64")
    link_directories(/usr/local/cuda/lib64)
endif()

# Find CUDA Toolkit (Provides cuBLAS and cuSOLVER)
find_package(CUDAToolkit REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find cuStateVec (cuQuantum SDK)
if(NOT CUSTATEVEC_LIB)
    find_library(CUSTATEVEC_LIB
        NAMES
            custatevec
            libcustatevec.so.1
            libcustatevec.so.12
            libcustatevec.so
        HINTS 
            ${CUQUANTUM_ROOT}/lib
            $ENV{CUQUANTUM_ROOT}/lib
            ${CUQUANTUM_ROOT}
            $ENV{CUQUANTUM_ROOT}
            /usr/local/cuquantum/lib
        PATH_SUFFIXES
            lib
            lib64
            cuquantum/lib
    )
endif()

if(NOT CUSTATEVEC_LIB)
    message(FATAL_ERROR "custatevec library not found. Please ensure cuQuantum is installed and CUQUANTUM_ROOT is set.")
endif()

# ==============================================================================
# Modern Unified Benchmark Executor (Primary Target)
# ==============================================================================
add_executable(run_benchmark 
    src/run_benchmark.cu
)

target_include_directories(run_benchmark PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/labs_energy
    ${CMAKE_SOURCE_DIR}/src/sampling
    ${CMAKE_SOURCE_DIR}/src/subspace
    ${CMAKE_SOURCE_DIR}/src/exact
    ${CMAKE_SOURCE_DIR}/src/classical
    ${CMAKE_SOURCE_DIR}/src/config
    ${CMAKE_SOURCE_DIR}/src/connector
)

if(DEFINED ENV{CUQUANTUM_ROOT})
    target_include_directories(run_benchmark PRIVATE $ENV{CUQUANTUM_ROOT}/include $ENV{CUQUANTUM_ROOT})
elseif(DEFINED CUQUANTUM_ROOT)
    target_include_directories(run_benchmark PRIVATE ${CUQUANTUM_ROOT}/include ${CUQUANTUM_ROOT})
endif()
if(DEFINED CUQUANTUM_INCLUDE_DIR)
    target_include_directories(run_benchmark PRIVATE ${CUQUANTUM_INCLUDE_DIR})
endif()

target_link_libraries(run_benchmark PRIVATE 
    ${CUSTATEVEC_LIB}
    CUDA::cublas 
    CUDA::cusolver 
    Eigen3::Eigen
)
if(EXISTS "/usr/local/cuda/lib64")
    target_link_options(run_benchmark PRIVATE "-Wl,-rpath-link,/usr/local/cuda/lib64")
endif()

# ==============================================================================
# Standalone CLI Utilities
# ==============================================================================
add_executable(compute_d_min_cli 
    src/tools/compute_d_min_cli.cpp
)
target_include_directories(compute_d_min_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/labs_energy
)

# ==============================================================================
# Deprecated Legacy Executables (Disabled by default)
# ==============================================================================
option(BUILD_LEGACY_TARGETS "Build deprecated legacy executables (fixed_g_custatevec, probability_dynamics)" OFF)
if(BUILD_LEGACY_TARGETS)
    # Define the main legacy executable
    add_executable(fixed_g_custatevec 
        src/fixed_g_custatevec.cu
    )

    target_include_directories(fixed_g_custatevec PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/labs_energy
        ${CMAKE_SOURCE_DIR}/src/sampling
        ${CMAKE_SOURCE_DIR}/src/subspace
        ${CMAKE_SOURCE_DIR}/src/exact
        ${CMAKE_SOURCE_DIR}/src/classical
        ${CMAKE_SOURCE_DIR}/src/baselines
    )

    if(DEFINED ENV{CUQUANTUM_ROOT})
        target_include_directories(fixed_g_custatevec PRIVATE $ENV{CUQUANTUM_ROOT}/include)
    elseif(DEFINED CUQUANTUM_ROOT)
        target_include_directories(fixed_g_custatevec PRIVATE ${CUQUANTUM_ROOT}/include)
    endif()

    target_link_libraries(fixed_g_custatevec PRIVATE 
        ${CUSTATEVEC_LIB}
        CUDA::cublas 
        CUDA::cusolver 
        Eigen3::Eigen
    )
    if(EXISTS "/usr/local/cuda/lib64")
        target_link_options(fixed_g_custatevec PRIVATE "-Wl,-rpath-link,/usr/local/cuda/lib64")
    endif()

    # Legacy probability dynamics executable
    add_executable(probability_dynamics 
        src/probability_dynamics.cu
    )

    target_include_directories(probability_dynamics PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/labs_energy
        ${CMAKE_SOURCE_DIR}/src/sampling
        ${CMAKE_SOURCE_DIR}/src/subspace
        ${CMAKE_SOURCE_DIR}/src/exact
        ${CMAKE_SOURCE_DIR}/src/classical
    )

    if(DEFINED ENV{CUQUANTUM_ROOT})
        target_include_directories(probability_dynamics PRIVATE $ENV{CUQUANTUM_ROOT}/include)
    elseif(DEFINED CUQUANTUM_ROOT)
        target_include_directories(probability_dynamics PRIVATE ${CUQUANTUM_ROOT}/include)
    endif()

    target_link_libraries(probability_dynamics PRIVATE 
        ${CUSTATEVEC_LIB}
        CUDA::cublas 
        CUDA::cusolver 
        Eigen3::Eigen
    )
    if(EXISTS "/usr/local/cuda/lib64")
        target_link_options(probability_dynamics PRIVATE "-Wl,-rpath-link,/usr/local/cuda/lib64")
    endif()
endif()