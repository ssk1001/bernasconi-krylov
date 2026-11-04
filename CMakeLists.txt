cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Check if CUDA language compiler is available in the environment
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()
    project(FixedG_Subspace_Solver LANGUAGES CXX CUDA)
    set(ENABLE_CUDA TRUE)
else()
    project(FixedG_Subspace_Solver LANGUAGES CXX)
    set(ENABLE_CUDA FALSE)
    message(STATUS "[CMake] CUDA compiler (nvcc) not found. Configuring CPU-only tools and unit tests.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --expt-relaxed-constexpr")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Find Eigen3
find_package(Eigen3 REQUIRED)

# ==============================================================================
# Modern Unified Benchmark Executor (CUDA + cuQuantum Target)
# ==============================================================================
if(ENABLE_CUDA)
    # Add CUDA library path if present (e.g. standard Linux or Colab)
    if(EXISTS "/usr/local/cuda/lib64")
        link_directories(/usr/local/cuda/lib64)
    endif()

    find_package(CUDAToolkit REQUIRED)

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

    if(CUSTATEVEC_LIB)
        add_executable(run_benchmark 
            src/run_benchmark.cu
        )

        target_include_directories(run_benchmark PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/labs_energy
            ${CMAKE_SOURCE_DIR}/src/sampling
            ${CMAKE_SOURCE_DIR}/src/subspace
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
    else()
        message(WARNING "[CMake] custatevec library not found. 'run_benchmark' will not be built.")
    endif()
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

if(EXISTS "${CMAKE_SOURCE_DIR}/src/tools/run_tabu.cpp")
    add_executable(run_tabu
        src/tools/run_tabu.cpp
    )
    target_include_directories(run_tabu PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/labs_energy
        ${CMAKE_SOURCE_DIR}/src/classical
    )
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/src/tools/exhaustive_minima.cpp")
    add_executable(exhaustive_minima
        src/tools/exhaustive_minima.cpp
    )
    target_include_directories(exhaustive_minima PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/labs_energy
        ${CMAKE_SOURCE_DIR}/src/classical
    )
endif()

# ==============================================================================
# Unit Tests
# ==============================================================================
option(BUILD_UNIT_TESTS "Build C++ unit test suite" ON)
if(BUILD_UNIT_TESTS)
    enable_testing()
    set(CORE_TESTS test_labs_energy test_d_min test_config test_connector test_exact_diag)
    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_tabu_search.cpp")
        list(APPEND CORE_TESTS test_tabu_search)
    endif()
    foreach(test_name ${CORE_TESTS})
        add_executable(${test_name} tests/${test_name}.cpp)
        target_include_directories(${test_name} PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/labs_energy
            ${CMAKE_SOURCE_DIR}/src/classical
            ${CMAKE_SOURCE_DIR}/src/config
            ${CMAKE_SOURCE_DIR}/src/connector
            ${CMAKE_SOURCE_DIR}/src/subspace
        )
        target_link_libraries(${test_name} PRIVATE Eigen3::Eigen)
        add_test(NAME ${test_name} COMMAND ${test_name} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endforeach()

    if(ENABLE_CUDA AND CUSTATEVEC_LIB)
        foreach(cuda_test test_lanczos test_sampler test_steepest_descent)
            add_executable(${cuda_test} tests/${cuda_test}.cu)
            target_include_directories(${cuda_test} PRIVATE
                ${CMAKE_SOURCE_DIR}/src
                ${CMAKE_SOURCE_DIR}/src/labs_energy
                ${CMAKE_SOURCE_DIR}/src/classical
                ${CMAKE_SOURCE_DIR}/src/config
                ${CMAKE_SOURCE_DIR}/src/connector
                ${CMAKE_SOURCE_DIR}/src/subspace
                ${CMAKE_SOURCE_DIR}/src/sampling
            )
            target_link_libraries(${cuda_test} PRIVATE
                ${CUSTATEVEC_LIB}
                CUDA::cublas
                CUDA::cusolver
                Eigen3::Eigen
            )
            add_test(NAME ${cuda_test} COMMAND ${cuda_test} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        endforeach()
    endif()
endif()