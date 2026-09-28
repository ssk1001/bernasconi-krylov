cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FixedG_Subspace_Solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for performance
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

# Find CUDA Toolkit (Provides cuBLAS and cuSOLVER)
find_package(CUDAToolkit REQUIRED)

# Find cuStateVec (cuQuantum SDK)
find_library(CUSTATEVEC_LIB custatevec
    HINTS 
        ${CUQUANTUM_ROOT}/lib
        $ENV{CUQUANTUM_ROOT}/lib
        /usr/local/cuquantum/lib
)

if(NOT CUSTATEVEC_LIB)
    message(FATAL_ERROR "custatevec library not found. Please ensure cuQuantum is installed and CUQUANTUM_ROOT is set.")
endif()

# Define the main executable
add_executable(fixed_g_custatevec 
    src/fixed_g_custatevec.cu
    src/sampling/custatevec_sampler.cu
)

# Set include directories based on your tree
target_include_directories(fixed_g_custatevec PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/labs_energy
    ${CMAKE_SOURCE_DIR}/src/sampling
    ${CMAKE_SOURCE_DIR}/src/subspace
    ${CMAKE_SOURCE_DIR}/src/exact
)

# Add cuQuantum headers if CUQUANTUM_ROOT is provided
if(DEFINED ENV{CUQUANTUM_ROOT})
    target_include_directories(fixed_g_custatevec PRIVATE $ENV{CUQUANTUM_ROOT}/include)
elseif(DEFINED CUQUANTUM_ROOT)
    target_include_directories(fixed_g_custatevec PRIVATE ${CUQUANTUM_ROOT}/include)
endif()

# Link libraries
target_link_libraries(fixed_g_custatevec PRIVATE 
    CUDA::cusolver 
    CUDA::cublas 
    ${CUSTATEVEC_LIB}
)