cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(FixedG_Subspace_Solver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimize for performance
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
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
    ${CMAKE_SOURCE_DIR}/src/classical
    ${CMAKE_SOURCE_DIR}/src/baselines
)

# Add cuQuantum headers if CUQUANTUM_ROOT is provided
if(DEFINED ENV{CUQUANTUM_ROOT})
    target_include_directories(fixed_g_custatevec PRIVATE $ENV{CUQUANTUM_ROOT}/include)
elseif(DEFINED CUQUANTUM_ROOT)
    target_include_directories(fixed_g_custatevec PRIVATE ${CUQUANTUM_ROOT}/include)
endif()

# Link libraries (ORDER MATTERS: cuStateVec must come BEFORE cuBLAS/cuSOLVER)
target_link_libraries(fixed_g_custatevec PRIVATE 
    ${CUSTATEVEC_LIB}
    CUDA::cublas 
    CUDA::cusolver 
    Eigen3::Eigen
)
if(EXISTS "/usr/local/cuda/lib64")
    target_link_options(fixed_g_custatevec PRIVATE "-Wl,-rpath-link,/usr/local/cuda/lib64")
endif()

# Probability dynamics & g-scan executable
add_executable(probability_dynamics 
    src/probability_dynamics.cu
    src/sampling/custatevec_sampler.cu
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