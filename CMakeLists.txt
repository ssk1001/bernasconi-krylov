cmake_minimum_required(VERSION 3.18)
project(BernasconiKrylov LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

find_library(PETSC_LIB petsc HINTS $ENV{PETSC_DIR}/lib $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib /usr/lib/petsc/lib /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(SLEPC_LIB slepc HINTS $ENV{SLEPC_DIR}/lib $ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib /usr/lib/slepc/lib /usr/lib/x86_64-linux-gnu /usr/lib)

if(NOT PETSC_LIB OR NOT SLEPC_LIB)
    message(FATAL_ERROR "PETSc or SLEPc libraries not found. Please verify they are installed.")
endif()


# 1. MPI Integration
find_package(MPI REQUIRED)

# 2. cuQuantum & cuBLAS Integration
find_path(CUQUANTUM_INCLUDE_DIR custatevec.h HINTS /opt/cuquantum/include /opt/nvidia/cuquantum/include)
find_library(CUSTATEVEC_LIB custatevec HINTS /opt/cuquantum/lib /opt/nvidia/cuquantum/lib64)
find_library(CUBLAS_LIB cublas HINTS /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

if(NOT CUQUANTUM_INCLUDE_DIR OR NOT CUSTATEVEC_LIB)
    message(FATAL_ERROR "cuQuantum not found. Check your installation paths.")
endif()

# 3. PETSc & SLEPc Integration
# Relies on $PETSC_DIR, $PETSC_ARCH, and $SLEPC_DIR being set in your environment
if(NOT DEFINED ENV{PETSC_DIR} OR NOT DEFINED ENV{SLEPC_DIR})
    message(FATAL_ERROR "Environment variables PETSC_DIR and SLEPC_DIR must be set.")
endif()

set(PETSC_INCLUDES $ENV{PETSC_DIR}/include $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/include)
set(SLEPC_INCLUDES $ENV{SLEPC_DIR}/include $ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/include)

set(PETSC_LIB_DIR $ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib)
set(SLEPC_LIB_DIR $ENV{SLEPC_DIR}/$ENV{PETSC_ARCH}/lib)

# 4. Source Files
set(SOURCES
    src/fixed_g_custatevec.cu
    src/sampling/custatevec_sampler.cu
    src/classical/steepest_descent.cuh
    src/labs_energy/labs_energy.hpp
    src/subspace/lanczos.hpp
    # Append your steepest descent / classical .cpp files here
)

add_executable(krylov_sampler ${SOURCES})

# 5. Include Directories
target_include_directories(krylov_sampler PRIVATE 
    ${MPI_CXX_INCLUDE_DIRS}
    ${CUQUANTUM_INCLUDE_DIR}
    ${PETSC_INCLUDES}
    ${SLEPC_INCLUDES}
    src/
)

# 6. Linking Phase
target_link_libraries(krylov_sampler PRIVATE 
    MPI::MPI_CXX
    ${CUSTATEVEC_LIB}
    ${CUBLAS_LIB}
    ${PETSC_LIB}
    ${SLEPC_LIB}
)