#!/bin/bash
# ==============================================================================
# Build script for Bernasconi-Krylov on HPC
# Sources environment and compiles fixed_g_custatevec & probability_dynamics
#
# Usage:
#   bash scripts/build.sh [conda_env_name]   (default conda env: cuquantum_env)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source environment
source "${SCRIPT_DIR}/setup_env.sh" "${1:-cuquantum_env}"

cd "${ROOT_DIR}"

# Purge stale CMakeCache if it was poisoned by a previous failed check
if [ -f "build/CMakeCache.txt" ]; then
    if grep -qE "CUDA_COMPILER-NOTFOUND|CUSTATEVEC_LIB-NOTFOUND" build/CMakeCache.txt 2>/dev/null; then
        echo "[BUILD] Purging stale CMakeCache.txt containing NOTFOUND entries..."
        rm -f build/CMakeCache.txt
    fi
fi

CUDA_ARCH="${CUDA_ARCH:-90}"
echo ""
echo "[BUILD] Configuring CMake for GPU architecture (sm_${CUDA_ARCH})..."
CMAKE_ARGS=(
    -B build
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
    -DREQUIRE_CUDA=ON
)

if command -v nvcc &> /dev/null; then
    NVCC_BIN="$(which nvcc)"
    echo "[BUILD] Explicit CUDA compiler: ${NVCC_BIN}"
    CMAKE_ARGS+=( -DCMAKE_CUDA_COMPILER="${NVCC_BIN}" )
fi
if command -v gcc &> /dev/null; then
    GCC_BIN="$(which gcc)"
    CMAKE_ARGS+=( -DCMAKE_CUDA_HOST_COMPILER="${GCC_BIN}" )
fi

if [ -n "${CONDA_PREFIX}" ]; then
    CMAKE_ARGS+=(
        -DCUQUANTUM_ROOT="${CONDA_PREFIX}"
        -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}"
    )
elif [ -n "${CUQUANTUM_ROOT}" ]; then
    CMAKE_ARGS+=(
        -DCUQUANTUM_ROOT="${CUQUANTUM_ROOT}"
    )
fi

cmake "${CMAKE_ARGS[@]}"

NUM_JOBS="$(nproc 2>/dev/null || echo 4)"
echo ""
echo "[BUILD] Compiling binaries with ${NUM_JOBS} threads..."
cmake --build build -j"${NUM_JOBS}"

echo ""
if [ ! -f "${ROOT_DIR}/build/run_benchmark" ]; then
    echo "======================================================================"
    echo "[FATAL ERROR] 'build/run_benchmark' was not generated!"
    echo "Check the CMake log above: verify nvcc is in PATH and cuQuantum is installed."
    echo "======================================================================"
    exit 1
fi

echo "======================================================================"
echo "[SUCCESS] Build complete! Binaries located in: ${ROOT_DIR}/build"
echo "  - ${ROOT_DIR}/build/run_benchmark (GPU Benchmark Pipeline)"
if [ -f "${ROOT_DIR}/build/compute_d_min_cli" ]; then
    echo "  - ${ROOT_DIR}/build/compute_d_min_cli"
fi
if [ -f "${ROOT_DIR}/build/exhaustive_minima" ]; then
    echo "  - ${ROOT_DIR}/build/exhaustive_minima"
fi
if [ -f "${ROOT_DIR}/build/run_tabu" ]; then
    echo "  - ${ROOT_DIR}/build/run_tabu"
fi
echo "======================================================================"
