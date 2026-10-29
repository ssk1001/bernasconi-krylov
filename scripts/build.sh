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

CUDA_ARCH="${CUDA_ARCH:-90}"
echo ""
echo "[BUILD] Configuring CMake for GPU architecture (sm_${CUDA_ARCH})..."
CMAKE_ARGS=(
    -B build
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
)
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
echo "[BUILD] Compiling run_benchmark and compute_d_min_cli with ${NUM_JOBS} threads..."
cmake --build build --target run_benchmark --target compute_d_min_cli -j"${NUM_JOBS}"

echo ""
echo "======================================================================"
echo "[SUCCESS] Build complete! Binaries located at:"
echo "  - ${ROOT_DIR}/build/run_benchmark"
echo "  - ${ROOT_DIR}/build/compute_d_min_cli"
echo "======================================================================"
