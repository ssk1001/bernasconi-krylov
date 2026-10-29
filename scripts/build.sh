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

echo ""
echo "[BUILD] Configuring CMake for NVIDIA H200 (sm_90)..."
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=90 \
      -DCUQUANTUM_ROOT="${CONDA_PREFIX}" \
      -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}"

echo ""
echo "[BUILD] Compiling run_benchmark..."
cmake --build build --target run_benchmark -j4

echo ""
echo "======================================================================"
echo "[SUCCESS] Build complete! Binary located at:"
echo "  - ${ROOT_DIR}/build/run_benchmark"
echo "======================================================================"
