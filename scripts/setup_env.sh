#!/bin/bash
# ==============================================================================
# Environment setup for Bernasconi-Krylov on HPC (Sharanga / Rocky 8)
# Loads Spack GCC 11.2.0 + CUDA 12.1.0 modules and connects Conda cuQuantum/Eigen
#
# Usage:
#   source scripts/setup_env.sh [conda_env_name]   (default conda env: qsim)
# ==============================================================================

ENV_NAME="${1:-qsim}"

echo "======================================================================"
echo "          Setting up Bernasconi-Krylov HPC Environment"
echo "======================================================================"

# 1. Load Spack Environment Modules
if command -v module &> /dev/null; then
    echo "[MODULES] Purging existing modules..."
    module purge 2> /dev/null || true

    echo "[MODULES] Loading GCC 11.2.0 (gcc-11.2.0-gcc-8.5.0-ov3qrz6)..."
    module load gcc-11.2.0-gcc-8.5.0-ov3qrz6

    echo "[MODULES] Loading CUDA 12.1.0 (cuda-12.1.0-gcc-11.2.0-s5o57xp)..."
    module load cuda-12.1.0-gcc-11.2.0-s5o57xp
else
    echo "[WARN] 'module' command not found. Skipping module load."
fi

# 2. Activate Conda Environment (for cuQuantum, Eigen, Python)
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook 2> /dev/null || true)"
    echo "[CONDA] Activating Conda environment '${ENV_NAME}'..."
    conda activate "${ENV_NAME}" 2> /dev/null || echo "[WARN] Could not activate conda env '${ENV_NAME}'. Using active environment."
elif [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    echo "[CONDA] Activating Conda environment '${ENV_NAME}'..."
    conda activate "${ENV_NAME}"
elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
    echo "[CONDA] Activating Conda environment '${ENV_NAME}'..."
    conda activate "${ENV_NAME}"
fi

# 3. Export Linker, Header, and Library Paths
if [ -n "${CONDA_PREFIX}" ]; then
    echo "[ENV] Linking Conda prefix: ${CONDA_PREFIX}"
    export CUQUANTUM_ROOT="${CONDA_PREFIX}"
    export CMAKE_PREFIX_PATH="${CONDA_PREFIX}:${CMAKE_PREFIX_PATH}"
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
    export CPATH="${CONDA_PREFIX}/include:${CONDA_PREFIX}/include/eigen3:${CPATH}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
else
    echo "[WARN] CONDA_PREFIX is not set. Ensure cuQuantum and Eigen are in standard paths."
fi

echo "======================================================================"
echo "[SUCCESS] Environment configured:"
echo "  GCC     : $(gcc --version 2> /dev/null | head -n 1 || echo 'Not found')"
echo "  NVCC    : $(nvcc --version 2> /dev/null | grep 'release' || echo 'Not found')"
echo "  CONDA   : ${CONDA_DEFAULT_ENV:-none} (${CONDA_PREFIX:-none})"
echo "  CUQUANTUM_ROOT: ${CUQUANTUM_ROOT:-none}"
echo "======================================================================"
