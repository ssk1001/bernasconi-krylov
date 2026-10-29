#!/bin/bash
# ==============================================================================
# Environment setup for Bernasconi-Krylov on HPC (Sharanga / Rocky 8)
# Loads Spack GCC 11.2.0 + CUDA 12.1.0 modules and connects Conda cuQuantum/Eigen
#
# Usage:
#   source scripts/setup_env.sh [conda_env_name]   (default conda env: cuquantum_env)
# ==============================================================================

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "[WARN] setup_env.sh was executed directly instead of sourced."
    echo "       Please use: source scripts/setup_env.sh [conda_env_name]"
fi

ENV_NAME="${1:-cuquantum_env}"

echo "======================================================================"
echo "          Setting up Bernasconi-Krylov HPC Environment"
echo "======================================================================"

# 1. Load Spack Environment Modules
if command -v module &> /dev/null; then
    echo "[MODULES] Purging existing modules..."
    module purge 2> /dev/null || true

    echo "[MODULES] Loading GCC..."
    module load gcc-11.2.0-gcc-8.5.0-ov3qrz6 2>/dev/null || module load gcc/11.2.0 2>/dev/null || module load gcc 2>/dev/null || true

    echo "[MODULES] Loading CUDA..."
    module load cuda-12.1.0-gcc-11.2.0-s5o57xp 2>/dev/null || module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true
else
    echo "[WARN] 'module' command not found. Skipping module load."
fi

# 2. Activate Conda Environment (for cuQuantum, Eigen, Python)
# Find and source conda.sh to ensure 'conda activate' is available in subshells
CONDA_PROFILE=""
for candidate in \
    "${CONDA_EXE%/bin/conda}/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "/apps/miniconda3/etc/profile.d/conda.sh" \
    "/apps/anaconda3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh"; do
    if [ -f "${candidate}" ]; then
        CONDA_PROFILE="${candidate}"
        break
    fi
done

if [ -n "${CONDA_PROFILE}" ]; then
    source "${CONDA_PROFILE}"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook 2> /dev/null || true)"
fi

ENV_NAME="${1:-cuquantum_env}"

if [ "${CONDA_DEFAULT_ENV}" = "${ENV_NAME}" ]; then
    echo "[CONDA] Already active in requested environment: '${ENV_NAME}'"
elif command -v conda &> /dev/null; then
    echo "[CONDA] Activating Conda environment '${ENV_NAME}'..."
    if ! conda activate "${ENV_NAME}"; then
        echo "[WARN] Could not activate '${ENV_NAME}'. Check available environments with: conda env list"
    fi
else
    echo "[WARN] 'conda' command not found. Ensure Conda is installed and added to PATH."
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

# 4. Set Cluster-Safe Matplotlib and Python Library Paths
SCRIPT_PARENT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLCONFIGDIR="/tmp/mpl_${USER:-cluster}"
if [ -d "${SCRIPT_PARENT}/lib_py" ]; then
    export PYTHONPATH="${SCRIPT_PARENT}/lib_py:${PYTHONPATH}"
fi

echo "======================================================================"
echo "[SUCCESS] Environment configured:"
echo "  GCC     : $(gcc --version 2> /dev/null | head -n 1 || echo 'Not found')"
echo "  NVCC    : $(nvcc --version 2> /dev/null | grep 'release' || echo 'Not found')"
echo "  CONDA   : ${CONDA_DEFAULT_ENV:-none} (${CONDA_PREFIX:-none})"
echo "  CUQUANTUM_ROOT: ${CUQUANTUM_ROOT:-none}"
echo "======================================================================"
