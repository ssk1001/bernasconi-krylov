#!/bin/bash
#SBATCH --job-name=bench_fast
#SBATCH --partition=gpu_h100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

set -e

echo "=== Starting Grouped Fast Benchmark (N = 20, 22, 24, 26) on $(hostname) at $(date) ==="
echo "--- GPU Allocation ---"
nvidia-smi

cd "${SLURM_SUBMIT_DIR:-.}"

# 1. Environment setup (Spack modules + Conda)
source scripts/setup_env.sh cuquantum_env

# 2. Ensure directories exist
mkdir -p data_processed seeds

# 4. Verify or build benchmark executable
if [ ! -f "build/run_benchmark" ]; then
    echo "[BUILD] build/run_benchmark not found. Building now..."
    bash scripts/build.sh cuquantum_env
fi

if [ ! -f "build/run_benchmark" ]; then
    echo "[FATAL ERROR] 'build/run_benchmark' does not exist! Compilation failed or was skipped."
    exit 1
fi

FAST_N=(20 22 24 26)

for n in "${FAST_N[@]}"; do
    echo "======================================================================"
    echo "  [Fast Suite] Running Delayed-Window Benchmark for N = ${n} at $(date)"
    echo "======================================================================"
    ./build/run_benchmark "configs/n${n}.ini"
    STATUS=$?
    if [ ${STATUS} -ne 0 ]; then
        echo "[ERROR] Benchmark failed for N = ${n} with exit code ${STATUS}!"
        exit ${STATUS}
    fi
done

echo "=== Grouped Fast Benchmark Completed Successfully at $(date) ==="
