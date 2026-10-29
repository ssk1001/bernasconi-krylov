#!/bin/bash
#SBATCH --job-name=bench_n26
#SBATCH --partition=gpu_h200_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

echo "=== Starting Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID}) on $(hostname) at $(date) ==="
echo "--- GPU Allocation ---"
nvidia-smi

cd "${SLURM_SUBMIT_DIR}"

# 1. Environment setup (Spack modules + Conda)
source scripts/setup_env.sh cuquantum_env

# 2. Ensure directories exist
mkdir -p data_processed seeds

# 3. Run Delayed Window Benchmark for N=26 (g in {0.3, 0.35, 0.45})
echo "--- Running Delayed Window Benchmark (N=26) ---"
./build/run_benchmark configs/n26.ini

echo "=== Job Completed with status $? at $(date) ==="
