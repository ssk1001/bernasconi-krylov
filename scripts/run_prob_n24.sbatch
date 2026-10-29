#!/bin/bash
#SBATCH --job-name=prob_n24
#SBATCH --partition=gpu_h200_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

echo "=== Starting Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID}) on $(hostname) at $(date) ==="
echo "--- GPU Allocation ---"
nvidia-smi

cd "${SLURM_SUBMIT_DIR}"

# 1. Environment setup (Spack modules + Conda)
source scripts/setup_env.sh cuquantum_env

# 2. Ensure directories exist
mkdir -p data_processed figures/n24 seeds

# 3. Run Probability Dynamics for N=24 (10 traps, g in {0.3, 0.35, 0.45}, t in [0, 60])
echo "--- Running Probability Dynamics (N=24) ---"
./build/run_benchmark configs/probability_scan_n24.cfg

# 4. Generate Individual Images for all 10 Traps and Overview Grid
echo "--- Generating Plots and Summary ---"
python3 scripts/plot_probability_dynamics.py \
    data_processed/probability_dynamics_n24.csv \
    figures/n24

# 5. Compute d_min, rise rates, and update structured JSON
echo "--- Updating Structured JSON with d_min and Rise Rates ---"
python3 scripts/compute_d_min.py \
    --input data_processed/probability_dynamics_n24_results.csv \
    --seeds seeds/seeds_N24.csv \
    --output data_processed/structured_benchmark_n24.json

echo "=== Job Completed with status $? at $(date) ==="
