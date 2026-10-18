#!/bin/bash
#SBATCH --job-name=prob_n28
#SBATCH --partition=gpu_h200_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

echo "=== Starting Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID}) on $(hostname) at $(date) ==="
echo "--- GPU Allocation ---"
nvidia-smi

cd "${SLURM_SUBMIT_DIR}"

# 1. Environment setup (Spack modules + Conda)
source scripts/setup_env.sh qsim

# 2. Ensure directories exist
mkdir -p data_processed figures

# 3. Run Probability Dynamics for N=28 (g in {0.3, 0.35, 0.45}, t in [0, 60])
echo "--- Running Probability Dynamics (N=28) ---"
./build/probability_dynamics configs/probability_scan_n28.cfg

# 4. Generate Plot and Print Summary
echo "--- Generating Plot and Summary ---"
python3 scripts/plot_probability_dynamics.py \
    data_processed/probability_dynamics_n28.csv \
    figures/probability_dynamics_n28.png

echo "=== Job Completed with status $? at $(date) ==="
