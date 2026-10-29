#!/bin/bash
#SBATCH --job-name=paper_plots
#SBATCH --partition=gpu_h200_8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

echo "=== Starting Job: ${SLURM_JOB_NAME} (ID: ${SLURM_JOB_ID}) on $(hostname) at $(date) ==="

cd "${SLURM_SUBMIT_DIR}"

# 1. Environment setup (Spack modules + Conda)
source scripts/setup_env.sh cuquantum_env

# 2. Ensure directories exist
mkdir -p figures/paper data_processed

# 3. Generate all 4 publication figures and plot_data_summary.json
echo "--- Generating All 4 Publication Figures ---"
python3 scripts/plot_paper_figures.py \
    --data-dir data_processed \
    --output-dir figures/paper \
    --bootstrap-samples 2000

echo "=== Job Completed with status $? at $(date) ==="
