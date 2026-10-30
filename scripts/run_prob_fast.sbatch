#!/bin/bash
#SBATCH --job-name=prob_fast
#SBATCH --partition=gpu_h100_4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_%x_%j.out
#SBATCH --error=slurm_%x_%j.err

echo "=== Starting Grouped Fast Probability Scans (N = 16, 20, 24, 26) on $(hostname) at $(date) ==="
echo "--- GPU Allocation ---"
nvidia-smi

cd "${SLURM_SUBMIT_DIR}"

# 1. Environment setup (Spack modules + Conda)
source scripts/setup_env.sh cuquantum_env

FAST_N=(16 20 24 26)

for n in "${FAST_N[@]}"; do
    echo "======================================================================"
    echo "  [Fast Prob Scan] Starting N = ${n} at $(date)"
    echo "======================================================================"
    mkdir -p data_processed "figures/n${n}" seeds

    # 1. Run Probability Dynamics
    ./build/run_benchmark "configs/probability_scan_n${n}.cfg"

    # 2. Generate Trap Plots and Overview Grid
    python3 scripts/plot_probability_dynamics.py \
        "data_processed/probability_dynamics_n${n}.csv" \
        "figures/n${n}"

    # 3. Compute exact d_min, rise rates, and update structured JSON
    python3 scripts/compute_d_min.py \
        --input "data_processed/probability_dynamics_n${n}_results.csv" \
        --seeds "seeds/seeds_N${n}.csv" \
        --output "data_processed/structured_benchmark_n${n}.json"
done

echo "=== Grouped Fast Probability Scans Completed Successfully at $(date) ==="
