#!/bin/bash
# ==============================================================================
# Submit 2 Grouped Probability Scan Jobs to respect Slurm QoS limits:
#   Job 1: Fast suite  (N = 16, 20, 24, 26) -> ~15-20 min
#   Job 2: Heavy suite (N = 28, 30)         -> ~1.5-2 hours
#
# Custom Partition usage (e.g. H100 vs H200):
#   PARTITION=gpu_h100_8 bash scripts/submit_grouped_prob.sh
#   PARTITION_FAST=gpu_h100_8 PARTITION_HEAVY=gpu_h200_8 bash scripts/submit_grouped_prob.sh
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PART_FAST="${PARTITION_FAST:-${PARTITION:-gpu_h100_4}}"
PART_HEAVY="${PARTITION_HEAVY:-${PARTITION:-gpu_h200_8}}"

echo "======================================================================"
echo "    Submitting Bernasconi-Krylov Grouped Probability Scans (2 Jobs)   "
echo "======================================================================"
echo "  Fast Partition  (N=16,20,24,26) : ${PART_FAST}"
echo "  Heavy Partition (N=28,30)       : ${PART_HEAVY}"
echo "======================================================================"

JOB_IDS=()

if command -v sbatch &> /dev/null; then
    # 1. Submit Fast Job
    RES_FAST=$(sbatch -p "${PART_FAST}" scripts/run_prob_fast.sbatch)
    JID_FAST=$(echo "${RES_FAST}" | awk '{print $NF}')
    JOB_IDS+=("${JID_FAST}")
    echo "[SUBMITTED] Fast Prob Scan (N=16,20,24,26) -> Job ID: ${JID_FAST}"

    # 2. Submit Heavy Job
    RES_HEAVY=$(sbatch -p "${PART_HEAVY}" scripts/run_prob_heavy.sbatch)
    JID_HEAVY=$(echo "${RES_HEAVY}" | awk '{print $NF}')
    JOB_IDS+=("${JID_HEAVY}")
    echo "[SUBMITTED] Heavy Prob Scan (N=28,30)       -> Job ID: ${JID_HEAVY}"

    # 3. Chain Paper Plotting Job after both finish
    DEP_STR="${JID_FAST}:${JID_HEAVY}"
    echo ""
    echo "[PLOTS] Submitting chained paper plotting job after jobs: ${DEP_STR}..."
    sbatch -p "${PART_FAST}" --dependency=afterok:${DEP_STR} scripts/run_paper_plots.sbatch || true
else
    echo "[DRY-RUN] sbatch not in PATH. Would submit:"
    echo "  sbatch -p ${PART_FAST} scripts/run_prob_fast.sbatch"
    echo "  sbatch -p ${PART_HEAVY} scripts/run_prob_heavy.sbatch"
fi

echo "======================================================================"
echo " Grouped submission complete! (Only 2 active compute jobs queued)"
echo "======================================================================"
