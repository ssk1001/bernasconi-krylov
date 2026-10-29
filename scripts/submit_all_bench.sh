#!/bin/bash
# ==============================================================================
# Submit separate Slurm benchmark jobs for each system size N
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

N_VALUES=(16 20 24 26 28 30)
JOB_IDS=()

echo "======================================================================"
echo "    Submitting Bernasconi-Krylov Separate Benchmark Jobs (Per-N)      "
echo "======================================================================"

for n in "${N_VALUES[@]}"; do
    SBATCH_FILE="scripts/run_bench_n${n}.sbatch"
    if [ -f "${SBATCH_FILE}" ]; then
        if command -v sbatch &> /dev/null; then
            RES=$(sbatch "${SBATCH_FILE}")
            JID=$(echo "${RES}" | awk '{print $NF}')
            JOB_IDS+=("${JID}")
            echo "[SUBMITTED] N = ${n} -> Job ID: ${JID} (${SBATCH_FILE})"
        else
            echo "[DRY-RUN] sbatch not in PATH. Would submit: sbatch ${SBATCH_FILE}"
        fi
    else
        echo "[WARN] Script not found: ${SBATCH_FILE}"
    fi
done

if command -v sbatch &> /dev/null && [ ${#JOB_IDS[@]} -gt 0 ]; then
    DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
    echo ""
    echo "[PLOTS] Submitting chained paper plotting job after jobs: ${DEP_STR}..."
    sbatch --dependency=afterok:${DEP_STR} scripts/run_paper_plots.sbatch || true
fi

echo "======================================================================"
echo " Submission complete!"
echo "======================================================================"
