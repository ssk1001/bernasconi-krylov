#!/bin/bash
# ==============================================================================
# Run Tabu Search benchmarks across available system sizes N = 20 through 30
# Evaluates all 10 benchmark ranks for each N with 50,000 Monte Carlo trials
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

TRIALS=${1:-50000}
KICK_D="2,4,6,8,10"
MAX_RANKS=10

echo "======================================================================"
echo " Starting Tabu Search Benchmarking Suite across N in [20, 22, 24, 26, 28, 30]"
echo " Trials per configuration : ${TRIALS}"
echo " Kick distances           : ${KICK_D}"
echo " Benchmark ranks per N    : ${MAX_RANKS}"
echo "======================================================================"

mkdir -p data_processed

for N in 20 22 24 26 28 30; do
    echo ""
    echo "======================================================================"
    echo " Running Tabu Search Benchmark for N = ${N} (50,000 trials, top ${MAX_RANKS} ranks) ..."
    echo "======================================================================"
    
    OUT_CSV="data_processed/tabu_benchmark_n${N}.csv"
    
    ./build/run_tabu \
        --n "${N}" \
        --all-ranks \
        --max-ranks "${MAX_RANKS}" \
        --trials "${TRIALS}" \
        --kick-d "${KICK_D}" \
        --output-csv "${OUT_CSV}"
        
    echo "[SUCCESS] Finished N = ${N}. Results saved to: ${OUT_CSV}"
done

echo ""
echo "======================================================================"
echo " All Tabu Search benchmarks across N = 20..30 completed successfully!"
echo "======================================================================"
