#!/bin/bash
# ==============================================================================
# Local Build & Run Script for Tabu Search Benchmark
# Compiles and runs standalone Tabu Search on local CPU without CUDA dependency
#
# Usage:
#   bash scripts/build_tabu.sh [--run [args...]]
#
# Examples:
#   bash scripts/build_tabu.sh
#   bash scripts/build_tabu.sh --run --n 20 --trials 50000 --rank 1
#   bash scripts/build_tabu.sh --run --n 28 --trials 50000 --rank 1
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
mkdir -p build

echo "======================================================================"
echo "          Building Standalone Tabu Search Benchmark Suite             "
echo "======================================================================"

if [ -z "${CXX}" ] || ! command -v "${CXX}" >/dev/null 2>&1; then
    CXX="/usr/bin/g++"
fi
CXXFLAGS="-O3 -std=c++17 -Wall -Wextra -Isrc -Isrc/labs_energy -Isrc/classical"

echo "Compiling tests/test_tabu_search.cpp -> build/test_tabu_search..."
${CXX} ${CXXFLAGS} tests/test_tabu_search.cpp -o build/test_tabu_search

echo "Compiling src/tools/run_tabu.cpp -> build/run_tabu..."
${CXX} ${CXXFLAGS} src/tools/run_tabu.cpp -o build/run_tabu

echo ""
echo "[SUCCESS] Build complete! Binaries created:"
echo "  - build/test_tabu_search  (Unit test suite)"
echo "  - build/run_tabu          (50,000-trial local benchmark CLI)"
echo "======================================================================"

if [ "$1" == "--run" ]; then
    shift
    echo ""
    echo "[Running local benchmark CLI with args: $@]"
    echo ""
    ./build/run_tabu "$@"
elif [ "$1" == "--test" ]; then
    echo ""
    echo "[Running Tabu Search unit tests...]"
    echo ""
    ./build/test_tabu_search
fi
