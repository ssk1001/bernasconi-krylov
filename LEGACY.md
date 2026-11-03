# Legacy & Redundant Files Registry

This document lists legacy and redundant files that were superseded and purged during the modern modular rewrite of the `bernasconi-krylov` library.

---

## 1. Classical Optimization & Symmetry

| Purged Legacy File | Modern Replacement | Notes |
| :--- | :--- | :--- |
| `src/classical/steepest_descent.cuh` | `src/classical/classical.cuh` | Superseded by unified CPU incremental descent and register-resident GPU multi-start descent kernel. |
| `src/classical/sd_collector.cuh` | `src/classical/classical.cuh` | Minima collection and frequency tracking are now handled by `ClassicalOptimizer::collect_distinct_minima_*`. |
| `src/classical/symmetry.hpp` | `src/labs_energy/labs_energy.hpp` | Order-8 symmetry group operations and canonical representative reduction are native methods of `SpinState` (`canonical(n)`). |
| `src/classical/seed_manager.hpp` | `src/classical/classical.cuh` | Seed generation and CSV metadata logging are unified in `ClassicalOptimizer` and `ClassicalLogger`. |
| `src/baselines/budgeted_baselines.hpp` | `src/classical/classical.cuh` | Random Restarts and Random Kick are unified with strict budget enforcement in `ClassicalOptimizer::random_restarts` and `random_kick`. |
| `src/classical/tabu_search.hpp` | (New active subsystem) | Comprehensive OpenMP-accelerated Tabu Search baseline (Direct, Kick, and Restart). |

---

## 2. Diagonalization & Lanczos Solvers

| Purged Legacy File | Modern Replacement | Notes |
| :--- | :--- | :--- |
| `src/exact/exact_diag.hpp` | `src/subspace/diagonalizer.cuh` | Dense Exact Diagonalization (ED) via Eigen for both full-space ($2^N$) and arbitrary subspaces is unified in `Diagonalizer::solve_*_dense()`. |
| `src/subspace/lanczos.cuh` | `src/subspace/diagonalizer.cuh` | GPU matrix-free Lanczos via cuBLAS/cuSOLVER for both full-space and arbitrary subspaces is unified in `Diagonalizer::solve_*_lanczos()`. |

---

## 3. Top-Level Executors & Configuration

| Purged Legacy File | Modern Replacement | Notes |
| :--- | :--- | :--- |
| `src/fixed_g_custatevec.cu` | `src/connector/connector.cuh`, `src/run_benchmark.cu` | Superseded by modern unified connector and benchmark executor. |
| `src/probability_dynamics.cu` | `src/connector/connector.cuh`, `src/run_benchmark.cu` | Superseded by modern unified connector and benchmark executor. |
| `src/benchmark_config.hpp` | `src/config/config.hpp` | Superseded by modern unified parser in `src/config/config.hpp`. |
| `src/tools/generate_benchmarks_all_n.cpp` | `scripts/compute_d_min.py --all-n` | Superseded by modular offline analysis and dataset ingestion script. |

---

## 4. Unit Test Suite

The active test suite is configured directly in `CMakeLists.txt` (`BUILD_UNIT_TESTS`):
- `tests/test_labs_energy.cpp`
- `tests/test_d_min.cpp`
- `tests/test_config.cpp`
- `tests/test_connector.cpp`
- `tests/test_tabu_search.cpp`
- `tests/test_exact_diag.cpp`
