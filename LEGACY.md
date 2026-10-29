# Legacy & Redundant Files Registry

This document lists legacy and redundant files superseded during the modern modular rewrite of the `bernasconi-krylov` library.

---

## 1. Classical Optimization & Symmetry

| Legacy File | Replacement | Notes |
| :--- | :--- | :--- |
| `src/classical/steepest_descent.cuh` | `src/classical/classical.cuh` | Superseded by unified CPU incremental descent and register-resident GPU multi-start descent kernel. |
| `src/classical/sd_collector.cuh` | `src/classical/classical.cuh` | Minima collection and frequency tracking are now handled by `ClassicalOptimizer::collect_distinct_minima_*`. |
| `src/classical/symmetry.hpp` | `src/labs_energy/labs_energy.hpp` | Order-8 symmetry group operations and canonical representative reduction are now native methods of `SpinState` (`canonical(n)`). |
| `src/classical/seed_manager.hpp` | `src/classical/classical.cuh` | Seed generation and CSV metadata logging are unified in `ClassicalOptimizer` and `ClassicalLogger`. |
| `src/baselines/budgeted_baselines.hpp` | `src/classical/classical.cuh` | Random Restarts and Random Kick are unified with strict budget enforcement in `ClassicalOptimizer::random_restarts` and `random_kick`. |

---

## 2. Diagonalization & Lanczos Solvers

| Legacy File | Replacement | Notes |
| :--- | :--- | :--- |
| `src/exact/exact_diag.hpp` | `src/subspace/diagonalizer.cuh` | Dense Exact Diagonalization (ED) via Eigen for both full-space ($2^N$) and arbitrary subspaces is unified in `Diagonalizer::solve_*_dense()`. |
| `src/subspace/lanczos.cuh` | `src/subspace/diagonalizer.cuh` | GPU matrix-free Lanczos via cuBLAS/cuSOLVER for both full-space and arbitrary subspaces is unified in `Diagonalizer::solve_*_lanczos()`. |

---

## 3. Top-Level Executors & Configuration

| Legacy File | Status | Notes |
| :--- | :--- | :--- |
| `src/fixed_g_custatevec.cu` | Deprecated | Superseded by modern unified connector `src/connector/connector.cuh` and executor `src/run_benchmark.cu`. |
| `src/probability_dynamics.cu` | Deprecated | Superseded by modern unified connector `src/connector/connector.cuh` and executor `src/run_benchmark.cu`. |
| `src/benchmark_config.hpp` | Deprecated | Superseded by modern unified parser in `src/config/config.hpp`. |

---

## 4. Legacy Unit Tests

| Legacy Test | Replacement | Notes |
| :--- | :--- | :--- |
| `tests/test_baselines.cpp` | `tests/test_steepest_descent.cu` | Tests superseded `budgeted_baselines.hpp`. |
| `tests/test_symmetry.cpp` | `tests/test_labs_energy.cpp` | Tests superseded `symmetry.hpp`. |
| `tests/test_seed_manager.cpp` | `tests/test_steepest_descent.cu` | Tests superseded `seed_manager.hpp`. |
| `tests/test_config.cpp` | `tests/test_config.cpp` | Modernized test suite verifying `src/config/config.hpp`. |
