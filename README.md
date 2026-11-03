# Bernasconi-Krylov

**Finding Low-Energy States of the Bernasconi Model via Simulated Transverse-Field Quantum Dynamics**

[![C++17](https://img.shields.io/badge/standard-C%2B%2B17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![cuQuantum](https://img.shields.io/badge/cuQuantum-cuStateVec-76B900.svg)](https://developer.nvidia.com/cuquantum-sdk)
[![OpenMP](https://img.shields.io/badge/OpenMP-Multi--threaded-red.svg)](https://www.openmp.org/)
[![CTest](https://img.shields.io/badge/CTest-6%2F6%20Passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## 1. Overview & Research Motivation

The **Low-Autocorrelation Binary Sequence (LABS)** problem, formulated as the ground-state search of the **Bernasconi spin model**, is a foundational benchmark in combinatorial optimization and disordered statistical physics. Its energy landscape is notoriously rugged: broad high-energy plateaus punctuated by steep, isolated local minima (metastable traps) with virtually zero macroscopic correlation to one another.

When optimization heuristics become trapped in a local minimum $s_0$ with energy $E_0$, the objective is to escape to a configuration of lower energy ($E < E_0$). In classical perturbation heuristics (such as random restarts, multi-spin kicks followed by steepest descent, or Tabu search), escape success rates degrade precipitously as the sequence length $N$ grows, because the volume fraction of configurations with $E < E_0$ drops rapidly (from $7.6 \times 10^{-6}$ at $N=20$ to $4.5 \times 10^{-8}$ at $N=30$ for deep traps).

This repository contains the high-performance simulation and benchmarking codebase accompanying the paper:
> **Finding Low-Energy States of the Bernasconi Model via Simulated Transverse-Field Quantum Dynamics**  
> *Shaashwat Sagar Kale & Rudranil Basu* (Department of Physics, BITS Pilani)

### Core Scientific Questions Addressed:
1. **Quantum Escape Acceleration**: Can continuous-time transverse-field quantum dynamics ($H = H_{\text{LABS}} - g \sum_{i=1}^N \sigma_x^i$) initialized in symmetry-reduced classical traps escape deep barriers more effectively than budget-matched classical heuristics?
2. **Tunneling vs. Diffusion**: How does the transverse field strength $g$ balance directed barrier penetration against unguided hypercube wavepacket diffusion?
3. **Barrier Distance Dependence**: How does the escape probability $P_<$ and initial barrier traversal rise rate scale with the exact Hamming distance to the nearest lower-energy state ($d_{\min}$)?

---

## 2. Key Results & Findings

* **Substantial Advantage on Isolated Traps**: For deep, isolated traps ($d_{\min} \ge 8$), quantum dynamics achieves advantage ratios $R = P_< / r^*_{\text{classical}}$ ranging from $\sim 4.6 \times 10^1$ at $N=20$ to over $1.15 \times 10^4$ at $N=28$.
* **Optimal Transverse-Field Window**: An optimal driving field $g^* \approx 0.35$ maximizes $P_<$ while avoiding both perturbative under-driving ($g < 0.25$) and featureless high-field diffusion ($g > 0.45$).
* **Wavepacket Localization**: During quantum evolution, wavepacket spreading remains localized with mean Hamming distance $\langle d_H \rangle$ reaching only $69\% - 79\%$ of the hypercube midpoint ($N/2$).
* **Classical Heuristic Breakdown**: Across $50{,}000$ independent trials per configuration, both Steepest Descent and OpenMP-accelerated Tabu Search degrade exponentially as barrier distance $d_{\min}$ increases.

---

## 3. Architecture & Core Modules

```
bernasconi-krylov/
├── configs/              # INI / CFG benchmark configurations for N = 16 to 30
├── data_processed/       # Processed benchmark datasets (CSVs and JSON summaries)
├── figures/              # Generated plots, per-N dynamics, and publication vector figures
│   └── paper/            # Publication figures (Fig a, b, c, d in PDF and PNG)
├── manuscript/           # LaTeX source and compiled publication manuscripts
├── scripts/              # Automation, Slurm batch scripts, Colab setup, and plotting pipelines
├── src/                  # High-performance C++17 / CUDA source code
│   ├── classical/        # Classical optimizers: Steepest Descent, Tabu Search, Exhaustive search
│   ├── config/           # Robust INI/CFG configuration parser
│   ├── connector/        # Unified benchmark driver connecting traps, dynamics, and baselines
│   ├── labs_energy/      # SIMD 64-bit bitwise energy, order-8 symmetries, and exact d_min
│   ├── sampling/         # cuStateVec GPU statevector evolution and projective measurement
│   ├── subspace/         # Dense ED (Eigen) and GPU matrix-free Lanczos (cuBLAS/cuSOLVER)
│   ├── tools/            # Standalone CLI tools (run_tabu, compute_d_min_cli)
│   └── run_benchmark.cu  # Modern unified CUDA benchmark executor
└── tests/                # CTest unit test suite (CPU and CUDA tests)
```

### Module Highlights:
* **Bitwise LABS Engine (`src/labs_energy/labs_energy.hpp`)**:
  Bitwise word-level energy evaluation using SIMD `__builtin_popcountll` ($O(N)$ operations), exact $D_4 \times Z_2$ (order-8) gauge and dihedral symmetry reduction, and exact branch-and-bound barrier distance ($d_{\min}$) computation for $N \le 60$.
* **Statevector Quantum Sampler (`src/sampling/sampler.cuh`)**:
  GPU-accelerated statevector evolution under NVIDIA cuStateVec (`custatevecApplyMatrix`, `custatevecApplyPauliRotation`), supporting exact noise-free state reduction $P_<(t) = \sum_{E(s) < E_0} |\langle s | \psi(t) \rangle|^2$, expectation $\langle d_H(t) \rangle$, and projective sampling.
* **Classical Baselines & Tabu Search (`src/classical/`)**:
  GPU multi-start parallel steepest descent kernel (`classical.cuh`) alongside an OpenMP multi-threaded Tabu Search baseline (`tabu_search.hpp`) supporting Direct, Multi-spin Kick, and Random Restart modalities.
* **Exact Diagonalization & Lanczos (`src/subspace/diagonalizer.cuh`)**:
  Eigen-based dense ED for exact spectrum verification and GPU matrix-free Lanczos solver for large-scale Krylov subspace projections.

---

## 4. System Requirements

### Hardware:
* **CPU Mode**: Standard x86_64 CPU (multi-core recommended for OpenMP Tabu Search).
* **GPU Mode**: NVIDIA GPU with Compute Capability $\ge 7.0$ (Volta, Ampere, Hopper; e.g., V100, A100, H100).
  * System sizes $N \le 26$: $\le 16\text{ GB}$ VRAM.
  * System sizes $N = 28 - 30$: $24 - 32\text{ GB}$ VRAM ($2^{30} \times 16\text{ bytes} \approx 17.2\text{ GB}$ for the complex statevector).

### Software & Libraries:
* **C++ Compiler**: GCC $\ge 9.0$ or Clang $\ge 10.0$ supporting C++17.
* **Build System**: CMake $\ge 3.18$.
* **Linear Algebra**: Eigen3 (`libeigen3-dev`).
* **Parallelism**: OpenMP (`libomp-dev` / `libgomp`).
* **GPU SDK (for GPU benchmarks)**:
  * NVIDIA CUDA Toolkit $\ge 11.8$ or $12.x$.
  * NVIDIA cuQuantum SDK (`cuStateVec` library / `pip install cuquantum-cu12`).
* **Python Environment (for plots & analysis)**:
  * Python $\ge 3.9$, `numpy`, `matplotlib`, `scipy`.

---

## 5. Build Instructions

The build system automatically detects whether `nvcc` and `custatevec` are available. If CUDA is not detected, it cleanly falls back to configuring CPU-only tools and tests.

### Option A: Local CPU Build (No GPU Required)
Builds the standalone Tabu search CLI, the barrier distance CLI, and the CPU unit test suite:
```bash
# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run CPU unit test suite (6/6 tests)
ctest --test-dir build --output-on-failure
```

Alternatively, use the dedicated Tabu build script:
```bash
bash scripts/build_tabu.sh
```

### Option B: High-Performance GPU Build (HPC / Workstation)
To compile the unified GPU benchmark executor `run_benchmark`:
```bash
# If using a Conda environment with cuQuantum installed:
export CUQUANTUM_ROOT="${CONDA_PREFIX}"

cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="native" \
    -DCUQUANTUM_ROOT="${CUQUANTUM_ROOT}"

cmake --build build --target run_benchmark compute_d_min_cli run_tabu -j$(nproc)
```
Or run the automated build helper:
```bash
bash scripts/build.sh [conda_env_name]
```

### Option C: Google Colab Setup (A100 / H100)
A ready-to-run setup script is provided for Google Colab in [`scripts/colab_setup.py`](scripts/colab_setup.py) and [`scripts/colab_setup.ipynb`](scripts/colab_setup.ipynb).
In a Colab notebook with an A100 GPU:
```python
# Upload repository zip or clone repo, then run:
import scripts.colab_setup
```
This automatically installs `cuquantum-cu12`, locates `libcustatevec.so`, configures CMake, and builds `run_benchmark`.

---

## 6. Usage & CLI Tools

### 1. Unified Benchmark Executor (`run_benchmark`)
Runs full quantum dynamics, projective sampling, and classical baselines according to a configuration file:
```bash
# Run probability dynamics across transverse field g for N = 28
./build/run_benchmark configs/probability_scan_n28.cfg

# Run benchmark pipeline with customized settings
./build/run_benchmark configs/benchmark.ini
```

### 2. Standalone Tabu Search Benchmark CLI (`run_tabu`)
Runs Monte Carlo evaluations (default: $50{,}000$ trials) of Tabu Search (Direct, Kick, and Restart) against classical Steepest Descent:
```bash
# Benchmark Rank 1 trap at N = 20
./build/run_tabu --n 20 --rank 1 --trials 50000

# Benchmark across top 10 traps at N = 28 and export results CSV
./build/run_tabu --n 28 --all-ranks --max-ranks 10 --trials 50000 --output-csv data_processed/tabu_benchmark_n28.csv

# Run automated sweep across N = 20, 22, 24, 26, 28, 30
bash scripts/run_all_tabu_benchmarks.sh
```

### 3. Exact Barrier Distance CLI (`compute_d_min_cli`)
Computes the exact minimum Hamming distance $d_{\min}$ from a given spin state to any configuration with energy strictly lower than $E_0$:
```bash
# Usage: ./build/compute_d_min_cli <N> <state_hex_or_dec> <target_energy_E0>
./build/compute_d_min_cli 28 0x3a9f1b2 70
```
Or compute across all benchmark datasets via Python:
```bash
python scripts/compute_d_min.py --all-n
```

---

## 7. Reproducing Paper Figures & Analyses

All benchmark datasets used in the paper are preserved in [`data_processed/`](data_processed/). Figures can be reproduced directly without re-running long GPU simulations:

### 1. Generate Publication Figures
Generates Figures (a), (b), (c), and (d) as both vector PDFs and 300-DPI PNGs into [`figures/paper/`](figures/paper/):
```bash
python scripts/plot_paper_figures.py
```
* **Figure (a)**: Quantum advantage ratio $R = P_< / r^*_{\text{classical}}$ vs. $N$ with $B=2000$ bootstrap confidence intervals.
* **Figure (b)**: Initial rise rate $dP_</dt$ and window-averaged $P_<$ vs. exact barrier distance $d_{\min}$.
* **Figure (c)**: Escape probability $P_<$ and hypercube diffusion $\langle d_H \rangle$ vs. transverse field $g$.
* **Figure (d)**: Continuous time-evolution trajectories $P_<(t)$ for the hardest local minima across $N \in [16, 30]$.

### 2. Tabu Search vs. Classical vs. Quantum Comparative Analysis
Performs cross-method comparison across $50{,}000$ trials:
```bash
python scripts/analyze_tabu_vs_all.py
```
Outputs tabular per-evaluation success rates, scaling slopes, and exports summary records to [`data_processed/tabu_vs_classical_quantum_comparison.json`](data_processed/tabu_vs_classical_quantum_comparison.json).

### 3. Compile Paper & Thesis
```bash
pdflatex paper.tex
# or
latexmk -pdf paper.tex
```

---

## 8. Unit Test Suite

The test suite validates energy calculations, symmetries, configurations, exact diagonalization, and classical algorithms:

```bash
ctest --test-dir build --output-on-failure
```

| Test Name | Target Source | Verification Scope |
| :--- | :--- | :--- |
| `test_labs_energy` | `tests/test_labs_energy.cpp` | Bitwise LABS energy, merit factor, order-8 dihedral $\times$ gauge symmetry invariance |
| `test_d_min` | `tests/test_d_min.cpp` | Exact branch-and-bound barrier distance $d_{\min}$ calculations |
| `test_config` | `tests/test_config.cpp` | INI/CFG parser robustness, edge cases, whitespace handling |
| `test_connector` | `tests/test_connector.cpp` | Connector data structures, metrics, and serialization |
| `test_tabu_search` | `tests/test_tabu_search.cpp` | Tabu tenure list, dynamic tenure, single-step move evaluation, monotonic descent |
| `test_exact_diag` | `tests/test_exact_diag.cpp` | Dense exact diagonalization spectrum and ground-state energies against known LABS levels |

When built with CUDA and cuQuantum enabled, additional tests (`test_lanczos`, `test_sampler`, `test_steepest_descent`) are automatically registered.

---

## 9. Citation & Contact

If you find this codebase or research useful, please cite:

```bibtex
@article{kale2026bernasconi,
  title   = {Finding Low-Energy States of the Bernasconi Model via Simulated Transverse-Field Quantum Dynamics},
  author  = {Kale, Shaashwat Sagar and Basu, Rudranil},
  journal = {Department of Physics, BITS Pilani},
  year    = {2026}
}
```

For questions or issues, please contact `f20220206@goa.bits-pilani.ac.in` or open an issue on GitHub.