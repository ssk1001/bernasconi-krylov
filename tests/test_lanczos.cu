#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>

#include "../src/subspace/diagonalizer.cuh"

// Test 1: Full Space Lanczos vs Dense ED
void test_fullspace_lanczos_vs_ed() {
    std::cout << "[Test 1] Testing Full Space Lanczos vs Dense ED..." << std::endl;
    int n = 4;
    double g = 0.5;

    Diagonalizer diag;

    // 1. Solve via Dense ED (Ground Truth)
    DiagConfig ed_cfg;
    ed_cfg.method = DiagMethod::ExactDense;
    ed_cfg.normalize_by_n = true;
    Eigensolution ed_sol = diag.solve(n, g, ed_cfg);

    // 2. Solve via GPU Lanczos
    DiagConfig lanczos_cfg;
    lanczos_cfg.method = DiagMethod::Lanczos;
    lanczos_cfg.max_krylov_dim = 30;
    lanczos_cfg.normalize_by_n = true;
    Eigensolution lanczos_sol = diag.solve(n, g, lanczos_cfg);

    std::cout << "  Dense ED Ground Energy   : " << std::fixed << std::setprecision(8) 
              << ed_sol.ground_energy() << "\n";
    std::cout << "  GPU Lanczos Ground Energy: " 
              << lanczos_sol.ground_energy() << "\n";

    // Energies must match to high numerical precision
    double diff = std::abs(ed_sol.ground_energy() - lanczos_sol.ground_energy());
    assert(diff < 1e-6);

    // Verify residual ||H v - E v||_2 for the Lanczos solution
    int dim = 1 << n;
    LabsEvaluator eval(n);
    double scale = 1.0 / static_cast<double>(n);
    double res_sq = 0.0;

    const auto& v = lanczos_sol.ground_state();
    double E = lanczos_sol.ground_energy();

    for (int i = 0; i < dim; ++i) {
        double Hv_i = (eval.calcEnergy(SpinState{static_cast<uint64_t>(i)}) * scale) * v[i];
        for (int j = 0; j < n; ++j) {
            int nbr = i ^ (1 << j);
            Hv_i -= g * v[nbr];
        }
        double r_i = Hv_i - E * v[i];
        res_sq += r_i * r_i;
    }
    double res_norm = std::sqrt(res_sq);
    std::cout << "  Lanczos Eigen-residual ||H v - E v||_2: " << std::scientific << res_norm << "\n";
    assert(res_norm < 1e-5);

    std::cout << "  --> PASSED: GPU Lanczos matches Dense ED with near-zero residual!\n" << std::endl;
}

// Test 2: Subspace Lanczos vs Subspace Dense ED
void test_subspace_lanczos_vs_ed() {
    std::cout << "[Test 2] Testing Subspace Lanczos vs Subspace Dense ED..." << std::endl;
    int n = 6;
    double g = 0.35;

    // Construct a subspace basis of 16 states
    std::vector<SpinState> basis;
    for (uint64_t i = 0; i < 16; ++i) {
        basis.push_back(SpinState{i * 3ULL + 1ULL});
    }

    Diagonalizer diag;

    // 1. Subspace Dense ED
    DiagConfig ed_cfg;
    ed_cfg.method = DiagMethod::ExactDense;
    ed_cfg.normalize_by_n = true;
    Eigensolution ed_sol = diag.solve(basis, g, n, ed_cfg);

    // 2. Subspace GPU Lanczos
    DiagConfig lanczos_cfg;
    lanczos_cfg.method = DiagMethod::Lanczos;
    lanczos_cfg.max_krylov_dim = 16;
    lanczos_cfg.normalize_by_n = true;
    Eigensolution lanczos_sol = diag.solve(basis, g, n, lanczos_cfg);

    std::cout << "  Subspace Dim=" << basis.size() << "\n";
    std::cout << "  Dense ED Subspace Energy   : " << std::fixed << std::setprecision(8) 
              << ed_sol.ground_energy() << "\n";
    std::cout << "  GPU Lanczos Subspace Energy: " 
              << lanczos_sol.ground_energy() << "\n";

    double diff = std::abs(ed_sol.ground_energy() - lanczos_sol.ground_energy());
    assert(diff < 1e-6);

    std::cout << "  --> PASSED: Subspace Lanczos matches Subspace Dense ED!\n" << std::endl;
}

int main() {
    std::cout << "=====================================================\n";
    std::cout << "              Testing GPU Lanczos Solver             \n";
    std::cout << "=====================================================\n";

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "  [Notice] No CUDA device detected. Skipping GPU Lanczos tests.\n" << std::endl;
        return 0;
    }

    test_fullspace_lanczos_vs_ed();
    test_subspace_lanczos_vs_ed();

    std::cout << "=====================================================\n";
    std::cout << "       ALL GPU LANCZOS TESTS PASSED SUCCESSFULLY!    \n";
    std::cout << "=====================================================" << std::endl;
    return 0;
}
