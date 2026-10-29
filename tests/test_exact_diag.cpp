#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cmath>

#include "../src/subspace/diagonalizer.cuh"

// Test 1: Full space dense ED for small N
void test_fullspace_dense_ed() {
    std::cout << "[Test 1] Testing Full Space Dense ED with Eigen..." << std::endl;
    int n = 4;
    double g = 0.5;

    Diagonalizer diag;
    DiagConfig config;
    config.method = DiagMethod::ExactDense;
    config.num_eigenpairs = 2;
    config.normalize_by_n = true;

    Eigensolution sol = diag.solve(n, g, config);

    assert(sol.pairs.size() == 2);
    // Ground state energy must be strictly <= first excited state energy
    assert(sol.pairs[0].energy <= sol.pairs[1].energy);
    assert(sol.pairs[0].state_vector.size() == (1ULL << n));

    // Verify eigenvector is normalized: ||v||^2 == 1
    double norm_sq = 0.0;
    for (double x : sol.pairs[0].state_vector) {
        norm_sq += x * x;
    }
    assert(std::abs(norm_sq - 1.0) < 1e-9);

    std::cout << "  N=" << n << ", g=" << g 
              << " -> Ground Energy: " << std::fixed << std::setprecision(6) << sol.ground_energy()
              << " | First Excited: " << sol.pairs[1].energy << "\n";
    std::cout << "  --> PASSED: Full space dense ED verified!\n" << std::endl;
}

// Test 2: Subspace dense ED with explicit basis
void test_subspace_dense_ed() {
    std::cout << "[Test 2] Testing Subspace Dense ED with explicit basis..." << std::endl;
    int n = 4;
    double g = 0.5;

    // Restricted basis of 4 states: |0000>, |0001>, |0011>, |0111>
    std::vector<SpinState> basis = {
        SpinState{0ULL},
        SpinState{1ULL},
        SpinState{3ULL},
        SpinState{7ULL}
    };

    Diagonalizer diag;
    DiagConfig config;
    config.method = DiagMethod::ExactDense;
    config.num_eigenpairs = 2;
    config.normalize_by_n = true;

    Eigensolution sol = diag.solve(basis, g, n, config);

    assert(sol.pairs.size() == 2);
    assert(sol.pairs[0].energy <= sol.pairs[1].energy);
    assert(sol.pairs[0].state_vector.size() == basis.size());

    // Verify normalization
    double norm_sq = 0.0;
    for (double x : sol.pairs[0].state_vector) {
        norm_sq += x * x;
    }
    assert(std::abs(norm_sq - 1.0) < 1e-9);

    // Verify orthogonality between ground and excited: <v0 | v1> == 0
    double dot = 0.0;
    for (size_t i = 0; i < basis.size(); ++i) {
        dot += sol.pairs[0].state_vector[i] * sol.pairs[1].state_vector[i];
    }
    assert(std::abs(dot) < 1e-9);

    std::cout << "  Subspace dim=" << basis.size() 
              << " -> Ground Energy: " << std::fixed << std::setprecision(6) << sol.ground_energy()
              << " | Excited Energy: " << sol.pairs[1].energy << "\n";
    std::cout << "  --> PASSED: Subspace dense ED and orthogonality verified!\n" << std::endl;
}

// Test 3: Mathematical consistency check: Subspace of entire basis equals Full Space
void test_subspace_full_equivalence() {
    std::cout << "[Test 3] Testing equivalence when subspace equals entire 2^N space..." << std::endl;
    int n = 3;
    double g = 0.75;

    Diagonalizer diag;
    DiagConfig config;
    config.method = DiagMethod::ExactDense;
    config.normalize_by_n = true;

    // 1. Solve via Full Space endpoint
    Eigensolution full_sol = diag.solve(n, g, config);

    // 2. Solve via Subspace endpoint with all 2^N states explicitly
    std::vector<SpinState> all_states(1ULL << n);
    for (uint64_t i = 0; i < (1ULL << n); ++i) {
        all_states[i] = SpinState{i};
    }
    Eigensolution sub_sol = diag.solve(all_states, g, n, config);

    // Energies must match exactly
    assert(std::abs(full_sol.ground_energy() - sub_sol.ground_energy()) < 1e-10);

    std::cout << "  Full Space Energy: " << std::fixed << std::setprecision(8) << full_sol.ground_energy() << "\n";
    std::cout << "  Subspace Energy  : " << sub_sol.ground_energy() << "\n";
    std::cout << "  --> PASSED: Exact match between Full Space and full Subspace!\n" << std::endl;
}

int main() {
    std::cout << "=====================================================\n";
    std::cout << "           Testing Exact Diagonalization (ED)        \n";
    std::cout << "=====================================================\n";

    test_fullspace_dense_ed();
    test_subspace_dense_ed();
    test_subspace_full_equivalence();

    std::cout << "=====================================================\n";
    std::cout << "     ALL EXACT DIAGONALIZATION TESTS PASSED!         \n";
    std::cout << "=====================================================" << std::endl;
    return 0;
}