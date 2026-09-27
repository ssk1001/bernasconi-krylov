#include <iostream>
#include <vector>
#include <iomanip>
#include "../src/exact/exact_diag.hpp"
// Assuming "spin_cache.hpp" and the Comparator_16bits class definition are included here

int main() {
    // Parameters for testing
    size_t N = 3;
    double g = 0.5;

    std::cout << "=== Testing Exact Diagonalization ===\n\n";

    // ---------------------------------------------------------
    // Test 1: Full 2^N Space Exact Diagonalization
    // ---------------------------------------------------------
    std::cout << "--- 1. Full Space (N=" << N << ", g=" << g << ") ---\n";
    EigensystemSolution full_sol = ExactDiagonalizer::solve_full_space(N, g);
    
    std::cout << "Ground Energy: " << std::fixed << std::setprecision(6) 
              << full_sol.ground_energy << "\n";
    std::cout << "Ground State Vector:\n" 
              << full_sol.ground_state.transpose() << "\n\n";

    // ---------------------------------------------------------
    // Test 2: Subspace Exact Diagonalization
    // ---------------------------------------------------------
    std::cout << "--- 2. Restricted Subspace ---\n";
    
    // Create a restricted basis of 4 specific states:
    // 000 (0), 001 (1), 011 (3), 111 (7)
    std::vector<SpinState> subspace = {
        {N, {0ULL}}, 
        {N, {1ULL}}, 
        {N, {3ULL}}, 
        {N, {7ULL}}
    };
    
    EigensystemSolution sub_sol = ExactDiagonalizer::solve_subspace(subspace, g);
    
    std::cout << "Subspace Dimension: " << subspace.size() << "\n";
    std::cout << "Ground Energy: " << std::fixed << std::setprecision(6) 
              << sub_sol.ground_energy << "\n";
    std::cout << "Ground State Vector:\n" 
              << sub_sol.ground_state.transpose() << "\n";

    return 0;
}