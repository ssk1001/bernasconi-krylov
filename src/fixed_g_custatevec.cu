#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "sampling/sampler.cuh"
#include "subspace/lanczos.cuh" 

int main(int argc, char **argv) {
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0) {
        std::cerr << "Error: No CUDA-capable devices found.\n";
        return 1;
    }
    cudaSetDevice(0);

    // Parse N from command line if provided
    size_t n_spins = 16; 
    if (argc > 1) n_spins = std::atoi(argv[1]);

    // Fixed-g algorithm parameters for (N, g, \Delta t, K, M)[cite: 3]
    SamplerConfig config;
    config.n = n_spins;
    config.g = 2.0;                  
    config.dt = 0.05;                
    config.k_steps = 20;             
    config.m_samples = 1000;         
    config.top_k_augmentation = 5;   

    std::cout << "====================================================\n";
    std::cout << "          FIXED-g ALGORITHM EXECUTION               \n";
    std::cout << "====================================================\n";
    std::cout << "Parameters:\n"
              << "  N (Spins)         : " << config.n << "\n"
              << "  g (Transverse)    : " << config.g << "\n"
              << "  dt (Time step)    : " << config.dt << "\n"
              << "  K (States)        : " << config.k_steps << "\n"
              << "  M (Samples/state) : " << config.m_samples << "\n"
              << "  Augment Threshold : " << config.top_k_augmentation << "\n";
    std::cout << "----------------------------------------------------\n";

    // 1. Prepare a seed |s_0>[cite: 3]
    std::cout << "[Step 1] Preparing Seed State...\n";
    SpinState seed_state;
    seed_state.num_bits = config.n;
    seed_state.bits.resize((config.n + 63) / 64, 0);
    std::vector<SpinState> seeds = { seed_state };

    // 2-5. Time Evolution, Sampling, Set Union, and Augmentation[cite: 3]
    std::cout << "[Step 2-5] Running cuStateVec Time-Evolution & Sampling...\n";
    CuStateVecSampler sampler;
    SpinCache cache; 
    
    auto start_sample = std::chrono::high_resolution_clock::now();
    std::vector<SpinState> subspace_basis = sampler.generate_subspace(seeds, config, cache);
    auto end_sample = std::chrono::high_resolution_clock::now();
    
    double time_sample = std::chrono::duration<double>(end_sample - start_sample).count();
    double subspace_ratio = (double)subspace_basis.size() / (1ULL << config.n);
    
    std::cout << "  -> Generated Subspace Size |S| = " << subspace_basis.size() << "\n";
    std::cout << "  -> Subspace Ratio = " << std::fixed << std::setprecision(4) << (subspace_ratio * 100.0) << "%\n";
    std::cout << "  -> Sampling Time  = " << std::fixed << std::setprecision(4) << time_sample << " s\n";
    std::cout << "----------------------------------------------------\n";

    // 6-7. Construct sparse matrix H_S(g) and compute lowest eigenvalues[cite: 3]
    std::cout << "[Step 6-7] Running Subspace Eigensolver (Lanczos)...\n";
    auto start_sub = std::chrono::high_resolution_clock::now();
    auto result_subspace = SpinEigensolver<SpinState, double>::compute_ground_state(subspace_basis, config.g);
    auto end_sub = std::chrono::high_resolution_clock::now();
    
    double time_sub = std::chrono::duration<double>(end_sub - start_sub).count();

    std::cout << "  -> Subspace Ground State Energy = " << std::setprecision(10) << result_subspace.energy << "\n";
    std::cout << "  -> Subspace Eigensolver Time    = " << std::setprecision(4) << time_sub << " s\n";

    // Extract dominant bitstrings of the approximate ground state[cite: 3]
    std::cout << "\n  -> Dominant Bitstrings:\n";
    std::vector<std::pair<double, size_t>> amplitudes;
    for (size_t i = 0; i < subspace_basis.size(); ++i) {
        amplitudes.push_back({std::abs(result_subspace.state_vector[i]), i});
    }
    // Sort descending by magnitude
    std::sort(amplitudes.rbegin(), amplitudes.rend());
    
    int top_k = std::min(5, (int)amplitudes.size());
    for (int i = 0; i < top_k; ++i) {
        size_t idx = amplitudes[i].second;
        double amp = result_subspace.state_vector[idx];
        uint64_t bits = subspace_basis[idx].bits[0]; // Assuming fits in uint64_t for display
        std::cout << "       [" << i+1 << "] |" << std::bitset<64>(bits).to_string().substr(64 - config.n) 
                  << "> : " << std::setprecision(6) << amp 
                  << " (Prob: " << std::setprecision(4) << (amp * amp) * 100.0 << "%)\n";
    }
    std::cout << "----------------------------------------------------\n";

    // 8. Compare against exact diagonalization where available[cite: 3]
    // Placeholder: We only trigger the full-space reference Lanczos for small N
    if (config.n <= 16) {
        std::cout << "[Step 8] Comparing against Full-Space Reference (Small N)...\n";
        // TODO: Replace SpinEigensolver full-space with True Exact Diagonalization (ED) placeholder
        auto start_full = std::chrono::high_resolution_clock::now();
        auto result_full = SpinEigensolver<SpinState, double>::compute_ground_state(config.n, config.g);
        auto end_full = std::chrono::high_resolution_clock::now();
        
        double time_full = std::chrono::duration<double>(end_full - start_full).count();
        double error = std::abs(result_full.energy - result_subspace.energy);

        std::cout << "  -> Full Space Ground State Energy = " << std::setprecision(10) << result_full.energy << "\n";
        std::cout << "  -> Absolute Energy Error          = " << std::scientific << error << "\n";
        std::cout << "  -> Full Space Eigensolver Time    = " << std::fixed << std::setprecision(4) << time_full << " s\n";
    } else {
        std::cout << "[Step 8] Skipped. (Exact diagonalization not available for N > 16)\n";
    }
    
    std::cout << "====================================================\n";
    return 0;
}