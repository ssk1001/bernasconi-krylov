#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "../src/subspace/lanczos.cuh" 

int main(int argc, char **argv) {
    // Optional: Select the specific GPU you want to run on if multiple exist
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        cudaSetDevice(0); 
    } else {
        std::cerr << "Error: No CUDA-capable devices found." << std::endl;
        return 1;
    }

    size_t num_bits = 4;
    double g_transverse = 2.0;
    
    std::cout << "Launching matrix-free GPU Eigensolver on " << num_bits 
              << " bits using GPU 0..." << std::endl;

    // 1. Test Full Space Endpoint
    auto result_full = SpinEigensolver<SpinState, double>::compute_ground_state(num_bits, g_transverse);
    std::cout << "[Full Space] Ground State Energy: " << result_full.energy << std::endl;

    // 2. Test Subspace Endpoint (Creating a dummy basis of even integer states)
    std::vector<SpinState> basis;
    for (uint64_t i = 0; i < (1ULL << num_bits); i += 2) {
        basis.push_back({static_cast<uint32_t>(num_bits), {i}}); 
    }

    auto result_subspace = SpinEigensolver<SpinState, double>::compute_ground_state(basis, g_transverse);
    std::cout << "[Subspace]   Ground State Energy: " << result_subspace.energy << std::endl;

    return 0;
}