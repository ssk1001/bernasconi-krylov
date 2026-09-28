#include <cmath>
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
     double g_transverse = 2.0;
    int num_bits = 4; 

    // If an argument is provided, parse it
    if (argc > 1) {
        num_bits = std::atoi(argv[1]);
        if (num_bits <= 0 || num_bits > 30) { // Add a safety guard for GPU memory limits
            std::cerr << "Error: Number of bits must be between 1 and 30.\n";
            return 1;
        }
    }
 
    std::cout << "Launching matrix-free GPU Eigensolver on " << num_bits 
              << " bits using GPU 0..." << std::endl;

    // 1. Test Full Space Endpoint
    auto result_full = SpinEigensolver<SpinState, double>::compute_ground_state(num_bits, g_transverse);
    std::cout << "[Full Space] Ground State Energy: " << result_full.energy << std::endl;

    // ------------------------------------------------------------
    // Compute ||H v - E v||_2
    // ------------------------------------------------------------

    const int N = 1 << num_bits;

    double residual_norm_squared = 0.0;
    SpinCache sc;
    for (int i = 0; i < N; ++i) {

        // Get the diagonal contribution.
        //
        // This must use the same diagonal calculation/order as
        // the full-space solver.
        SpinState state{
            static_cast<uint32_t>(num_bits),
            {static_cast<uint64_t>(i)}
        };

        double diag =
            static_cast<double>(SpinLib::get_energy(sc, state))
            / static_cast<double>(num_bits);

        // (H v)_i
        double Hv_i = diag * result_full.state_vector[i];

        // Transverse-field contribution:
        //
        // H_ij = -g whenever j = i XOR (1 << bit)
        for (int j = 0; j < static_cast<int>(num_bits); ++j) {
            int nxt = i ^ (1 << j);

            Hv_i -= g_transverse *
                    result_full.state_vector[nxt];
        }

        // r_i = (H v)_i - E v_i
        double residual_i =
            Hv_i - result_full.energy *
                    result_full.state_vector[i];

        residual_norm_squared += residual_i * residual_i;
    }

    double residual_norm = std::sqrt(residual_norm_squared);

    std::cout << "[Full Space] Residual Norm: "
              << residual_norm << std::endl;

    // 2. Test Subspace Endpoint (Creating a dummy basis of even integer states)
    std::vector<SpinState> basis;
    for (uint64_t i = 0; i < (1ULL << num_bits); i += 2) {
        basis.push_back({static_cast<uint32_t>(num_bits), {i}}); 
    }

    auto result_subspace = SpinEigensolver<SpinState, double>::compute_ground_state(basis, g_transverse);
    std::cout << "[Subspace]   Ground State Energy: " << result_subspace.energy << std::endl;

    return 0;
}
