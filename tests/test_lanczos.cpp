#include <slepceps.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../src/subspace/lanczos.hpp" 

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // int num_devices = 0;
    // cudaGetDeviceCount(&num_devices);
    // if (num_devices > 0) {
    //     cudaSetDevice(rank % num_devices);
    // } else {
    //     if (rank == 0) std::cerr << "Error: No CUDA-capable devices found." << std::endl;
    //     MPI_Abort(MPI_COMM_WORLD, 1);
    // }

    SlepcInitialize(&argc, &argv, NULL, NULL);

    size_t num_bits = 4;
    double g_transverse = 2.0;
    
    // if (rank == 0) {
    //     std::cout << "Launching matrix-free GPU Eigensolver on " << num_bits 
    //               << " bits across " << num_devices << " local GPUs..." << std::endl;
    // }

    auto result = SpinEigensolver<SpinState, double>::compute_ground_state(num_bits, g_transverse);

    if (rank == 0) {
        std::cout << "Ground State Energy: " << result.energy << std::endl;
    }

    SlepcFinalize();
    MPI_Finalize(); // <--- ADDED THIS TO FIX THE MPI ABORT ERROR
    return 0;
}