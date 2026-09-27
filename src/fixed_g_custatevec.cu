#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <slepcsys.h>

#include "sampling/sampler.cuh"
#include "classical/steepest_descent.cuh"
#include "subspace/lanczos.hpp"

// Extracts the top-K highest amplitude states from the sequential SLEPc eigenvector
std::vector<SpinState> extract_dominant_states(
    const std::vector<SpinState>& basis, 
    const std::vector<double>& state_vector, 
    size_t num_seeds) 
{
    std::vector<size_t> indices(basis.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices based on absolute amplitude in descending order
    std::sort(indices.begin(), indices.end(), 
        [&](size_t a, size_t b) {
            return std::abs(state_vector[a]) > std::abs(state_vector[b]);
        });

    std::vector<SpinState> next_seeds;
    size_t limit = std::min(num_seeds, basis.size());
    for (size_t i = 0; i < limit; ++i) {
        next_seeds.push_back(basis[indices[i]]);
    }
    
    return next_seeds;
}

// Forward declarations for matrix-free shell operations
extern PetscErrorCode matmul_bridge(Mat A, Vec x, Vec y);
void broadcast_subspace(std::vector<SpinState>& subspace, int rank); 

int main(int argc, char **argv) {
    SlepcInitialize(&argc, &argv, nullptr, nullptr);

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    SamplerConfig config = {
        .n = 10,
        .g = 0.5,
        .dt = 0.01,
        .k_steps = 10,
        .m_samples = 100,
        .top_k_augmentation = 2
    };

    SpinCache cache;
    int num_iterations = 5;
    size_t top_k_seeds_to_keep = 10;
    size_t steepest_descent_starts = 50;

    std::vector<SpinState> current_seeds;

    // 1. Initial seeds from Steepest Descent (Root Rank)
    if (rank == 0) {
        DescentResult desc = multi_start_steepest_descent(config.n, steepest_descent_starts, cache);
        current_seeds = desc.all_states; // Use the full pool of candidates
        std::cout << "Initial Steepest Descent Best Energy: " << desc.best_energy << "\n";
    }

    auto sampler = create_sampler(config.n);

    // The Feedback Loop
    for (int iter = 0; iter < num_iterations; ++iter) {
        std::vector<SpinState> subspace;

        // 2. Generate Subspace via cuStateVec (Root Rank)
        if (rank == 0) {
            subspace = sampler->generate_subspace(current_seeds, config, cache);
            std::cout << "Iter " << iter << ": Subspace size = " << subspace.size() << "\n";
        }

        // 3. Sync Subspace across all MPI ranks
        // Required because MatCreateShell and matmul_bridge need the basis on all nodes
        broadcast_subspace(subspace, rank); 

        // 4. Setup Matrix-Free Shell for this iteration's subspace size
        PetscInt local_N = PETSC_DECIDE; // Or explicit partitioning if required
        PetscInt global_N = subspace.size();
        Mat A;
        MatCreateShell(PETSC_COMM_WORLD, local_N, local_N, global_N, global_N, (void*)&subspace, &A);

        // 5. Run Lanczos/Krylov-Schur
        SpinEigensolver<>::Eigenpair ground_state = SpinEigensolver<>::compute_ground_state(subspace, config.g);

        // 6. Extract dominant bitstrings for the next iteration's seeds
        if (rank == 0) {
            current_seeds = extract_dominant_states(subspace, ground_state.state_vector, top_k_seeds_to_keep);
            std::cout << "Iter " << iter << " Approx Ground Energy: " << ground_state.energy << "\n";
        }
    }

    SlepcFinalize();
    return 0;
}



void broadcast_subspace(std::vector<SpinState>& subspace, int rank) {
    size_t num_states = subspace.size();
    MPI_Bcast(&num_states, 1, MPI_UNSIGNED_LONG_LONG, 0, PETSC_COMM_WORLD);

    if (num_states == 0) return;

    size_t num_bits = (rank == 0) ? subspace[0].num_bits : 0;
    MPI_Bcast(&num_bits, 1, MPI_UNSIGNED_LONG_LONG, 0, PETSC_COMM_WORLD);

    size_t words_per_state = (num_bits + 63) / 64;
    std::vector<uint64_t> flat_data(num_states * words_per_state);

    if (rank == 0) {
        for (size_t i = 0; i < num_states; ++i) {
            std::copy(subspace[i].bits.begin(), subspace[i].bits.end(), 
                      flat_data.begin() + i * words_per_state);
        }
    }

    MPI_Bcast(flat_data.data(), flat_data.size(), MPI_UNSIGNED_LONG_LONG, 0, PETSC_COMM_WORLD);

    if (rank != 0) {
        subspace.resize(num_states);
        for (size_t i = 0; i < num_states; ++i) {
            subspace[i].num_bits = num_bits;
            subspace[i].bits.assign(flat_data.begin() + i * words_per_state, 
                                    flat_data.begin() + (i + 1) * words_per_state);
        }
    }
}