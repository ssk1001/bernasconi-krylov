#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <unordered_map>
#include <random>
#include <vector>
#include <cmath>
#include "sampler.cuh"

// CUDA kernel to apply the diagonal LABS energy phase directly to state vector amplitudes
__global__ void apply_labs_phase_kernel(
    cuDoubleComplex* d_state, 
    uint64_t num_states, 
    size_t n, 
    double dt) 
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    // Compute LABS energy: E_B = N(N-1)/2 + 2 * sum_k (C_k)^2
    double H_B = (n * (n - 1)) / 2.0;

    for (size_t k = 1; k < n; ++k) {
        double sum_pairs = 0.0;
        for (size_t i = 0; i < n - k; ++i) {
            for (size_t j = i + 1; j < n - k; ++j) {
                // Extract bits (0-indexed). Z = 1 if bit is 0, -1 if bit is 1.
                int bit_i   = (idx >> i) & 1ULL;
                int bit_ik  = (idx >> (i + k)) & 1ULL;
                int bit_j   = (idx >> j) & 1ULL;
                int bit_jk  = (idx >> (j + k)) & 1ULL;

                // Z_i * Z_{i+k} * Z_j * Z_{j+k} equivalent to XOR parity
                int xor_sum = bit_i ^ bit_ik ^ bit_j ^ bit_jk;
                sum_pairs += (xor_sum == 0) ? 1.0 : -1.0;
            }
        }
        H_B += 2.0 * sum_pairs;
    }

    // Calculate phase shift: theta = - (dt / N) * H_B
    double theta = - (dt / (double)n) * H_B;
    
    cuDoubleComplex phase = make_cuDoubleComplex(cos(theta), sin(theta));
    cuDoubleComplex amp = d_state[idx];
    
    // Apply scalar multiplication to the amplitude
    d_state[idx] = make_cuDoubleComplex(
        amp.x * phase.x - amp.y * phase.y, 
        amp.x * phase.y + amp.y * phase.x
    );
}

std::vector<SpinState> CuStateVecSampler::generate_subspace(
    const std::vector<SpinState>& seed_states, 
    const SamplerConfig& config, 
    SpinCache& cache) 
{
    uint64_t num_states = 1ULL << config.n;
    size_t state_size_bytes = num_states * sizeof(cuDoubleComplex);

    // Allocate memory and handles ONCE
    cuDoubleComplex* d_state = nullptr;
    cudaMalloc(&d_state, state_size_bytes);

    custatevecHandle_t handle;
    custatevecCreate(&handle);

    // Global map to aggregate samples across all seeds
    std::unordered_map<SpinState, size_t, SpinStateHash> global_unique_samples;
    
    std::random_device rd;
    std::mt19937 rng(rd()); 
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Process each steepest descent seed independently
    for (const auto& seed_state : seed_states) {
        
        // Reset state to all zeros
        cudaMemset(d_state, 0, state_size_bytes);

        // Decode seed and set amplitude |s0> = 1.0
        uint64_t seed_idx = 0;
        for (size_t i = 0; i < config.n; ++i) {
            if (seed_state.get_spin(i) == -1) { 
                seed_idx |= (1ULL << i);
            }
        }
        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        cudaMemcpy(d_state + seed_idx, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        // Fixed-g Real-Time Evolution Loop
        for (size_t step = 0; step < config.k_steps; ++step) {
            
            // A. Diagonal Phase Evolution
            int threads_per_block = 256;
            int blocks = (num_states + threads_per_block - 1) / threads_per_block;
            apply_labs_phase_kernel<<<blocks, threads_per_block>>>(d_state, num_states, config.n, config.dt);
            cudaDeviceSynchronize();

            // B. Transverse Field Evolution
            double rx_angle = -2.0 * config.g * config.dt;
            cuDoubleComplex rx_matrix[4] = {
                make_cuDoubleComplex(cos(rx_angle / 2.0), 0.0),
                make_cuDoubleComplex(0.0, -sin(rx_angle / 2.0)),
                make_cuDoubleComplex(0.0, -sin(rx_angle / 2.0)),
                make_cuDoubleComplex(cos(rx_angle / 2.0), 0.0)
            };

            for (int32_t target_qubit = 0; target_qubit < static_cast<int32_t>(config.n); ++target_qubit) {
                custatevecApplyMatrix(
                    handle, d_state, CUDA_C_64F, static_cast<uint32_t>(config.n),
                    rx_matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                    0, &target_qubit, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0
                );
            }

            // C. Batched Measurement Sampling
            custatevecSamplerDescriptor_t sampler;
            size_t sampler_workspace_size = 0;
            custatevecSamplerCreate(
                handle, d_state, CUDA_C_64F, static_cast<uint32_t>(config.n), 
                &sampler, static_cast<uint32_t>(config.m_samples), &sampler_workspace_size
            );

            std::vector<int32_t> bit_ordering(config.n);
            for (int32_t i = 0; i < static_cast<int32_t>(config.n); ++i) bit_ordering[i] = i;
            
            std::vector<double> rand_nums(config.m_samples);
            for (size_t i = 0; i < config.m_samples; ++i) rand_nums[i] = dist(rng);

            std::vector<custatevecIndex_t> sampled_indices(config.m_samples);
            custatevecSamplerSample(
                handle, sampler, sampled_indices.data(), bit_ordering.data(), 
                static_cast<uint32_t>(config.n), rand_nums.data(), 
                static_cast<uint32_t>(config.m_samples), CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER
            );
            custatevecSamplerDestroy(sampler);

            // D. Aggregate into the global map
            for (auto idx : sampled_indices) {
                SpinState sampled_state;
                sampled_state.num_bits = config.n;
                sampled_state.bits.resize((config.n + 63) / 64, 0);
                for (size_t b = 0; b < config.n; ++b) {
                    if ((idx >> b) & 1ULL) sampled_state.bits[b / 64] |= (1ULL << (b % 64));
                }
                global_unique_samples[sampled_state]++;
            }
        }
    }

    // 4. One-Spin-Flip Augmentation for High-Weight Strings (across ALL seeds)
    std::vector<SpinState> final_subspace;
    for (const auto& [state, count] : global_unique_samples) {
        final_subspace.push_back(state);
        if (count >= config.top_k_augmentation) {
            for (size_t i = 0; i < config.n; ++i) {
                SpinState neighbor = state.flip_bit(i);
                if (global_unique_samples.find(neighbor) == global_unique_samples.end()) {
                    final_subspace.push_back(neighbor);
                }
            }
        }
    }

    custatevecDestroy(handle);
    cudaFree(d_state);
    return final_subspace;
}