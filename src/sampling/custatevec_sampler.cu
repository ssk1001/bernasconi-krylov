#include "sampler.cuh"
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <custatevec.h>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <stdexcept>

#define CHECK_CUSTATEVEC(call)                                                                                                           \
    {                                                                                                                                    \
        custatevecStatus_t status = call;                                                                                                \
        if (status != CUSTATEVEC_STATUS_SUCCESS)                                                                                         \
        {                                                                                                                                \
            throw std::runtime_error("cuStateVec error " + std::to_string(status) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
        }                                                                                                                                \
    }

// Precompute energies once to save O(N^3) work inside the evolution loop
__global__ void precompute_labs_energies_kernel(
    double *d_energies,
    uint64_t num_states,
    size_t n)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states)
        return;

    double H_B = (n * (n - 1)) / 2.0;
    for (size_t k = 1; k < n; ++k)
    {
        double sum_pairs = 0.0;
        for (size_t i = 0; i < n - k; ++i)
        {
            for (size_t j = i + 1; j < n - k; ++j)
            {
                int bit_i = (idx >> i) & 1ULL;
                int bit_ik = (idx >> (i + k)) & 1ULL;
                int bit_j = (idx >> j) & 1ULL;
                int bit_jk = (idx >> (j + k)) & 1ULL;
                int xor_sum = bit_i ^ bit_ik ^ bit_j ^ bit_jk;
                sum_pairs += (xor_sum == 0) ? 1.0 : -1.0;
            }
        }
        H_B += 2.0 * sum_pairs;
    }
    d_energies[idx] = H_B;
}

// O(1) Phase Kernel using precomputed energies
__global__ void apply_precomputed_phase_kernel(
    cuDoubleComplex *d_state,
    const double *d_energies,
    uint64_t num_states,
    size_t n,
    double dt)
{
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states)
        return;

    double theta = -(dt / (double)n) * d_energies[idx];

    cuDoubleComplex phase = make_cuDoubleComplex(cos(theta), sin(theta));
    cuDoubleComplex amp = d_state[idx];

    d_state[idx] = make_cuDoubleComplex(
        amp.x * phase.x - amp.y * phase.y,
        amp.x * phase.y + amp.y * phase.x);
}

// Warp reduction helper
__inline__ __device__ double sampler_warp_reduce_sum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Sum |c_s|^2 strictly where E(s) < seed_energy
__global__ void sum_improved_prob_kernel(
    const cuDoubleComplex* __restrict__ d_state,
    const double* __restrict__ d_energies,
    double seed_energy,
    uint64_t num_states,
    double* __restrict__ d_out_prob)
{
    __shared__ double warp_sums[32];
    double thread_sum = 0.0;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;

    for (uint64_t i = idx; i < num_states; i += stride) {
        if (d_energies[i] < seed_energy) {
            cuDoubleComplex c = d_state[i];
            thread_sum += (c.x * c.x + c.y * c.y);
        }
    }

    double warp_sum = sampler_warp_reduce_sum(thread_sum);
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = warp_sum;
    __syncthreads();

    if (warp_id == 0) {
        double block_sum = (lane < (blockDim.x / warpSize)) ? warp_sums[lane] : 0.0;
        block_sum = sampler_warp_reduce_sum(block_sum);
        if (lane == 0) atomicAdd(d_out_prob, block_sum);
    }
}

CuStateVecSampler::~CuStateVecSampler()
{
    if (d_cached_energies)
    {
        cudaFree(d_cached_energies);
        d_cached_energies = nullptr;
    }
    if (d_out_prob)
    {
        cudaFree(d_out_prob);
        d_out_prob = nullptr;
    }
}

std::vector<SpinState> CuStateVecSampler::generate_subspace(
    const std::vector<SpinState> &seed_states,
    const SamplerConfig &config,
    SpinCache &cache)
{
    uint64_t num_states = 1ULL << config.n;
    size_t state_size_bytes = num_states * sizeof(cuDoubleComplex);
    size_t energy_size_bytes = num_states * sizeof(double);

    int threads = 256;
    int blocks = (num_states + threads - 1) / threads;

    cuDoubleComplex *d_state = nullptr;
    cudaMalloc(&d_state, state_size_bytes);

    // Precompute diagonal LABS energies ONCE per system size N
    if (cached_n != config.n || d_cached_energies == nullptr)
    {
        if (d_cached_energies)
        {
            cudaFree(d_cached_energies);
            d_cached_energies = nullptr;
        }
        cudaMalloc(&d_cached_energies, energy_size_bytes);
        precompute_labs_energies_kernel<<<blocks, threads>>>(d_cached_energies, num_states, config.n);
        cudaDeviceSynchronize();
        cached_n = config.n;
    }
    const double *d_energies = d_cached_energies;

    custatevecHandle_t handle;
    CHECK_CUSTATEVEC(custatevecCreate(&handle));

    std::unordered_map<SpinState, size_t, SpinStateHash> global_unique_samples;
    std::mt19937 rng(config.seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    uint64_t total_flips = 0;
    uint64_t total_shots = 0;

    if (config.compute_p_less && d_out_prob == nullptr)
    {
        cudaMalloc(&d_out_prob, sizeof(double));
    }
    last_p_less_history.clear();
    last_window_mean_p_less = 0.0;
    double sum_p_less = 0.0;

    // =========================================================================
    // 1. HOISTED ALLOCATIONS & SETUP (Do this ONCE before the loops)
    // =========================================================================

    // Pre-define the Rx matrix since it relies on constant dt and g
    double rx_angle = -2.0 * config.g * config.dt;
    cuDoubleComplex rx_matrix[4] = {
        make_cuDoubleComplex(cos(rx_angle / 2.0), 0.0),
        make_cuDoubleComplex(0.0, -sin(rx_angle / 2.0)),
        make_cuDoubleComplex(0.0, -sin(rx_angle / 2.0)),
        make_cuDoubleComplex(cos(rx_angle / 2.0), 0.0)};

    // Create the cuStateVec Sampler Descriptor ONCE
    custatevecSamplerDescriptor_t sampler;
    size_t sampler_workspace_size = 0;
    CHECK_CUSTATEVEC(custatevecSamplerCreate(
        handle, d_state, CUDA_C_64F, static_cast<uint32_t>(config.n),
        &sampler, static_cast<uint32_t>(config.m_samples), &sampler_workspace_size));

    // Allocate device workspace ONCE
    void *d_sampler_workspace = nullptr;
    if (sampler_workspace_size > 0)
    {
        cudaMalloc(&d_sampler_workspace, sampler_workspace_size);
    }

    // Pre-allocate CPU arrays for sampling variables
    std::vector<int32_t> bit_ordering(config.n);
    for (int32_t i = 0; i < static_cast<int32_t>(config.n); ++i)
        bit_ordering[i] = i;

    std::vector<double> rand_nums(config.m_samples);
    std::vector<custatevecIndex_t> sampled_indices(config.m_samples);

    // =========================================================================
    // 2. TIME EVOLUTION & SAMPLING LOOPS
    // =========================================================================

    for (const auto &seed_state : seed_states)
    {
        // Reset state vector to |0...0>
        cudaMemset(d_state, 0, state_size_bytes);

        // Map the spin array to the exact computational basis state
        uint64_t seed_idx = 0;
        for (size_t i = 0; i < config.n; ++i)
        {
            if (seed_state.get_spin(i) == -1)
            {
                seed_idx |= (1ULL << i);
            }
        }

        // Inject the amplitude 1.0 at the targeted seed index
        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        cudaMemcpy(d_state + seed_idx, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

        // Burn-in phase (evolve unitarily without measurement until t_start)
        if (config.t_start > 0.0)
        {
            size_t burnin_steps = static_cast<size_t>(std::round(config.t_start / config.dt));
            for (size_t b_step = 0; b_step < burnin_steps; ++b_step)
            {
                apply_precomputed_phase_kernel<<<blocks, threads>>>(
                    d_state, d_energies, num_states, config.n, config.dt);
                for (int32_t target_qubit = 0; target_qubit < static_cast<int32_t>(config.n); ++target_qubit)
                {
                    CHECK_CUSTATEVEC(custatevecApplyMatrix(
                        handle, d_state, CUDA_C_64F, static_cast<uint32_t>(config.n),
                        rx_matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                        0, &target_qubit, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
                }
            }
        }

        double step_interval = (config.dt_sample > 0.0) ? config.dt_sample : config.dt;
        size_t interval_steps = std::max((size_t)1, static_cast<size_t>(std::round(step_interval / config.dt)));

        for (size_t step = 0; step < config.k_steps; ++step)
        {
            if (config.t_start > 0.0)
            {
                if (step > 0)
                {
                    // Evolve for interval_steps between sample points
                    for (size_t s = 0; s < interval_steps; ++s)
                    {
                        apply_precomputed_phase_kernel<<<blocks, threads>>>(
                            d_state, d_energies, num_states, config.n, config.dt);
                        for (int32_t target_qubit = 0; target_qubit < static_cast<int32_t>(config.n); ++target_qubit)
                        {
                            CHECK_CUSTATEVEC(custatevecApplyMatrix(
                                handle, d_state, CUDA_C_64F, static_cast<uint32_t>(config.n),
                                rx_matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                                0, &target_qubit, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
                        }
                    }
                }
            }
            else
            {
                // Standard legacy evolution: evolve 1 step of dt
                apply_precomputed_phase_kernel<<<blocks, threads>>>(
                    d_state, d_energies, num_states, config.n, config.dt);
                for (int32_t target_qubit = 0; target_qubit < static_cast<int32_t>(config.n); ++target_qubit)
                {
                    CHECK_CUSTATEVEC(custatevecApplyMatrix(
                        handle, d_state, CUDA_C_64F, static_cast<uint32_t>(config.n),
                        rx_matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                        0, &target_qubit, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
                }
            }

            // Exact statevector P_< measurement at this sample point
            if (config.compute_p_less && config.seed_energy > 0.0 && d_out_prob != nullptr)
            {
                cudaMemset(d_out_prob, 0, sizeof(double));
                int red_blocks = std::min(1024, blocks);
                sum_improved_prob_kernel<<<red_blocks, threads>>>(
                    d_state, d_energies, config.seed_energy, num_states, d_out_prob);
                double p_val = 0.0;
                cudaMemcpy(&p_val, d_out_prob, sizeof(double), cudaMemcpyDeviceToHost);
                last_p_less_history.push_back(p_val);
                sum_p_less += p_val;
            }

            // C. Sampling
            // Only run Preprocess inside the loop to recalculate probabilities from the newly evolved state
            CHECK_CUSTATEVEC(custatevecSamplerPreprocess(
                handle, sampler, d_sampler_workspace, sampler_workspace_size));

            // Generate fresh random [0,1) numbers for this specific time step
            for (size_t i = 0; i < config.m_samples; ++i)
            {
                rand_nums[i] = dist(rng);
            }

            CHECK_CUSTATEVEC(custatevecSamplerSample(
                handle, sampler, sampled_indices.data(), bit_ordering.data(),
                static_cast<uint32_t>(config.n), rand_nums.data(),
                static_cast<uint32_t>(config.m_samples), CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER));

            // D. Reconstruct bitstrings and collect sample multiplicities
            for (auto idx : sampled_indices)
            {
                total_flips += __builtin_popcountll(idx ^ seed_idx);
                total_shots++;

                SpinState sampled_state;
                sampled_state.num_bits = config.n;
                sampled_state.bits.resize((config.n + 63) / 64, 0);
                for (size_t b = 0; b < config.n; ++b)
                {
                    if ((idx >> b) & 1ULL)
                    {
                        sampled_state.bits[b / 64] |= (1ULL << (b % 64));
                    }
                }
                // Retains the exact sample multiplicities (and inherently forms the unique set)
                global_unique_samples[sampled_state]++;
            }
        }
    }

    // =========================================================================
    // 3. CLEANUP (Do this ONCE after all loops finish)
    // =========================================================================
    CHECK_CUSTATEVEC(custatevecSamplerDestroy(sampler));
    if (d_sampler_workspace)
    {
        cudaFree(d_sampler_workspace);
    }

    // Form the unique set S
    std::vector<SpinState> subspace_basis;
    subspace_basis.reserve(global_unique_samples.size());

    if (config.top_k_augmentation > 0)
    {
        std::vector<std::pair<SpinState, int>> sorted_samples(
            global_unique_samples.begin(), global_unique_samples.end());

        std::sort(sorted_samples.begin(), sorted_samples.end(),
                  [](const auto &a, const auto &b)
                  {
                      return a.second > b.second; // Sort descending by sample count
                  });

        for (const auto &pair : sorted_samples)
        {
            subspace_basis.push_back(pair.first);
        }

        int top_k = std::min((int)sorted_samples.size(), (int)config.top_k_augmentation);
        for (int i = 0; i < top_k; ++i)
        {
            SpinState dominant_state = sorted_samples[i].first;
            for (size_t b = 0; b < config.n; ++b)
            {
                SpinState neighbor = dominant_state;
                size_t word_idx = b / 64;
                size_t bit_idx = b % 64;
                neighbor.bits[word_idx] ^= (1ULL << bit_idx);

                if (global_unique_samples.find(neighbor) == global_unique_samples.end())
                {
                    subspace_basis.push_back(neighbor);
                    global_unique_samples[neighbor] = 0;
                }
            }
        }
    }
    else
    {
        // Pure quantum measurement sampling without classical neighbor augmentation
        for (const auto &pair : global_unique_samples)
        {
            subspace_basis.push_back(pair.first);
        }
    }
    custatevecDestroy(handle);
    cudaFree(d_state);

    last_mean_flips = (total_shots > 0) ? static_cast<double>(total_flips) / static_cast<double>(total_shots) : 0.0;
    if (!last_p_less_history.empty())
    {
        last_window_mean_p_less = sum_p_less / static_cast<double>(last_p_less_history.size());
    }

    std::cout << "[Sampler] Sampled " << total_shots << " quantum shots | Mean bit flips from seed: "
              << std::fixed << std::setprecision(2) << last_mean_flips << std::defaultfloat;
    if (config.compute_p_less && !last_p_less_history.empty())
    {
        std::cout << " | Window Mean P_<: " << std::scientific << std::setprecision(4) 
                  << last_window_mean_p_less << std::defaultfloat;
    }
    std::cout << std::endl;

    return subspace_basis;
}