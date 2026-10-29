#pragma once

#include <vector>
#include <unordered_set>
#include <memory>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>

#include "../labs_energy/labs_energy.hpp"

// Optional cuStateVec & CUDA runtime headers guarded by C++17 __has_include
#if defined(__CUDACC__) && __has_include(<custatevec.h>) && __has_include(<cuda_runtime.h>)
    #define HAS_CUSTATEVEC 1
    #include <cuda_runtime.h>
    #include <cuComplex.h>
    #include <custatevec.h>
#else
    #define HAS_CUSTATEVEC 0
#endif

#if HAS_CUSTATEVEC

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
    } \
}
#endif

#ifndef CHECK_CUSTATEVEC
#define CHECK_CUSTATEVEC(call) { \
    custatevecStatus_t status = call; \
    if (status != CUSTATEVEC_STATUS_SUCCESS) { \
        throw std::runtime_error("cuStateVec error " + std::to_string(status) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
}
#endif

// ============================================================================
// CUDA Kernels for cuStateVec Quantum Evolution
// ============================================================================

/// Precomputes diagonal LABS energies on device using hardware __popcll in O(N) instructions per state
static __global__ void precompute_labs_energies_popcount_kernel(
    double* __restrict__ d_energies,
    uint64_t num_states,
    int n)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    double total_energy = 0.0;
    for (int k = 1; k < n; ++k) {
        uint64_t mask_k = (1ULL << (n - k)) - 1ULL;
        int diffs = __popcll((idx ^ (idx >> k)) & mask_k);
        int64_t ck = static_cast<int64_t>(n - k) - 2 * diffs;
        total_energy += static_cast<double>(ck * ck);
    }
    d_energies[idx] = total_energy;
}

/// Applies diagonal unitary phase e^{-i (dt / n) * E(s)} to statevector amplitudes
static __global__ void apply_diagonal_phase_kernel(
    cuDoubleComplex* __restrict__ d_state,
    const double* __restrict__ d_energies,
    uint64_t num_states,
    int n,
    double dt)
{
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;

    double theta = -(dt / static_cast<double>(n)) * d_energies[idx];
    cuDoubleComplex phase = make_cuDoubleComplex(cos(theta), sin(theta));
    cuDoubleComplex amp = d_state[idx];

    d_state[idx] = make_cuDoubleComplex(
        amp.x * phase.x - amp.y * phase.y,
        amp.x * phase.y + amp.y * phase.x
    );
}

/// Warp-reduction helper for parallel summation
__inline__ __device__ double warp_reduce_sum_prob(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/// Parallel reduction kernel to sum |c_s|^2 strictly where E(s) < E_0 in O(N) tree depth
static __global__ void sum_p_less_kernel(
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

    double warp_sum = warp_reduce_sum_prob(thread_sum);
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = warp_sum;
    __syncthreads();

    if (warp_id == 0) {
        double block_sum = (lane < (blockDim.x / warpSize)) ? warp_sums[lane] : 0.0;
        block_sum = warp_reduce_sum_prob(block_sum);
        if (lane == 0) atomicAdd(d_out_prob, block_sum);
    }
}

/// Parallel reduction kernel to count states with E(s) < seed_energy for P_blind
static __global__ void count_improved_states_kernel(
    const double* __restrict__ d_energies,
    double seed_energy,
    uint64_t num_states,
    unsigned long long* __restrict__ d_out_count)
{
    __shared__ unsigned long long warp_sums[32];
    unsigned long long thread_count = 0;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;

    for (uint64_t i = idx; i < num_states; i += stride) {
        if (d_energies[i] < seed_energy) thread_count++;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        thread_count += __shfl_down_sync(0xffffffff, thread_count, offset);

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = thread_count;
    __syncthreads();

    if (warp_id == 0) {
        unsigned long long block_sum = (lane < (blockDim.x / warpSize)) ? warp_sums[lane] : 0ULL;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        if (lane == 0) atomicAdd(d_out_count, block_sum);
    }
}

#endif // HAS_CUSTATEVEC

// ============================================================================
// 1. Data Structures & Configuration
// ============================================================================

struct SamplerConfig {
    // Physical & Trotter parameters
    int n = 16;
    double g = 0.5;                // Transverse field strength
    double dt = 0.05;              // Trotter simulation step size
    double t_max = 60.0;           // Total evolution duration

    // Sampling parameters: K measurement points, M shots per point
    int num_sample_points = 20;    // K: Number of measurement intervals
    int shots_per_point = 100;     // M: Number of bitstring samples per point
    uint64_t seed = 42;            // RNG seed for quantum measurement collapses

    // Delayed sampling / burn-in parameters
    double t_start = 0.0;          // Burn-in time before sampling begins (e.g. 10.0s)
    double dt_sample = 0.0;        // Interval between measurement collapses (<= 0 defaults to dt)

    // Tunneling probability options
    bool track_p_less = true;      // Accumulate P(E < E_0) at each measurement point
    bool verbose = true;           // Log progress / summary

    // Continuous trajectory logging options
    bool record_trajectory = false; // When true: records continuous P_<(t) from t=0 to t_max
    double dt_trajectory = 0.0;     // Trajectory logging interval (<= 0 defaults to dt, i.e. every step)
};

struct SamplingResult {
    // 1. Quantum sampled Krylov basis
    std::vector<SpinState> unique_basis;
    size_t total_shots_collected = 0; // Total trials (samples)
    size_t improved_shots = 0;        // Sampled states with E(s) < E_og (successes)
    double success_rate = 0.0;        // improved_shots / total_shots_collected
    double per_eval_rate = 0.0;       // improved_shots / total_evals
    size_t total_evals = 0;           // Total evaluations taken
    double mean_hamming_flips = 0.0;  // Average Hamming distance of samples from seed

    // 2. Tunneling Probability Dynamics
    std::vector<double> time_points;
    std::vector<double> p_less_history; // Instantaneous P_<(t) at each time point
    int64_t start_energy = 0;           // E_0 of initial trap
    double p_blind = 0.0;               // Fraction of Hilbert space with E(s) < E_0
    double peak_p_less = 0.0;           // Max P_<(t) observed
    double peak_time = 0.0;             // Time at which peak P_<(t) occurred
};

// ============================================================================
// Abstract Sampler Interface
// ============================================================================

class ISampler {
public:
    virtual ~ISampler() = default;

    virtual SamplingResult sample(
        const SpinState& start_state,
        const SamplerConfig& config) = 0;
};

// ============================================================================
// cuStateVec Sampler Backend (for N <= 30)
// ============================================================================

#if HAS_CUSTATEVEC

class CuStateVecSampler : public ISampler {
private:
    int cached_n = 0;
    double* d_cached_energies = nullptr;

    void ensure_cached_energies(int n, uint64_t num_states) {
        if (cached_n != n || d_cached_energies == nullptr) {
            if (d_cached_energies) {
                cudaFree(d_cached_energies);
                d_cached_energies = nullptr;
            }
            CHECK_CUDA(cudaMalloc(&d_cached_energies, num_states * sizeof(double)));

            int threads = 256;
            int blocks = static_cast<int>((num_states + threads - 1) / threads);
            precompute_labs_energies_popcount_kernel<<<blocks, threads>>>(d_cached_energies, num_states, n);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            cached_n = n;
        }
    }

    double compute_p_less_gpu(
        const cuDoubleComplex* d_state,
        uint64_t num_states,
        double seed_energy,
        double* d_out_prob)
    {
        CHECK_CUDA(cudaMemset(d_out_prob, 0, sizeof(double)));
        int threads = 256;
        int blocks = std::min(1024, static_cast<int>((num_states + threads - 1) / threads));

        sum_p_less_kernel<<<blocks, threads>>>(d_state, d_cached_energies, seed_energy, num_states, d_out_prob);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        double h_prob = 0.0;
        CHECK_CUDA(cudaMemcpy(&h_prob, d_out_prob, sizeof(double), cudaMemcpyDeviceToHost));
        return h_prob;
    }

    double compute_p_blind_gpu(uint64_t num_states, double seed_energy) {
        unsigned long long* d_out_count = nullptr;
        CHECK_CUDA(cudaMalloc(&d_out_count, sizeof(unsigned long long)));
        CHECK_CUDA(cudaMemset(d_out_count, 0, sizeof(unsigned long long)));

        int threads = 256;
        int blocks = std::min(1024, static_cast<int>((num_states + threads - 1) / threads));
        count_improved_states_kernel<<<blocks, threads>>>(d_cached_energies, seed_energy, num_states, d_out_count);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        unsigned long long h_count = 0;
        CHECK_CUDA(cudaMemcpy(&h_count, d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        cudaFree(d_out_count);

        return static_cast<double>(h_count) / static_cast<double>(num_states);
    }


public:
    CuStateVecSampler() = default;

    ~CuStateVecSampler() override {
        if (d_cached_energies) {
            cudaFree(d_cached_energies);
            d_cached_energies = nullptr;
        }
    }

    SamplingResult sample(
        const SpinState& start_state,
        const SamplerConfig& config) override
    {
        if (config.n > 30) {
            throw std::runtime_error("cuStateVec backend exceeds memory bounds for N > 30.");
        }

        SamplingResult result;
        LabsEvaluator eval(config.n);
        result.start_energy = eval.calcEnergy(start_state);

        uint64_t num_states = 1ULL << config.n;
        size_t state_bytes = num_states * sizeof(cuDoubleComplex);

        // 1. Allocate device statevector and initialize to |start_state>
        cuDoubleComplex* d_state = nullptr;
        CHECK_CUDA(cudaMalloc(&d_state, state_bytes));
        CHECK_CUDA(cudaMemset(d_state, 0, state_bytes));

        ensure_cached_energies(config.n, num_states);
        result.p_blind = compute_p_blind_gpu(num_states, static_cast<double>(result.start_energy));

        uint64_t start_idx = start_state.bits & ((config.n >= 64) ? ~0ULL : ((1ULL << config.n) - 1ULL));
        cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
        CHECK_CUDA(cudaMemcpy(d_state + start_idx, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

        double* d_out_prob = nullptr;
        if (config.track_p_less || config.record_trajectory) {
            CHECK_CUDA(cudaMalloc(&d_out_prob, sizeof(double)));
        }

        // 2. Initialize cuStateVec handle and precompute single-qubit Rx gate matrix
        custatevecHandle_t handle;
        CHECK_CUSTATEVEC(custatevecCreate(&handle));

        double rx_angle = -2.0 * config.g * config.dt;
        cuDoubleComplex rx_matrix[4] = {
            make_cuDoubleComplex(cos(rx_angle / 2.0), 0.0),
            make_cuDoubleComplex(0.0, -sin(rx_angle / 2.0)),
            make_cuDoubleComplex(0.0, -sin(rx_angle / 2.0)),
            make_cuDoubleComplex(cos(rx_angle / 2.0), 0.0)
        };

        // 3. Create cuStateVec sampler descriptor and workspace
        custatevecSamplerDescriptor_t sampler_desc;
        size_t workspace_size = 0;
        int shots = std::max(1, config.shots_per_point);

        CHECK_CUSTATEVEC(custatevecSamplerCreate(
            handle, d_state, CUDA_C_64F, static_cast<uint32_t>(config.n),
            &sampler_desc, static_cast<uint32_t>(shots), &workspace_size));

        void* d_workspace = nullptr;
        if (workspace_size > 0) {
            CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));
        }

        std::vector<int32_t> bit_ordering(config.n);
        for (int i = 0; i < config.n; ++i) bit_ordering[i] = i;

        std::vector<double> rand_nums(shots);
        std::vector<custatevecIndex_t> sampled_indices(shots);
        std::mt19937_64 rng(config.seed);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // 4. Time Evolution & Measurement Loops
        std::unordered_set<SpinState, SpinStateHash> unique_set;
        uint64_t total_hamming_flips = 0;

        int threads = 256;
        int blocks = static_cast<int>((num_states + threads - 1) / threads);

        int total_trotter_steps = static_cast<int>(std::round(config.t_max / config.dt));
        double sample_interval = (config.dt_sample > 0.0) 
            ? config.dt_sample 
            : (config.t_max / std::max(1, config.num_sample_points));

        double next_sample_time = config.t_start;
        double traj_interval = (config.dt_trajectory > 0.0) ? config.dt_trajectory : config.dt;
        double next_traj_time = 0.0;

        for (int step = 0; step <= total_trotter_steps; ++step) {
            double current_t = step * config.dt;

            // Continuous trajectory recording
            if (config.record_trajectory) {
                if (current_t >= next_traj_time - 1e-9 && next_traj_time <= config.t_max + 1e-9) {
                    double p_less_val = compute_p_less_gpu(
                        d_state, num_states, static_cast<double>(result.start_energy), d_out_prob);
                    result.time_points.push_back(current_t);
                    result.p_less_history.push_back(p_less_val);
                    if (p_less_val > result.peak_p_less) {
                        result.peak_p_less = p_less_val;
                        result.peak_time = current_t;
                    }
                    next_traj_time += traj_interval;
                }
            } else if (config.track_p_less) {
                // Coarse delayed-window measurement points
                if (current_t >= next_sample_time - 1e-9 && next_sample_time <= config.t_max + 1e-9) {
                    double p_less_val = compute_p_less_gpu(
                        d_state, num_states, static_cast<double>(result.start_energy), d_out_prob);
                    result.time_points.push_back(current_t);
                    result.p_less_history.push_back(p_less_val);
                    if (p_less_val > result.peak_p_less) {
                        result.peak_p_less = p_less_val;
                        result.peak_time = current_t;
                    }
                }
            }

            // Bitstring projective sampling (if shots > 0)
            if (config.shots_per_point > 0 && current_t >= config.t_start &&
                current_t >= next_sample_time - 1e-9 && next_sample_time <= config.t_max + 1e-9) {
                CHECK_CUSTATEVEC(custatevecSamplerPreprocess(
                    handle, sampler_desc, d_workspace, workspace_size));

                for (int i = 0; i < shots; ++i) rand_nums[i] = dist(rng);

                CHECK_CUSTATEVEC(custatevecSamplerSample(
                    handle, sampler_desc, sampled_indices.data(), bit_ordering.data(),
                    static_cast<uint32_t>(config.n), rand_nums.data(),
                    static_cast<uint32_t>(shots), CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER));

                for (int i = 0; i < shots; ++i) {
                    uint64_t bit_val = static_cast<uint64_t>(sampled_indices[i]);
                    SpinState s{bit_val};
                    unique_set.insert(s);
                    total_hamming_flips += start_state.hamming_distance(s);
                    if (eval.calcEnergy(s) < result.start_energy) {
                        result.improved_shots++;
                    }
                }
                result.total_shots_collected += shots;
                next_sample_time += sample_interval;
            } else if (config.shots_per_point == 0 && !config.record_trajectory && config.track_p_less &&
                       current_t >= next_sample_time - 1e-9 && next_sample_time <= config.t_max + 1e-9) {
                next_sample_time += sample_interval;
            }

            if (step < total_trotter_steps) {
                // Apply 1 Trotter step:
                // (i) Diagonal LABS phase: e^{-i (dt / n) * H_Z}
                apply_diagonal_phase_kernel<<<blocks, threads>>>(
                    d_state, d_cached_energies, num_states, config.n, config.dt);
                CHECK_CUDA(cudaGetLastError());

                // (ii) Transverse field rotations: Rx(-2*g*dt) on all qubits
                for (int32_t qubit = 0; qubit < config.n; ++qubit) {
                    CHECK_CUSTATEVEC(custatevecApplyMatrix(
                        handle, d_state, CUDA_C_64F, static_cast<uint32_t>(config.n),
                        rx_matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                        0, &qubit, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
                }
            }
        }

        // 5. Package results
        result.unique_basis.assign(unique_set.begin(), unique_set.end());
        result.total_evals = result.total_shots_collected;
        if (result.total_shots_collected > 0) {
            result.mean_hamming_flips = static_cast<double>(total_hamming_flips) / static_cast<double>(result.total_shots_collected);
            result.success_rate = static_cast<double>(result.improved_shots) / static_cast<double>(result.total_shots_collected);
            result.per_eval_rate = (result.total_evals > 0) 
                ? (static_cast<double>(result.improved_shots) / static_cast<double>(result.total_evals)) 
                : 0.0;
        }

        // 6. Log successes/trials
        if (config.verbose) {
            std::cout << "[Quantum Sampler] Successes / Trials: " 
                      << result.improved_shots << " / " << result.total_shots_collected 
                      << " (" << std::fixed << std::setprecision(2) << (result.success_rate * 100.0) << "%)"
                      << " | E < E_og (E_og = " << result.start_energy << ")"
                      << " | Evals: " << result.total_evals
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << result.per_eval_rate
                      << std::defaultfloat << "\n";
        }

        // Cleanup
        custatevecSamplerDestroy(sampler_desc);
        if (d_workspace) cudaFree(d_workspace);
        if (d_out_prob) cudaFree(d_out_prob);
        custatevecDestroy(handle);
        cudaFree(d_state);

        return result;
    }
};

#else

// Fallback stub when cuStateVec headers are not available
class CuStateVecSampler : public ISampler {
public:
    SamplingResult sample(const SpinState&, const SamplerConfig&) override {
        throw std::runtime_error("cuStateVec backend is not available in this build (requires CUDA Toolkit and cuQuantum).");
    }
};

#endif // HAS_CUSTATEVEC

// ============================================================================
// cuTensorNet Sampler Backend (Placeholder for N > 30)
// ============================================================================

class CuTensorNetSampler : public ISampler {
public:
    SamplingResult sample(const SpinState&, const SamplerConfig&) override {
        throw std::runtime_error("cuTensorNet backend for N > 30 is under development.");
    }
};

// ============================================================================
// Dynamic Factory
// ============================================================================

inline std::unique_ptr<ISampler> create_sampler(int n) {
    if (n <= 30) {
        return std::make_unique<CuStateVecSampler>();
    } else {
        return std::make_unique<CuTensorNetSampler>();
    }
}