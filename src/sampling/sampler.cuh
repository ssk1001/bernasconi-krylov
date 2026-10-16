#include <vector>
#include <memory>
#include <unordered_map>
#include <stdexcept>
// Include your existing LABS energy library headers here
// This provides SpinState, SpinStateHash, SpinCache, SpinLib, etc.
#include <cuda_runtime.h>
#include <cuComplex.h>
#pragma once
#include "../labs_energy/labs_energy.hpp"

// Forward declarations of GPU kernels implemented in custatevec_sampler.cu
__global__ void precompute_labs_energies_kernel(
    double *d_energies,
    uint64_t num_states,
    size_t n);

__global__ void apply_precomputed_phase_kernel(
    cuDoubleComplex *d_state,
    const double *d_energies,
    uint64_t num_states,
    size_t n,
    double dt);

// Configuration Parameters for the Fixed-g Algorithm
struct SamplerConfig {
    size_t n = 16;                  // System size N
    double g = 0.5;                 // Transverse field strength g
    double dt = 0.1;                // Time step \Delta t
    size_t k_steps = 5;             // Number of states \psi_m at times m\Delta t
    size_t m_samples = 400;         // Number of samples collected per state
    size_t top_k_augmentation = 0;  // Threshold for highest-weight string augmentation (0 = disabled)
    uint64_t seed = 42;             // RNG seed for quantum measurement sampling

    // Delayed sampling window parameters
    double t_start = 0.0;           // Delayed window start time (e.g. 10.0; 0 = start at t=0)
    double dt_sample = 0.0;         // Interval between measurement points (e.g. 1.5; <= 0 = sample at dt)

    // Statevector P_< computation
    bool compute_p_less = false;    // Whether to compute exact P_< from statevector at each sample time
    double seed_energy = 0.0;       // Seed local minimum energy E0 for P_< computation
};

class ISampler {
public:
    virtual ~ISampler() = default;

    virtual std::vector<SpinState> generate_subspace(
        const std::vector<SpinState>& seed_states, 
        const SamplerConfig& config, 
        SpinCache& cache) = 0;

    virtual double get_last_mean_flips() const { return 0.0; }
    virtual double get_last_window_mean_p_less() const { return 0.0; }
    virtual const std::vector<double>& get_last_p_less_history() const {
        static std::vector<double> empty;
        return empty;
    }
};

class CuStateVecSampler : public ISampler {
private:
    size_t cached_n = 0;
    double *d_cached_energies = nullptr;
    double *d_out_prob = nullptr;
    double last_mean_flips = 0.0;
    double last_window_mean_p_less = 0.0;
    std::vector<double> last_p_less_history;

public:
    CuStateVecSampler() = default;
    ~CuStateVecSampler() override;

    double get_last_mean_flips() const override { return last_mean_flips; }
    double get_last_window_mean_p_less() const override { return last_window_mean_p_less; }
    const std::vector<double>& get_last_p_less_history() const override { return last_p_less_history; }

    std::vector<SpinState> generate_subspace(
        const std::vector<SpinState>& seed_states, 
        const SamplerConfig& config, 
        SpinCache& cache) override;
};


class CuTensorNetSampler : public ISampler {
public:
    std::vector<SpinState> generate_subspace(
        const std::vector<SpinState>& seed_states, 
        const SamplerConfig& config, 
        SpinCache& cache) override;
};

// Dynamic Backend Factory
inline std::unique_ptr<ISampler> create_sampler(size_t n) {
    if (n <= 30) {
        return std::make_unique<CuStateVecSampler>();
    } else {
        // return std::make_unique<CuTensorNetSampler>();
        throw std::runtime_error("CuTensorNetSampler backend is not yet implemented for N > 30.");
    }
}