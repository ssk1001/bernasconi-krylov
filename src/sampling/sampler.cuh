#include <vector>
#include <memory>
#include <unordered_map>
#include <stdexcept>
// Include your existing LABS energy library headers here
// This provides SpinState, SpinStateHash, SpinCache, SpinLib, etc.
#pragma once
#include "../labs_energy/labs_energy.hpp"

// Configuration Parameters for the Fixed-g Algorithm
struct SamplerConfig {
    size_t n;                  // System size N
    double g;                  // Transverse field strength g
    double dt;                 // Time step \Delta t
    size_t k_steps;            // Number of states \psi_m at times m\Delta t
    size_t m_samples;          // Number of samples collected per state
    size_t top_k_augmentation; // Threshold for highest-weight string augmentation
};

class ISampler {
public:
    virtual ~ISampler() = default;

    virtual std::vector<SpinState> generate_subspace(
        const std::vector<SpinState>& seed_states, 
        const SamplerConfig& config, 
        SpinCache& cache) = 0;
};

class CuStateVecSampler : public ISampler {
public:
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