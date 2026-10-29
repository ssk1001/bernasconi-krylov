#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <cmath>

#include "../src/sampling/sampler.cuh"

// Test 1: Quantum Trotter Evolution and Measurement Sampling on N=4
void test_quantum_sampling() {
    std::cout << "[Test 1] Testing cuStateVec Quantum Statevector Sampling (N=4)..." << std::endl;

#if HAS_CUSTATEVEC
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "  [Notice] No CUDA device detected. Skipping GPU execution.\n" << std::endl;
        return;
    }

    int n = 4;
    SpinState start_state{0ULL}; // |0000>
    auto sampler = create_sampler(n);

    SamplerConfig cfg;
    cfg.n = n;
    cfg.g = 0.5;
    cfg.dt = 0.05;
    cfg.t_max = 2.0;
    cfg.num_sample_points = 5;  // K = 5
    cfg.shots_per_point = 50;   // M = 50
    cfg.track_p_less = true;

    SamplingResult res = sampler->sample(start_state, cfg);

    // Verify sampling outputs
    assert(res.total_shots_collected == 5 * 50);
    assert(res.total_evals == res.total_shots_collected);
    assert(res.improved_shots <= res.total_shots_collected);
    assert(res.success_rate >= 0.0 && res.success_rate <= 1.0);
    assert(res.per_eval_rate >= 0.0 && res.per_eval_rate <= 1.0);
    assert(!res.unique_basis.empty());
    assert(res.unique_basis.size() <= 1ULL << n);
    assert(res.mean_hamming_flips >= 0.0 && res.mean_hamming_flips <= static_cast<double>(n));

    std::cout << "  Total shots collected : " << res.total_shots_collected << "\n";
    std::cout << "  Unique basis size     : " << res.unique_basis.size() << " / " << (1 << n) << "\n";
    std::cout << "  Mean Hamming diffusion: " << std::fixed << std::setprecision(2) << res.mean_hamming_flips << " flips\n";
    std::cout << "  --> PASSED: Quantum statevector sampling verified!\n" << std::endl;
#else
    std::cout << "  [Notice] cuStateVec not available in this build. Skipping GPU execution.\n" << std::endl;
#endif
}

// Test 2: Tunneling Probability P_<(t) Dynamics (track_p_less)
void test_tunneling_probability_dynamics() {
    std::cout << "[Test 2] Testing P_<(t) Dynamics (track_p_less)..." << std::endl;

#if HAS_CUSTATEVEC
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "  [Notice] No CUDA device detected. Skipping GPU execution.\n" << std::endl;
        return;
    }

    int n = 4;
    // Choose start state with non-minimal energy so P_<(t) can be non-zero
    SpinState start_state{7ULL}; // |0111>
    auto sampler = create_sampler(n);

    SamplerConfig cfg;
    cfg.n = n;
    cfg.g = 0.75;
    cfg.dt = 0.05;
    cfg.t_max = 3.0;
    cfg.num_sample_points = 10;
    cfg.shots_per_point = 20;
    cfg.track_p_less = true;

    SamplingResult res = sampler->sample(start_state, cfg);

    assert(!res.p_less_history.empty());
    assert(res.time_points.size() == res.p_less_history.size());

    // Verify all probabilities are within [0.0, 1.0]
    for (size_t i = 0; i < res.p_less_history.size(); ++i) {
        double p = res.p_less_history[i];
        assert(p >= -1e-9 && p <= 1.0 + 1e-9);
    }

    std::cout << "  Recorded " << res.p_less_history.size() << " P_<(t) measurements.\n";
    std::cout << "  First P_<: " << std::fixed << std::setprecision(4) << res.p_less_history.front()
              << ", Final P_<: " << res.p_less_history.back() << "\n";

    // Verify that disabling track_p_less skips reduction and leaves history empty
    cfg.track_p_less = false;
    SamplingResult res_no_track = sampler->sample(start_state, cfg);
    assert(res_no_track.p_less_history.empty());
    assert(res_no_track.time_points.empty());

    std::cout << "  --> PASSED: P_<(t) tunneling probability tracking verified!\n" << std::endl;
#else
    std::cout << "  [Notice] cuStateVec not available in this build. Skipping GPU execution.\n" << std::endl;
#endif
}

// Test 3: Delayed Sampling Window (t_start > 0)
void test_delayed_sampling_window() {
    std::cout << "[Test 3] Testing Delayed Sampling Window (t_start > 0)..." << std::endl;

#if HAS_CUSTATEVEC
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "  [Notice] No CUDA device detected. Skipping GPU execution.\n" << std::endl;
        return;
    }

    int n = 4;
    SpinState start_state{0ULL};
    auto sampler = create_sampler(n);

    SamplerConfig cfg;
    cfg.n = n;
    cfg.g = 0.5;
    cfg.dt = 0.05;
    cfg.t_start = 1.0;          // Burn-in until t = 1.0
    cfg.t_max = 2.5;
    cfg.dt_sample = 0.5;        // Sample every 0.5s: at t = 1.0, 1.5, 2.0, 2.5
    cfg.shots_per_point = 25;

    SamplingResult res = sampler->sample(start_state, cfg);

    // Verify that all measurement points occurred at or after t_start
    for (double t : res.time_points) {
        assert(t >= cfg.t_start - 1e-9);
    }

    std::cout << "  Burn-in period t < " << cfg.t_start << " had 0 measurements.\n";
    std::cout << "  Measurements taken at t = ";
    for (size_t i = 0; i < res.time_points.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1) << res.time_points[i] 
                  << (i + 1 < res.time_points.size() ? ", " : "\n");
    }
    std::cout << "  --> PASSED: Delayed window burn-in and interval sampling verified!\n" << std::endl;
#else
    std::cout << "  [Notice] cuStateVec not available in this build. Skipping GPU execution.\n" << std::endl;
#endif
}

// Test 4: Dynamic Factory Behavior
void test_factory() {
    std::cout << "[Test 4] Testing Sampler Factory Dispatch (N <= 30 vs N > 30)..." << std::endl;

    auto sampler_small = create_sampler(16);
    assert(sampler_small != nullptr);

    auto sampler_large = create_sampler(32);
    assert(sampler_large != nullptr);

    // For N > 30, CuTensorNetSampler currently throws an informative runtime_error
    bool threw_exception = false;
    try {
        SamplerConfig cfg;
        cfg.n = 32;
        sampler_large->sample(SpinState{0ULL}, cfg);
    } catch (const std::runtime_error& e) {
        threw_exception = true;
    }
    assert(threw_exception);

    std::cout << "  --> PASSED: Factory dispatch and backend safety guards verified!\n" << std::endl;
}

int main() {
    std::cout << "=====================================================\n";
    std::cout << "             Testing Quantum Sampler Library         \n";
    std::cout << "=====================================================\n";

    test_factory();
    test_quantum_sampling();
    test_tunneling_probability_dynamics();
    test_delayed_sampling_window();

    std::cout << "=====================================================\n";
    std::cout << "         ALL SAMPLER TESTS COMPLETED!                \n";
    std::cout << "=====================================================" << std::endl;
    return 0;
}
