#include "../src/classical/classical.cuh"
#include <iostream>
#include <cassert>
#include <vector>
#include <cstdio>

// Test 1: Verify CPU steepest descent monotonic decrease and local optimality
void test_single_descent_optimality() {
    std::cout << "[Test 1] Testing single-trajectory steepest descent local optimality..." << std::endl;
    int n = 16;
    LabsEvaluator eval(n);
    ClassicalOptimizer opt(eval, 12345);

    for (int trial = 0; trial < 20; ++trial) {
        SpinState start = opt.random_state();
        int64_t e_start = eval.calcEnergy(start);
        DescentResult res = opt.steepest_descent(start);

        // Final energy must be <= initial energy
        assert(res.final_energy <= e_start);
        assert(res.final_energy == eval.calcEnergy(res.final_state));

        // If it terminated as a local minimum, verify all N 1-bit neighbors have E >= E_min
        if (res.reached_local_min) {
            for (int p = 0; p < n; ++p) {
                SpinState neighbor = res.final_state.flipped(p);
                int64_t e_nbr = eval.calcEnergy(neighbor);
                assert(e_nbr >= res.final_energy);
            }
        }

        // Eval count check: 1 initial + N evaluations per pass (steps_taken + 1 for final verification)
        assert(res.evals_used == 1 + static_cast<size_t>(n) * (res.steps_taken + (res.reached_local_min ? 1 : 0)));
    }
    std::cout << "  --> PASSED: All trajectories converged to true local minima!\n" << std::endl;
}

// Test 2: Verify strict budget enforcement
void test_budget_enforcement() {
    std::cout << "[Test 2] Testing strict evaluation budget enforcement..." << std::endl;
    int n = 20;
    LabsEvaluator eval(n);
    ClassicalOptimizer opt(eval, 54321);

    size_t budget = 1000;
    SpinState og_state = opt.random_state();
    BudgetedResult b_res = opt.random_restarts(og_state, budget, false);

    // Total evals used must never exceed budget
    assert(b_res.total_evals_used <= budget);
    assert(b_res.total_trajectories > 0);
    assert(b_res.best_energy == eval.calcEnergy(b_res.best_state));

    std::cout << "  Budget = " << budget 
              << " | Total evals used = " << b_res.total_evals_used 
              << " | Trajectories = " << b_res.total_trajectories << "\n";
    std::cout << "  --> PASSED: Evaluation budget was strictly respected!\n" << std::endl;
}

// Test 3: Verify GPU batch steepest descent matches CPU steepest descent
void test_gpu_cpu_equivalence() {
    std::cout << "[Test 3] Testing GPU vs CPU steepest descent equivalence..." << std::endl;

#if HAS_CUDA_CLASSICAL
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cout << "  [Notice] No CUDA device detected. Skipping GPU equivalence test.\n" << std::endl;
        return;
    }

    int n = 16;
    LabsEvaluator eval(n);
    ClassicalOptimizer opt(eval, 9999);

    const int num_starts = 64;
    std::vector<SpinState> start_states(num_starts);
    for (int i = 0; i < num_starts; ++i) {
        start_states[i] = opt.random_state();
    }

    // Run on GPU
    BatchDescentResult gpu_res = opt.run_batch_gpu(start_states);

    // Run on CPU
    for (int i = 0; i < num_starts; ++i) {
        DescentResult cpu_res = opt.steepest_descent(start_states[i]);
        assert(gpu_res.final_energies[i] == cpu_res.final_energy);
        assert(gpu_res.final_states[i] == cpu_res.final_state);
        assert(gpu_res.eval_counts[i] == static_cast<int64_t>(cpu_res.evals_used));
    }

    std::cout << "  --> PASSED: All " << num_starts << " GPU trajectories matched CPU bit-for-bit!\n" << std::endl;
#else
    std::cout << "  [Notice] CUDA not available in this build. Skipping GPU equivalence test.\n" << std::endl;
#endif
}

// Test 4: Verify distinct minima collection and symmetry deduplication
void test_distinct_minima_collection() {
    std::cout << "[Test 4] Testing distinct minima collection and symmetry deduplication..." << std::endl;
    int n = 12;
    LabsEvaluator eval(n);
    ClassicalOptimizer opt(eval, 777);

    size_t target_families = 5;
    auto families = opt.collect_distinct_minima_cpu(target_families, 10000, LogLevel::Silent);

    assert(families.size() == target_families);

    // Verify all canonical states are unique
    for (size_t i = 0; i < families.size(); ++i) {
        assert(families[i].canonical_state == families[i].canonical_state.canonical(n));
        for (size_t j = i + 1; j < families.size(); ++j) {
            assert(families[i].canonical_state != families[j].canonical_state);
        }
    }

    // Verify sorted by energy ascending
    for (size_t i = 1; i < families.size(); ++i) {
        assert(families[i].energy >= families[i - 1].energy);
    }

    std::cout << "  --> PASSED: Collected " << families.size() << " unique, sorted symmetry families!\n" << std::endl;
}

// Test 5: Verify random kick generation and Hamming distance
void test_random_kick() {
    std::cout << "[Test 5] Testing random kick perturbation and Hamming distance..." << std::endl;
    int n = 20;
    LabsEvaluator eval(n);
    ClassicalOptimizer opt(eval, 10101);

    SpinState seed = opt.random_state();
    for (int d = 1; d <= 10; ++d) {
        SpinState kicked = opt.random_kick_state(seed, d);
        assert(seed.hamming_distance(kicked) == d);
    }

    BudgetedResult kick_res = opt.random_kick(seed, 4, 1000, false);
    assert(kick_res.total_evals_used <= 1000);
    assert(kick_res.total_trajectories > 0);

    std::cout << "  --> PASSED: Exact Hamming distance d and kick search verified!\n" << std::endl;
}

// Test 6: Verify CSV export and run reconstruction
void test_csv_export() {
    std::cout << "[Test 6] Testing CSV export and metadata reconstruction..." << std::endl;
    int n = 10;
    LabsEvaluator eval(n);
    ClassicalOptimizer opt(eval, 2026);

    auto families = opt.collect_distinct_minima_cpu(3, 5000, LogLevel::Silent);
    ClassicalRunMetadata meta;
    meta.n = n;
    meta.seed = 2026;
    meta.algorithm_name = "Test Run";
    meta.total_starts = 100;
    meta.total_evals = 1500;
    meta.elapsed_ms = 4.2;
    meta.device_name = "CPU Serial";

    const std::string tmp_csv = "/tmp/test_minima_export.csv";
    bool ok = ClassicalLogger::export_to_csv(tmp_csv, families, meta);
    assert(ok);

    // Verify file exists and is non-empty
    std::ifstream in(tmp_csv);
    assert(in.is_open());
    std::string line;
    std::getline(in, line);
    assert(line.find("# N=10,Seed=2026") != std::string::npos);
    in.close();
    std::remove(tmp_csv.c_str());

    std::cout << "  --> PASSED: CSV export and metadata headers verified!\n" << std::endl;
}

// Test 7: Verify successes/trials and per-eval rates logging across all classical methods
void test_successes_and_rates() {
    std::cout << "[Test 7] Testing Successes/Trials and Per-Eval Rates Logging..." << std::endl;
    int n = 12;
    LabsEvaluator eval(n);
    ClassicalOptimizer opt(eval, 3030);

    // Pick an og_state with non-minimal energy
    SpinState og_state{15ULL};
    int64_t og_e = eval.calcEnergy(og_state);

    // 1. Steepest Descent (Fixed Budget)
    std::cout << "--- 1. Steepest Descent (Fixed Budget) ---" << std::endl;
    DescentResult sd_res = opt.steepest_descent(og_state, 100, true);
    assert(sd_res.evals_used <= 100);
    assert(sd_res.success == (sd_res.final_energy < og_e));
    assert(sd_res.success_rate == (sd_res.success ? 1.0 : 0.0));
    assert(std::abs(sd_res.per_eval_rate - (sd_res.success ? (1.0 / sd_res.evals_used) : 0.0)) < 1e-9);

    // 2. Multi-Start Steepest Descent
    std::cout << "--- 2. Multi-Start Steepest Descent ---" << std::endl;
    MultiStartResult ms_res = opt.multi_start_descent(og_state, 10, 50, false, true);
    assert(ms_res.total_starts == 10);
    assert(ms_res.total_evals > 0);
    assert(std::abs(ms_res.success_rate - (static_cast<double>(ms_res.improved_hits) / ms_res.total_starts)) < 1e-9);
    assert(std::abs(ms_res.per_eval_rate - (static_cast<double>(ms_res.improved_hits) / ms_res.total_evals)) < 1e-9);

    // 3. Random Restarts
    std::cout << "--- 3. Random Restarts ---" << std::endl;
    BudgetedResult rr_res = opt.random_restarts(300, og_state, true);
    assert(rr_res.total_evals_used <= 300);
    assert(rr_res.total_trajectories > 0);
    assert(std::abs(rr_res.success_rate - (static_cast<double>(rr_res.improved_hits) / rr_res.total_trajectories)) < 1e-9);
    assert(std::abs(rr_res.per_eval_rate - (static_cast<double>(rr_res.improved_hits) / rr_res.total_evals_used)) < 1e-9);

    // 4. Random Kick
    std::cout << "--- 4. Random Kick ---" << std::endl;
    BudgetedResult rk_res = opt.random_kick(og_state, 2, 300, true);
    assert(rk_res.total_evals_used <= 300);
    assert(rk_res.total_trajectories > 0);
    assert(std::abs(rk_res.success_rate - (static_cast<double>(rk_res.improved_hits) / rk_res.total_trajectories)) < 1e-9);
    assert(std::abs(rk_res.per_eval_rate - (static_cast<double>(rk_res.improved_hits) / rk_res.total_evals_used)) < 1e-9);

    std::cout << "  --> PASSED: All classical success tracking and rate logs verified!\n" << std::endl;
}

int main() {
    std::cout << "=====================================================\n";
    std::cout << "          Testing Classical Optimizer Library        \n";
    std::cout << "=====================================================\n";

    test_single_descent_optimality();
    test_budget_enforcement();
    test_gpu_cpu_equivalence();
    test_distinct_minima_collection();
    test_random_kick();
    test_csv_export();
    test_successes_and_rates();

    std::cout << "=====================================================\n";
    std::cout << " ALL CLASSICAL OPTIMIZER TESTS PASSED SUCCESSFULLY!  \n";
    std::cout << "=====================================================" << std::endl;
    return 0;
}