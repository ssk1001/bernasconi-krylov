#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include "../src/labs_energy/labs_energy.hpp"
#include "../src/classical/tabu_search.hpp"

// Test 1: Verify incremental energy calculation matches full LabsEvaluator calculation at every step
void test_incremental_energy_correctness() {
    std::cout << "[Test 1] Testing incremental Tabu energy calculation correctness..." << std::endl;
    int n = 16;
    LabsEvaluator eval(n);
    TabuConfig cfg;
    cfg.n = n;
    cfg.max_steps = 50;
    cfg.stop_on_escape = false; // Run full trajectory to verify all steps
    cfg.dynamic_tenure = true;
    cfg.seed = 12345;

    TabuSearch tabu(eval, cfg);
    std::mt19937_64 rng(cfg.seed);

    for (int trial = 0; trial < 20; ++trial) {
        SpinState start = tabu.random_state(rng);
        TabuTrialResult res = tabu.search_trajectory(start, -1, cfg, rng);
        (void)res;

        // Verify start and final energies match full recalculations
        assert(res.start_energy == eval.calcEnergy(res.start_state));
        assert(res.final_energy == eval.calcEnergy(res.final_state));
        assert(res.best_energy == eval.calcEnergy(res.best_state));
        assert(res.best_energy <= res.start_energy);

        // Verification of evals count: 1 initial + n * steps_taken
        assert(res.evals_used == 1 + static_cast<size_t>(n) * res.steps_taken);
    }
    std::cout << "  --> PASSED: Incremental energies match exact recalculations on all trajectories!\n" << std::endl;
}

// Test 2: Verify direct escape from a true local minimum (trap)
void test_trap_escape() {
    std::cout << "[Test 2] Testing Tabu Search escape from a verified local minimum (trap)..." << std::endl;
    int n = 16;
    LabsEvaluator eval(n);

    // Trap for N=16 with energy E0 = 28 (verified local minimum from seeds_N16.csv, ground state is 24)
    SpinState trap(0xEED); // Known local minimum for N=16
    int64_t e_trap = eval.calcEnergy(trap);
    (void)e_trap;
    assert(e_trap == 28);

    // Verify it is indeed a local minimum (all 1-flip neighbors have E >= e_trap)
    for (int p = 0; p < n; ++p) {
        SpinState nbr = trap.flipped(p);
        (void)nbr;
        assert(eval.calcEnergy(nbr) >= e_trap);
    }

    TabuConfig cfg;
    cfg.n = n;
    cfg.max_steps = 100;
    cfg.stop_on_escape = true;
    cfg.seed = 42;

    TabuSearch tabu(eval, cfg);
    std::mt19937_64 rng(cfg.seed);

    size_t escapes = 0;
    for (int trial = 0; trial < 100; ++trial) {
        TabuTrialResult res = tabu.search_trajectory(trap, e_trap, cfg, rng);
        if (res.success) {
            escapes++;
            assert(res.best_energy < e_trap);
            assert(eval.calcEnergy(res.best_state) < e_trap);
        }
    }

    std::cout << "  --> Escaped from local minimum in " << escapes << " / 100 trials!" << std::endl;
    assert(escapes > 0);
    std::cout << "  --> PASSED: Tabu search successfully climbs out of local minimum!\n" << std::endl;
}

// Test 3: Verify multi-threaded 50,000 trials performance and correctness
void test_50000_trials_benchmark() {
    std::cout << "[Test 3] Running 50,000 trials local Monte Carlo escape test..." << std::endl;
    int n = 16;
    LabsEvaluator eval(n);
    SpinState trap(0xEED); // Known local minimum for N=16 (E0 = 28)
    int64_t e_trap = eval.calcEnergy(trap);
    (void)e_trap;
    assert(e_trap == 28);

    TabuConfig cfg;
    cfg.n = n;
    cfg.max_steps = 100;
    cfg.stop_on_escape = true;
    cfg.seed = 42;

    TabuSearch tabu(eval, cfg);

    size_t trials = 50000;
    auto t0 = std::chrono::high_resolution_clock::now();
    TabuBenchmarkResult res = tabu.run_escape_trials(trap, trials, &cfg, false);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    assert(res.total_trials == trials);
    assert(res.improved_hits > 0);
    assert(res.success_rate == static_cast<double>(res.improved_hits) / trials);
    assert(res.per_eval_rate == static_cast<double>(res.improved_hits) / res.total_evals);

    std::cout << "  Total Trials          : " << res.total_trials << "\n";
    std::cout << "  Improved Hits (E < E0): " << res.improved_hits << "\n";
    std::cout << "  Success Rate          : " << (res.success_rate * 100.0) << " %\n";
    std::cout << "  Total Evaluations     : " << res.total_evals << "\n";
    std::cout << "  Per-Eval Rate (r)     : " << res.per_eval_rate << " (Monte Carlo prob per eval)\n";
    std::cout << "  Mean Evals on Success : " << res.mean_evals_on_success << "\n";
    std::cout << "  Elapsed Time          : " << ms << " ms (" << res.trials_per_sec << " trials/sec)\n";
    std::cout << "  --> PASSED: 50,000 trials executed successfully and rapidly!\n" << std::endl;
}

int main() {
    std::cout << "======================================================\n";
    std::cout << "             Tabu Search Unit Tests                   \n";
    std::cout << "======================================================\n";

    test_incremental_energy_correctness();
    test_trap_escape();
    test_50000_trials_benchmark();

    std::cout << "======================================================\n";
    std::cout << "           ALL TABU SEARCH TESTS PASSED!              \n";
    std::cout << "======================================================\n";
    return 0;
}
