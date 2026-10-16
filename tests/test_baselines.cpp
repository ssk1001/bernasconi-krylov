#include <iostream>
#include <cassert>
#include <random>
#include "baselines/budgeted_baselines.hpp"
#include "labs_energy/labs_energy.hpp"

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "    Testing Budgeted Baselines Module    " << std::endl;
    std::cout << "=========================================" << std::endl;

    size_t n = 16;
    std::mt19937_64 rng(42);
    SpinCache cache;

    // -------------------------------------------------------------------------
    // Test 1: steepest_descent_serial convergence & exact energy match
    // -------------------------------------------------------------------------
    std::cout << "[Test 1] Testing steepest_descent_serial convergence & energy correctness..." << std::endl;
    {
        SpinState rand_state;
        rand_state.num_bits = n;
        rand_state.bits.resize((n + 63) / 64);
        rand_state.bits[0] = rng() & ((1ULL << n) - 1);

        long long initial_e = SpinLib::get_energy(cache, rand_state);
        auto res = steepest_descent_serial(rand_state);

        assert(res.reached_local_min);
        assert(res.final_energy <= initial_e);
        assert(res.evals_used >= 1 + n);

        // Verify incremental energy matches SpinLib reference energy calculation
        long long ref_e = SpinLib::get_energy(cache, res.final_state);
        assert(res.final_energy == ref_e);

        // Verify that res.final_state is genuinely a local minimum (all 1-flip neighbors have >= energy)
        for (size_t i = 0; i < n; ++i) {
            SpinState neighbor = res.final_state.flip_bit(i);
            long long neighbor_e = SpinLib::get_energy(cache, neighbor);
            assert(neighbor_e >= res.final_energy);
        }

        std::cout << "  --> Initial E: " << initial_e << " | Local Min E: " << res.final_energy 
                  << " | Evals: " << res.evals_used << " [PASSED]" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 2: steepest_descent_serial strict budget enforcement
    // -------------------------------------------------------------------------
    std::cout << "[Test 2] Testing strict budget constraints in steepest_descent_serial..." << std::endl;
    {
        SpinState rand_state;
        rand_state.num_bits = n;
        rand_state.bits.resize((n + 63) / 64);
        rand_state.bits[0] = rng() & ((1ULL << n) - 1);

        size_t budget = 20; // Budget 20 can afford 1 initial eval + 1 step (16 evals) = 17 evals <= 20
        auto res = steepest_descent_serial(rand_state, budget);
        assert(res.evals_used <= budget);
        std::cout << "  --> Requested budget: " << budget << " | Evals used: " << res.evals_used << " [PASSED]" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 3: run_serial_sd_quench
    // -------------------------------------------------------------------------
    std::cout << "[Test 3] Testing run_serial_sd_quench..." << std::endl;
    {
        SpinState rand_state;
        rand_state.num_bits = n;
        rand_state.bits.resize((n + 63) / 64);
        rand_state.bits[0] = rng() & ((1ULL << n) - 1);

        auto quench = run_serial_sd_quench(rand_state);
        assert(quench.reached_local_min);
        long long ref_e = SpinLib::get_energy(cache, quench.final_state);
        assert(quench.final_energy == ref_e);
        std::cout << "  --> Quench reached local min with energy: " << quench.final_energy << " [PASSED]" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 4: run_serial_random_kick_sd
    // -------------------------------------------------------------------------
    std::cout << "[Test 4] Testing run_serial_random_kick_sd..." << std::endl;
    {
        // First get a seed local minimum
        SpinState rand_state;
        rand_state.num_bits = n;
        rand_state.bits.resize((n + 63) / 64);
        rand_state.bits[0] = rng() & ((1ULL << n) - 1);
        auto min_res = steepest_descent_serial(rand_state);
        SpinState seed_state = min_res.final_state;
        long long seed_energy = min_res.final_energy;

        size_t budget = 100;
        int hamming_d = 4;
        auto kick_res = run_serial_random_kick_sd(seed_state, seed_energy, hamming_d, budget, rng);

        assert(kick_res.evals_used <= budget);
        assert(kick_res.num_restarts >= 1);
        long long ref_e = SpinLib::get_energy(cache, kick_res.best_state);
        assert(kick_res.best_energy == ref_e);
        std::cout << "  --> Seed E: " << seed_energy << " | Best Kick E: " << kick_res.best_energy 
                  << " | Kicks executed: " << kick_res.num_restarts 
                  << " | Evals used: " << kick_res.evals_used << " <= " << budget << " [PASSED]" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 5: run_serial_random_restart_sd
    // -------------------------------------------------------------------------
    std::cout << "[Test 5] Testing run_serial_random_restart_sd..." << std::endl;
    {
        long long seed_energy = 50;
        size_t budget = 120;
        auto restart_res = run_serial_random_restart_sd(n, seed_energy, budget, rng);

        assert(restart_res.evals_used <= budget);
        assert(restart_res.num_restarts >= 1);
        long long ref_e = SpinLib::get_energy(cache, restart_res.best_state);
        assert(restart_res.best_energy == ref_e);
        std::cout << "  --> Best Restart E: " << restart_res.best_energy 
                  << " | Restarts: " << restart_res.num_restarts 
                  << " | Evals used: " << restart_res.evals_used << " <= " << budget << " [PASSED]" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 6: run_serial_blind_sampling
    // -------------------------------------------------------------------------
    std::cout << "[Test 6] Testing run_serial_blind_sampling..." << std::endl;
    {
        long long seed_energy = 50;
        size_t budget = 50;
        auto blind_res = run_serial_blind_sampling(n, seed_energy, budget, rng);

        assert(blind_res.evals_used == budget);
        assert(blind_res.num_restarts == budget);
        long long ref_e = SpinLib::get_energy(cache, blind_res.best_state);
        assert(blind_res.best_energy == ref_e);
        std::cout << "  --> Best Blind E: " << blind_res.best_energy 
                  << " | Evals used: " << blind_res.evals_used << " == " << budget << " [PASSED]" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 7: run_serial_kick_distance_scan
    // -------------------------------------------------------------------------
    std::cout << "[Test 7] Testing run_serial_kick_distance_scan across d = {2, 4, 6}..." << std::endl;
    {
        SpinState rand_state;
        rand_state.num_bits = n;
        rand_state.bits = {rng() & ((1ULL << n) - 1)};
        auto min_res = steepest_descent_serial(rand_state);
        SpinState seed_state = min_res.final_state;
        long long seed_energy = min_res.final_energy;

        std::vector<int> d_grid = {2, 4, 6};
        double mean_flips = 4.2;
        size_t budget_per_d = 80;

        auto scan_res = run_serial_kick_distance_scan(
            seed_state, seed_energy, d_grid, mean_flips, budget_per_d, rng);

        assert(scan_res.matched_d == 4);
        assert(scan_res.results_by_d.size() == 3);
        for (int d : d_grid) {
            assert(scan_res.results_by_d[d].evals_used <= budget_per_d);
        }
        std::cout << "  --> Matched d: " << scan_res.matched_d << " (expected 4) | Best d: " 
                  << scan_res.best_d << " (E=" << scan_res.best_energy << ") [PASSED]" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 8: run_large_scale_kick_sd_rate
    // -------------------------------------------------------------------------
    std::cout << "[Test 8] Testing run_large_scale_kick_sd_rate..." << std::endl;
    {
        SpinState rand_state;
        rand_state.num_bits = n;
        rand_state.bits = {rng() & ((1ULL << n) - 1)};
        auto min_res = steepest_descent_serial(rand_state);
        SpinState seed_state = min_res.final_state;
        long long seed_energy = min_res.final_energy;

        size_t num_trials = 1000;
        int d = 4;
        auto rate_res = run_large_scale_kick_sd_rate(seed_state, seed_energy, d, num_trials, rng);

        assert(rate_res.d == d);
        assert(rate_res.total_trials == num_trials);
        assert(rate_res.total_evals >= num_trials * (1 + n));
        assert(std::abs(rate_res.success_rate - (double)rate_res.total_successes / num_trials) < 1e-12);
        assert(std::abs(rate_res.per_eval_rate - (double)rate_res.total_successes / rate_res.total_evals) < 1e-12);
        assert(std::abs(rate_res.mean_evals_per_trial - (double)rate_res.total_evals / num_trials) < 1e-12);

        double budget_succ = rate_res.effective_budget_succ_rate(2000);
        assert(budget_succ >= 0.0 && budget_succ <= 1.0);

        std::cout << "  --> Trials: " << rate_res.total_trials 
                  << " | Successes: " << rate_res.total_successes 
                  << " | Evals: " << rate_res.total_evals
                  << " | Rate r: " << rate_res.per_eval_rate 
                  << " | Budget-2000 Succ: " << (budget_succ * 100.0) << "% [PASSED]" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 9: run_large_scale_kick_scan (argmax over average rates)
    // -------------------------------------------------------------------------
    std::cout << "[Test 9] Testing run_large_scale_kick_scan average rate selection..." << std::endl;
    {
        SpinState rand_state;
        rand_state.num_bits = n;
        rand_state.bits = {rng() & ((1ULL << n) - 1)};
        auto min_res = steepest_descent_serial(rand_state);
        SpinState seed_state = min_res.final_state;
        long long seed_energy = min_res.final_energy;

        std::vector<int> d_grid = {2, 4, 6};
        double mean_flips = 3.8;
        size_t trials_per_d = 500;

        auto scan_res = run_large_scale_kick_scan(seed_state, seed_energy, d_grid, mean_flips, trials_per_d, rng);

        assert(scan_res.matched_d == 4);
        assert(scan_res.results_by_d.size() == 3);

        // Verify that best_d indeed has the maximum per_eval_rate among scanned d's
        for (int d : d_grid) {
            assert(scan_res.best_rate.per_eval_rate >= scan_res.results_by_d[d].per_eval_rate - 1e-12);
        }

        std::cout << "  --> Matched d: " << scan_res.matched_d 
                  << " | Best d*: " << scan_res.best_d 
                  << " (Rate r*: " << scan_res.best_rate.per_eval_rate << ") [PASSED]" << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 10: run_large_scale_restart_sd_rate
    // -------------------------------------------------------------------------
    std::cout << "[Test 10] Testing run_large_scale_restart_sd_rate..." << std::endl;
    {
        long long seed_energy = 50;
        size_t num_trials = 1000;
        auto restart_res = run_large_scale_restart_sd_rate(n, seed_energy, num_trials, rng);

        assert(restart_res.d == -1);
        assert(restart_res.total_trials == num_trials);
        assert(restart_res.total_evals >= num_trials * (1 + n));
        assert(std::abs(restart_res.success_rate - (double)restart_res.total_successes / num_trials) < 1e-12);
        assert(std::abs(restart_res.per_eval_rate - (double)restart_res.total_successes / restart_res.total_evals) < 1e-12);

        std::cout << "  --> Trials: " << restart_res.total_trials 
                  << " | Successes: " << restart_res.total_successes 
                  << " | Rate r: " << restart_res.per_eval_rate << " [PASSED]" << std::endl;
    }

    std::cout << "\nALL BUDGETED BASELINES TESTS PASSED SUCCESSFULLY!" << std::endl;
    return 0;
}
