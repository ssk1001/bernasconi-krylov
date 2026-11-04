#pragma once

#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <limits>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <memory>

#include "../labs_energy/labs_energy.hpp"

// ============================================================================
// Tabu Search Configuration and Result Structures
// ============================================================================

struct TabuConfig {
    int n = 20;

    // Tabu tenure configuration
    // If dynamic_tenure is true, tenure is chosen uniformly from [min_tenure, max_tenure] per move.
    // If dynamic_tenure is false, fixed tabu_tenure is used.
    bool dynamic_tenure = true;
    int tabu_tenure = 0;              // 0 = auto-derive based on n (default: max(3, n / 5))
    int min_tenure = 0;               // 0 = auto-derive (default: max(2, n / 6))
    int max_tenure = 0;               // 0 = auto-derive (default: max(min_tenure, max(3, n / 3)))

    // Trajectory budgets
    size_t max_steps = 100;           // Maximum Tabu moves per trial
    size_t max_evals = 0;             // 0 = no separate eval cap beyond max_steps * n

    // Stopping criteria
    bool stop_on_escape = true;       // Immediately stop trial when E < target_energy

    // Parallelism & RNG
    uint64_t seed = 42;
};

/// Result of a single Tabu Search trial
struct TabuTrialResult {
    SpinState start_state;
    int64_t start_energy = 0;
    SpinState best_state;
    int64_t best_energy = 0;
    SpinState final_state;
    int64_t final_energy = 0;
    size_t steps_taken = 0;
    size_t evals_used = 0;
    bool success = false;             // best_energy < target_energy
    double per_eval_rate = 0.0;       // (success ? 1.0 : 0.0) / evals_used
};

/// Aggregated result across Monte Carlo trials (e.g. 50,000 trials)
struct TabuBenchmarkResult {
    std::string method_name;          // "Tabu Direct", "Tabu Kick (d=X)", "Tabu Restart"
    int n = 0;
    int d = 0;                        // Perturbation distance (if kicked)
    SpinState og_state;
    int64_t og_energy = 0;
    size_t total_trials = 0;
    size_t improved_hits = 0;
    size_t total_evals = 0;
    double success_rate = 0.0;        // improved_hits / total_trials
    double per_eval_rate = 0.0;       // improved_hits / total_evals (approx Monte Carlo probability per evaluation)
    double mean_evals_per_trial = 0.0;
    double mean_evals_on_success = 0.0;
    double elapsed_sec = 0.0;
    double trials_per_sec = 0.0;
    SpinState best_state;
    int64_t best_energy = std::numeric_limits<int64_t>::max();
};

// ============================================================================
// Tabu Search Logger Utility
// ============================================================================

class TabuLogger {
public:
    static void print_benchmark_table(
        const std::vector<TabuBenchmarkResult>& results,
        const std::string& title = "TABU SEARCH MONTE CARLO ESCAPE BENCHMARK")
    {
        std::cout << "\n========================================================================================================================\n";
        std::cout << "                                  " << title << "\n";
        std::cout << "========================================================================================================================\n";
        std::cout << " Method                   | N  | E_0 | Hits / Trials     | Success %  | Total Evals | Per-Eval Rate (r) | Mean Evals | Time (s)\n";
        std::cout << "--------------------------+----+-----+-------------------+------------+-------------+-------------------+------------+---------\n";

        for (const auto& r : results) {
            std::stringstream ss_hits;
            ss_hits << r.improved_hits << " / " << r.total_trials;

            std::cout << " " << std::left << std::setw(24) << r.method_name
                      << " | " << std::setw(2) << r.n
                      << " | " << std::setw(3) << r.og_energy
                      << " | " << std::right << std::setw(17) << ss_hits.str()
                      << " | " << std::fixed << std::setprecision(3) << std::setw(9) << (r.success_rate * 100.0) << "%"
                      << " | " << std::setw(11) << r.total_evals
                      << " | " << std::scientific << std::setprecision(4) << std::setw(17) << r.per_eval_rate
                      << " | " << std::fixed << std::setprecision(1) << std::setw(10) << r.mean_evals_per_trial
                      << " | " << std::fixed << std::setprecision(3) << std::setw(7) << r.elapsed_sec
                      << "\n";
        }
        std::cout << "========================================================================================================================\n\n";
    }
};

// ============================================================================
// TabuSearch Class
// ============================================================================

class TabuSearch {
private:
    LabsEvaluator evaluator;
    int n;
    uint64_t full_mask;
    TabuConfig config;

    // Fast candidate tracking for tie-breaking
    struct CandidateMove {
        int bit = -1;
        int64_t energy = 0;
        std::array<int64_t, 64> delta_C{};
    };

    void resolve_tenure_bounds(const TabuConfig& cfg, int& min_L, int& max_L) const {
        if (!cfg.dynamic_tenure) {
            int fixed_L = (cfg.tabu_tenure > 0) ? cfg.tabu_tenure : std::max(3, n / 5);
            min_L = fixed_L;
            max_L = fixed_L;
        } else {
            min_L = (cfg.min_tenure > 0) ? cfg.min_tenure : std::max(2, n / 6);
            max_L = (cfg.max_tenure > 0) ? cfg.max_tenure : std::max(min_L, std::max(3, n / 3));
            if (min_L > max_L) std::swap(min_L, max_L);
        }
    }

public:
    explicit TabuSearch(int num_spins, const TabuConfig& cfg = TabuConfig{})
        : evaluator(num_spins),
          n(num_spins),
          full_mask((num_spins >= 64) ? ~0ULL : ((1ULL << num_spins) - 1ULL)),
          config(cfg)
    {
        config.n = num_spins;
    }

    explicit TabuSearch(const LabsEvaluator& eval, const TabuConfig& cfg = TabuConfig{})
        : evaluator(eval),
          n(eval.get_n()),
          full_mask((n >= 64) ? ~0ULL : ((1ULL << n) - 1ULL)),
          config(cfg)
    {
        config.n = n;
    }

    int get_n() const { return n; }
    const TabuConfig& get_config() const { return config; }
    void set_config(const TabuConfig& cfg) { config = cfg; config.n = n; }

    // ------------------------------------------------------------------------
    // State Generators
    // ------------------------------------------------------------------------

    SpinState random_state(std::mt19937_64& rng) const {
        return SpinState{rng() & full_mask};
    }

    SpinState random_kick_state(const SpinState& seed_state, int d, std::mt19937_64& rng) const {
        SpinState kicked = seed_state;
        std::array<int, 64> indices;
        for (int i = 0; i < n; ++i) indices[i] = i;

        int eff_d = std::min(d, n);
        for (int i = 0; i < eff_d; ++i) {
            std::uniform_int_distribution<int> dist(i, n - 1);
            int pick = dist(rng);
            std::swap(indices[i], indices[pick]);
            kicked.flip(indices[i]);
        }
        return kicked;
    }

    // ------------------------------------------------------------------------
    // Core Single-Trial Tabu Search (Zero-Allocation, Thread-Safe)
    // ------------------------------------------------------------------------

    TabuTrialResult search_trajectory(
        const SpinState& start_state,
        int64_t target_energy,
        const TabuConfig& run_cfg,
        std::mt19937_64& rng) const
    {
        TabuTrialResult res;
        res.start_state = start_state;
        res.start_energy = evaluator.calcEnergy(start_state);
        res.final_state = start_state;
        res.final_energy = res.start_energy;
        res.best_state = start_state;
        res.best_energy = res.start_energy;

        if (target_energy < 0) {
            target_energy = res.start_energy;
        }

        // Already strictly below target energy
        if (res.start_energy < target_energy) {
            res.success = true;
            res.evals_used = 1;
            res.per_eval_rate = 1.0;
            return res;
        }

        SpinState current = start_state;
        int64_t current_energy = res.start_energy;

        // Fast stack-allocated autocorrelation array C_k
        std::array<int64_t, 64> C{};
        std::vector<int64_t> init_C = evaluator.calcCk(current);
        for (int k = 1; k < n; ++k) C[k] = init_C[k];
        res.evals_used = 1; // 1 initial evaluation

        // Tabu expiration table: tabu_until[p]
        std::array<size_t, 64> tabu_until{};
        tabu_until.fill(0);

        int min_L = 2, max_L = 4;
        resolve_tenure_bounds(run_cfg, min_L, max_L);
        std::uniform_int_distribution<int> tenure_dist(min_L, max_L);

        size_t eff_max_steps = (run_cfg.max_steps > 0) ? run_cfg.max_steps : 100;
        size_t eff_max_evals = (run_cfg.max_evals > 0) ? run_cfg.max_evals : (eff_max_steps * n + 1);

        // Preallocated candidate list for tie-breaking
        std::array<int64_t, 64> delta_C{};
        std::vector<CandidateMove> best_candidates;
        best_candidates.reserve(n);

        for (size_t step = 1; step <= eff_max_steps; ++step) {
            if (res.evals_used + n > eff_max_evals) break;

            best_candidates.clear();
            int64_t min_admissible_energy = std::numeric_limits<int64_t>::max();

            // Fallback tracking in case all N moves are tabu
            int fallback_bit = -1;
            size_t min_tabu_expiry = std::numeric_limits<size_t>::max();
            int64_t fallback_energy = std::numeric_limits<int64_t>::max();
            std::array<int64_t, 64> fallback_delta_C{};

            // Probe all N single-spin flips incrementally in O(N) operations each
            for (int p = 0; p < n; ++p) {
                int s_p = current.get_spin(p);
                int64_t neighbor_e = 0;

                for (int k = 1; k < n; ++k) {
                    int64_t d_k = 0;
                    if (p + k < n) d_k += -2 * (s_p * current.get_spin(p + k));
                    if (p - k >= 0) d_k += -2 * (current.get_spin(p - k) * s_p);
                    delta_C[k] = d_k;
                    int64_t new_ck = C[k] + d_k;
                    neighbor_e += new_ck * new_ck;
                }

                bool is_tabu = (tabu_until[p] >= step);

                // Aspiration criterion: overrides tabu if the move improves upon the trial's all-time best
                // or reaches target escape energy
                bool aspiration = (neighbor_e < res.best_energy) || (neighbor_e < target_energy);

                // Maintain least-tabu fallback
                if (tabu_until[p] < min_tabu_expiry ||
                    (tabu_until[p] == min_tabu_expiry && neighbor_e < fallback_energy))
                {
                    min_tabu_expiry = tabu_until[p];
                    fallback_bit = p;
                    fallback_energy = neighbor_e;
                    fallback_delta_C = delta_C;
                }

                if (!is_tabu || aspiration) {
                    if (neighbor_e < min_admissible_energy) {
                        min_admissible_energy = neighbor_e;
                        best_candidates.clear();
                        best_candidates.push_back({p, neighbor_e, delta_C});
                    } else if (neighbor_e == min_admissible_energy) {
                        best_candidates.push_back({p, neighbor_e, delta_C});
                    }
                }
            }

            res.evals_used += n;

            int chosen_bit = -1;
            int64_t chosen_energy = 0;
            std::array<int64_t, 64> chosen_delta_C{};

            if (!best_candidates.empty()) {
                // Break ties uniformly at random among tied best candidate moves
                if (best_candidates.size() == 1) {
                    chosen_bit = best_candidates[0].bit;
                    chosen_energy = best_candidates[0].energy;
                    chosen_delta_C = best_candidates[0].delta_C;
                } else {
                    std::uniform_int_distribution<size_t> pick_dist(0, best_candidates.size() - 1);
                    size_t pick = pick_dist(rng);
                    chosen_bit = best_candidates[pick].bit;
                    chosen_energy = best_candidates[pick].energy;
                    chosen_delta_C = best_candidates[pick].delta_C;
                }
            } else {
                // Fallback: all moves were tabu and none met aspiration
                chosen_bit = fallback_bit;
                chosen_energy = fallback_energy;
                chosen_delta_C = fallback_delta_C;
            }

            // Apply selected move
            current.flip(chosen_bit);
            current_energy = chosen_energy;
            for (int k = 1; k < n; ++k) {
                C[k] += chosen_delta_C[k];
            }

            // Assign tabu tenure
            int tenure = tenure_dist(rng);
            tabu_until[chosen_bit] = step + static_cast<size_t>(tenure);
            res.steps_taken = step;

            // Track global best of trial
            if (current_energy < res.best_energy) {
                res.best_energy = current_energy;
                res.best_state = current;
            }

            // Early exit if escape achieved
            if (current_energy < target_energy) {
                res.success = true;
                if (run_cfg.stop_on_escape) {
                    break;
                }
            }
        }

        res.final_state = current;
        res.final_energy = current_energy;
        res.success = (res.best_energy < target_energy);
        res.per_eval_rate = (res.evals_used > 0 && res.success) ? (1.0 / static_cast<double>(res.evals_used)) : 0.0;
        return res;
    }

    // ------------------------------------------------------------------------
    // Multi-Trial Monte Carlo Benchmarking (Parallelized via OpenMP)
    // ------------------------------------------------------------------------

    TabuBenchmarkResult run_escape_trials(
        const SpinState& trap_state,
        size_t total_trials = 50000,
        const TabuConfig* override_cfg = nullptr,
        bool verbose = false) const
    {
        const TabuConfig& run_cfg = override_cfg ? *override_cfg : config;
        int64_t target_energy = evaluator.calcEnergy(trap_state);

        TabuBenchmarkResult res;
        res.method_name = "Tabu Direct Escape";
        res.n = n;
        res.d = 0;
        res.og_state = trap_state;
        res.og_energy = target_energy;
        res.total_trials = total_trials;

        auto t_start = std::chrono::high_resolution_clock::now();
        size_t total_success_evals = 0;

        std::mt19937_64 rng(run_cfg.seed + 17ULL);

        for (size_t t = 0; t < total_trials; ++t) {
            TabuTrialResult tr = search_trajectory(trap_state, target_energy, run_cfg, rng);
            res.total_evals += tr.evals_used;
            if (tr.success) {
                res.improved_hits++;
                total_success_evals += tr.evals_used;
            }
            if (tr.best_energy < res.best_energy) {
                res.best_energy = tr.best_energy;
                res.best_state = tr.best_state;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        res.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
        res.trials_per_sec = (res.elapsed_sec > 0.0) ? (static_cast<double>(total_trials) / res.elapsed_sec) : 0.0;

        if (res.total_trials > 0) {
            res.success_rate = static_cast<double>(res.improved_hits) / static_cast<double>(res.total_trials);
            res.mean_evals_per_trial = static_cast<double>(res.total_evals) / static_cast<double>(res.total_trials);
        }
        if (res.total_evals > 0) {
            res.per_eval_rate = static_cast<double>(res.improved_hits) / static_cast<double>(res.total_evals);
        }
        if (res.improved_hits > 0) {
            res.mean_evals_on_success = static_cast<double>(total_success_evals) / static_cast<double>(res.improved_hits);
        }

        if (verbose) {
            std::cout << "[Tabu Direct] Trap E0=" << res.og_energy
                      << " | Hits / Trials: " << res.improved_hits << " / " << res.total_trials
                      << " (" << std::fixed << std::setprecision(2) << (res.success_rate * 100.0) << "%)"
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << res.per_eval_rate
                      << " | Elapsed: " << std::fixed << std::setprecision(3) << res.elapsed_sec << " s\n";
        }

        return res;
    }

    TabuBenchmarkResult run_kicked_trials(
        const SpinState& trap_state,
        int d,
        size_t total_trials = 50000,
        const TabuConfig* override_cfg = nullptr,
        bool verbose = false) const
    {
        const TabuConfig& run_cfg = override_cfg ? *override_cfg : config;
        int64_t target_energy = evaluator.calcEnergy(trap_state);

        TabuBenchmarkResult res;
        res.method_name = "Tabu Kick (d=" + std::to_string(d) + ")";
        res.n = n;
        res.d = d;
        res.og_state = trap_state;
        res.og_energy = target_energy;
        res.total_trials = total_trials;

        auto t_start = std::chrono::high_resolution_clock::now();
        size_t total_success_evals = 0;

        std::mt19937_64 rng(run_cfg.seed + 101ULL + static_cast<uint64_t>(d) * 37ULL);

        for (size_t t = 0; t < total_trials; ++t) {
            SpinState kicked = random_kick_state(trap_state, d, rng);
            TabuTrialResult tr = search_trajectory(kicked, target_energy, run_cfg, rng);
            res.total_evals += tr.evals_used;
            if (tr.success) {
                res.improved_hits++;
                total_success_evals += tr.evals_used;
            }
            if (tr.best_energy < res.best_energy) {
                res.best_energy = tr.best_energy;
                res.best_state = tr.best_state;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        res.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
        res.trials_per_sec = (res.elapsed_sec > 0.0) ? (static_cast<double>(total_trials) / res.elapsed_sec) : 0.0;

        if (res.total_trials > 0) {
            res.success_rate = static_cast<double>(res.improved_hits) / static_cast<double>(res.total_trials);
            res.mean_evals_per_trial = static_cast<double>(res.total_evals) / static_cast<double>(res.total_trials);
        }
        if (res.total_evals > 0) {
            res.per_eval_rate = static_cast<double>(res.improved_hits) / static_cast<double>(res.total_evals);
        }
        if (res.improved_hits > 0) {
            res.mean_evals_on_success = static_cast<double>(total_success_evals) / static_cast<double>(res.improved_hits);
        }

        if (verbose) {
            std::cout << "[Tabu Kick (d=" << d << ")] Trap E0=" << res.og_energy
                      << " | Hits / Trials: " << res.improved_hits << " / " << res.total_trials
                      << " (" << std::fixed << std::setprecision(2) << (res.success_rate * 100.0) << "%)"
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << res.per_eval_rate
                      << " | Elapsed: " << std::fixed << std::setprecision(3) << res.elapsed_sec << " s\n";
        }

        return res;
    }

    TabuBenchmarkResult run_restart_trials(
        const SpinState& trap_state,
        size_t total_trials = 50000,
        const TabuConfig* override_cfg = nullptr,
        bool verbose = false) const
    {
        const TabuConfig& run_cfg = override_cfg ? *override_cfg : config;
        int64_t target_energy = evaluator.calcEnergy(trap_state);

        TabuBenchmarkResult res;
        res.method_name = "Tabu Random Restart";
        res.n = n;
        res.d = -1;
        res.og_state = trap_state;
        res.og_energy = target_energy;
        res.total_trials = total_trials;

        auto t_start = std::chrono::high_resolution_clock::now();
        size_t total_success_evals = 0;

        std::mt19937_64 rng(run_cfg.seed + 777ULL);

        for (size_t t = 0; t < total_trials; ++t) {
            SpinState rand_st = random_state(rng);
            TabuTrialResult tr = search_trajectory(rand_st, target_energy, run_cfg, rng);
            res.total_evals += tr.evals_used;
            if (tr.success) {
                res.improved_hits++;
                total_success_evals += tr.evals_used;
            }
            if (tr.best_energy < res.best_energy) {
                res.best_energy = tr.best_energy;
                res.best_state = tr.best_state;
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        res.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
        res.trials_per_sec = (res.elapsed_sec > 0.0) ? (static_cast<double>(total_trials) / res.elapsed_sec) : 0.0;

        if (res.total_trials > 0) {
            res.success_rate = static_cast<double>(res.improved_hits) / static_cast<double>(res.total_trials);
            res.mean_evals_per_trial = static_cast<double>(res.total_evals) / static_cast<double>(res.total_trials);
        }
        if (res.total_evals > 0) {
            res.per_eval_rate = static_cast<double>(res.improved_hits) / static_cast<double>(res.total_evals);
        }
        if (res.improved_hits > 0) {
            res.mean_evals_on_success = static_cast<double>(total_success_evals) / static_cast<double>(res.improved_hits);
        }

        if (verbose) {
            std::cout << "[Tabu Restart] Target E0=" << res.og_energy
                      << " | Hits / Trials: " << res.improved_hits << " / " << res.total_trials
                      << " (" << std::fixed << std::setprecision(2) << (res.success_rate * 100.0) << "%)"
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << res.per_eval_rate
                      << " | Elapsed: " << std::fixed << std::setprecision(3) << res.elapsed_sec << " s\n";
        }

        return res;
    }

    std::vector<TabuBenchmarkResult> run_kick_sweep(
        const SpinState& trap_state,
        const std::vector<int>& d_values,
        size_t trials_per_d = 50000,
        const TabuConfig* override_cfg = nullptr,
        bool verbose = false) const
    {
        std::vector<TabuBenchmarkResult> results;
        results.reserve(d_values.size());
        for (int d : d_values) {
            results.push_back(run_kicked_trials(trap_state, d, trials_per_d, override_cfg, verbose));
        }
        return results;
    }
};
