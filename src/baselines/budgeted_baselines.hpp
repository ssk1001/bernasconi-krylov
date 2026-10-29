#ifndef BUDGETED_BASELINES_HPP
#define BUDGETED_BASELINES_HPP

// ============================================================================
// DEPRECATED / REDUNDANT
// This file is superseded by src/classical/classical.cuh (ClassicalOptimizer::random_restarts & random_kick).
// Do not use in new implementations.
// ============================================================================

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <cstddef>
#include <map>
#include "../labs_energy/labs_energy.hpp"

// Result of a single serial steepest descent trajectory
struct SerialDescentResult {
    SpinState final_state;
    long long final_energy = 0;
    size_t evals_used = 0;
    bool reached_local_min = false;
};

// Summary result of a budgeted baseline comparison method
struct BudgetedComparisonResult {
    SpinState best_state;
    long long best_energy = std::numeric_limits<long long>::max();
    size_t evals_used = 0;
    size_t num_restarts = 0; // Number of restarts, kicks, or random samples executed
    bool improved_seed = false;
};

/**
 * @brief Core Serial Steepest Descent Stepper with strict budget enforcement.
 * 
 * Evaluates all N 1-bit neighbors incrementally in O(N) per neighbor (O(N^2) per step).
 * Greedily flips the bit giving the steepest energy descent.
 * Halts when a local minimum is reached or when the remaining evaluation budget
 * is insufficient to test all N neighbors.
 * 
 * @param start_state Initial spin configuration.
 * @param max_evals Maximum classical energy evaluations allowed (default: unlimited).
 * @return SerialDescentResult containing the best state, energy, evals consumed, and min flag.
 */
inline SerialDescentResult steepest_descent_serial(
    const SpinState& start_state,
    size_t max_evals = std::numeric_limits<size_t>::max())
{
    SerialDescentResult result;
    if (max_evals == 0) {
        result.final_state = start_state;
        return result;
    }

    size_t n = start_state.num_bits;
    SpinState current_state = start_state;
    if (current_state.bits.empty()) {
        current_state.bits.resize((n + 63) / 64, 0);
    }

    // 1. Initial full O(N^2) autocorrelation and energy computation
    std::vector<long long> C(n, 0);
    for (size_t k = 1; k < n; ++k) {
        int diffs = 0;
        for (size_t i = 0; i < n - k; ++i) {
            if (current_state.get_spin(i) != current_state.get_spin(i + k)) {
                diffs++;
            }
        }
        C[k] = static_cast<long long>(n - k) - 2LL * diffs;
    }

    long long current_energy = 0;
    for (size_t k = 1; k < n; ++k) {
        current_energy += C[k] * C[k];
    }

    result.evals_used = 1; // 1 initial full evaluation
    result.final_state = current_state;
    result.final_energy = current_energy;

    // 2. Iterative steepest descent loop
    while (true) {
        // If remaining budget cannot evaluate all N single-bit neighbors, halt immediately
        if (result.evals_used + n > max_evals) {
            break;
        }

        bool improved = false;
        long long best_neighbor_energy = current_energy;
        size_t best_bit = 0;

        // Evaluate all N single-bit flips incrementally in O(N) operations per neighbor
        for (size_t p = 0; p < n; ++p) {
            size_t word_p = p / 64;
            size_t bit_p = p % 64;
            uint64_t bp_old = (current_state.bits[word_p] >> bit_p) & 1ULL;

            long long neighbor_energy = 0;
            for (size_t k = 1; k < n; ++k) {
                long long delta_k = 0;
                if (p + k < n) {
                    size_t w2 = (p + k) / 64;
                    size_t b2 = (p + k) % 64;
                    uint64_t bp_k = (current_state.bits[w2] >> b2) & 1ULL;
                    int prod = (bp_old == bp_k) ? 1 : -1;
                    delta_k += -2 * prod;
                }
                if (p >= k) {
                    size_t w1 = (p - k) / 64;
                    size_t b1 = (p - k) % 64;
                    uint64_t b_pmk = (current_state.bits[w1] >> b1) & 1ULL;
                    int prod = (b_pmk == bp_old) ? 1 : -1;
                    delta_k += -2 * prod;
                }
                long long new_Ck = C[k] + delta_k;
                neighbor_energy += new_Ck * new_Ck;
            }

            if (neighbor_energy < best_neighbor_energy) {
                best_neighbor_energy = neighbor_energy;
                improved = true;
                best_bit = p;
            }
        }

        result.evals_used += n;

        if (improved) {
            // Apply the best bit flip and update C[k] values permanently
            size_t word_best = best_bit / 64;
            size_t bit_best = best_bit % 64;
            uint64_t bp_old = (current_state.bits[word_best] >> bit_best) & 1ULL;

            for (size_t k = 1; k < n; ++k) {
                long long delta_k = 0;
                if (best_bit + k < n) {
                    size_t w2 = (best_bit + k) / 64;
                    size_t b2 = (best_bit + k) % 64;
                    uint64_t bp_k = (current_state.bits[w2] >> b2) & 1ULL;
                    int prod = (bp_old == bp_k) ? 1 : -1;
                    delta_k += -2 * prod;
                }
                if (best_bit >= k) {
                    size_t w1 = (best_bit - k) / 64;
                    size_t b1 = (best_bit - k) % 64;
                    uint64_t b_pmk = (current_state.bits[w1] >> b1) & 1ULL;
                    int prod = (b_pmk == bp_old) ? 1 : -1;
                    delta_k += -2 * prod;
                }
                C[k] += delta_k;
            }

            current_state.bits[word_best] ^= (1ULL << bit_best);
            current_energy = best_neighbor_energy;
            result.final_state = current_state;
            result.final_energy = current_energy;
        } else {
            result.reached_local_min = true;
            break;
        }
    }

    return result;
}

/**
 * @brief Method 2: Steepest Descent Quench from the best quantum-sampled state.
 * Relaxes the quantum state to its local minimum basin (unconstrained budget).
 */
inline SerialDescentResult run_serial_sd_quench(const SpinState& best_subspace_state)
{
    return steepest_descent_serial(best_subspace_state, std::numeric_limits<size_t>::max());
}

/**
 * @brief Method 3: Budgeted Random Kick + Steepest Descent Loop.
 * 
 * At each iteration, perturbs the seed minimum by flipping exactly `hamming_d`
 * randomly selected distinct bits, then runs steepest descent until budget exhaustion.
 */
inline BudgetedComparisonResult run_serial_random_kick_sd(
    const SpinState& seed_state,
    long long seed_energy,
    int hamming_d,
    size_t total_budget,
    std::mt19937_64& rng)
{
    BudgetedComparisonResult result;
    size_t n = seed_state.num_bits;
    if (total_budget == 0 || n == 0) return result;

    int d = std::max(1, std::min(hamming_d, static_cast<int>(n)));
    std::vector<size_t> indices(n);

    while (result.evals_used < total_budget) {
        size_t remaining = total_budget - result.evals_used;
        if (remaining == 0) break;

        // Partial Fisher-Yates shuffle to pick d distinct bits uniformly at random
        std::iota(indices.begin(), indices.end(), 0);
        for (int i = 0; i < d; ++i) {
            std::uniform_int_distribution<size_t> dist(i, n - 1);
            size_t j = dist(rng);
            std::swap(indices[i], indices[j]);
        }

        SpinState kicked_state = seed_state;
        for (int i = 0; i < d; ++i) {
            kicked_state = kicked_state.flip_bit(indices[i]);
        }

        SerialDescentResult sd = steepest_descent_serial(kicked_state, remaining);
        result.evals_used += sd.evals_used;
        result.num_restarts++;

        if (sd.final_energy < result.best_energy) {
            result.best_energy = sd.final_energy;
            result.best_state = sd.final_state;
        }

        if (result.evals_used >= total_budget) {
            break;
        }
    }

    result.improved_seed = (result.best_energy < seed_energy);
    return result;
}

/**
 * @brief Method 4: Budgeted Multi-Start Steepest Descent (Random Restarts).
 * 
 * Repeatedly samples uniform random states in {0, 1}^N and descends until total
 * evaluations reach the allocated budget.
 */
inline BudgetedComparisonResult run_serial_random_restart_sd(
    size_t n,
    long long seed_energy,
    size_t total_budget,
    std::mt19937_64& rng)
{
    BudgetedComparisonResult result;
    if (total_budget == 0 || n == 0) return result;

    size_t num_words = (n + 63) / 64;

    while (result.evals_used < total_budget) {
        size_t remaining = total_budget - result.evals_used;
        if (remaining == 0) break;

        // Generate uniform random configuration in {0, 1}^N
        SpinState s;
        s.num_bits = n;
        s.bits.resize(num_words, 0);
        for (size_t w = 0; w < num_words; ++w) {
            s.bits[w] = rng();
        }
        if (n % 64 != 0) {
            s.bits.back() &= (1ULL << (n % 64)) - 1;
        }

        SerialDescentResult sd = steepest_descent_serial(s, remaining);
        result.evals_used += sd.evals_used;
        result.num_restarts++;

        if (sd.final_energy < result.best_energy) {
            result.best_energy = sd.final_energy;
            result.best_state = sd.final_state;
        }

        if (result.evals_used >= total_budget) {
            break;
        }
    }

    result.improved_seed = (result.best_energy < seed_energy);
    return result;
}

/**
 * @brief Method 5: Budgeted Blind Random Sampling.
 * 
 * Draws uniform random states from {0, 1}^N without any descent, testing whether
 * pure random chance under the same evaluation budget discovers better configurations.
 */
inline BudgetedComparisonResult run_serial_blind_sampling(
    size_t n,
    long long seed_energy,
    size_t total_budget,
    std::mt19937_64& rng)
{
    BudgetedComparisonResult result;
    if (total_budget == 0 || n == 0) return result;

    size_t num_words = (n + 63) / 64;

    for (size_t eval = 0; eval < total_budget; ++eval) {
        SpinState s;
        s.num_bits = n;
        s.bits.resize(num_words, 0);
        for (size_t w = 0; w < num_words; ++w) {
            s.bits[w] = rng();
        }
        if (n % 64 != 0) {
            s.bits.back() &= (1ULL << (n % 64)) - 1;
        }

        long long energy = 0;
        for (size_t k = 1; k < n; ++k) {
            int diffs = 0;
            for (size_t i = 0; i < n - k; ++i) {
                if (s.get_spin(i) != s.get_spin(i + k)) {
                    diffs++;
                }
            }
            long long Ck = static_cast<long long>(n - k) - 2LL * diffs;
            energy += Ck * Ck;
        }

        result.evals_used++;
        result.num_restarts++;

        if (energy < result.best_energy) {
            result.best_energy = energy;
            result.best_state = s;
        }
    }

    result.improved_seed = (result.best_energy < seed_energy);
    return result;
}

// Result of scanning multiple kick distances d with fixed budget per d
struct KickScanResult {
    std::map<int, BudgetedComparisonResult> results_by_d;
    int best_d = 2;
    int matched_d = 2;
    long long best_energy = std::numeric_limits<long long>::max();
    bool best_improved = false;
};

/**
 * @brief Scans random kick distances d in d_values, executing budgeted kick+SD with
 * total_budget_per_d for each distance d.
 * Identifies both the optimal d* and the d that matches mean_sample_flips.
 */
inline KickScanResult run_serial_kick_distance_scan(
    const SpinState& seed_state,
    long long seed_energy,
    const std::vector<int>& d_values,
    double mean_sample_flips,
    size_t total_budget_per_d,
    std::mt19937_64& rng)
{
    KickScanResult scan_res;
    if (d_values.empty()) return scan_res;

    // Find the d that is closest to round(mean_sample_flips)
    int target_match = std::max(1, static_cast<int>(std::round(mean_sample_flips)));
    int min_diff = std::numeric_limits<int>::max();
    for (int d : d_values) {
        int diff = std::abs(d - target_match);
        if (diff < min_diff) {
            min_diff = diff;
            scan_res.matched_d = d;
        }
    }

    for (int d : d_values) {
        std::mt19937_64 d_rng(rng() ^ (0xdeadbeefULL + static_cast<uint64_t>(d) * 10007ULL));
        BudgetedComparisonResult res = run_serial_random_kick_sd(
            seed_state, seed_energy, d, total_budget_per_d, d_rng);

        scan_res.results_by_d[d] = res;

        if (res.best_energy < scan_res.best_energy) {
            scan_res.best_energy = res.best_energy;
            scan_res.best_d = d;
            scan_res.best_improved = res.improved_seed;
        }
    }

    return scan_res;
}

// -----------------------------------------------------------------------------
// Large-Scale Ensemble Baselines (10^4 - 10^5 trials per parameter)
// -----------------------------------------------------------------------------

// Result of large-scale statistical trials for a single distance d (or restart)
struct ClassicalRateResult {
    int d = 0;                          // Perturbation distance (-1 for Random Restart)
    size_t total_trials = 0;            // Total independent trajectories (e.g. 10^4 - 10^5)
    size_t total_successes = 0;         // Number of trajectories with final_energy < seed_energy
    size_t total_evals = 0;             // Cumulative classical energy evaluations across all trials
    double success_rate = 0.0;          // Per-trial success rate: total_successes / total_trials
    double per_eval_rate = 0.0;         // Per-eval rate: total_successes / total_evals (r)
    double mean_evals_per_trial = 0.0;  // total_evals / total_trials
    long long min_energy = std::numeric_limits<long long>::max();
    double mean_final_energy = 0.0;

    // Effective success probability under a fixed evaluation budget B (e.g. B=2000):
    // P(success in budget B) = 1 - (1 - success_rate)^(B / mean_evals_per_trial)
    double effective_budget_succ_rate(size_t budget_B) const {
        if (mean_evals_per_trial <= 0.0 || total_trials == 0 || success_rate <= 0.0) return 0.0;
        if (success_rate >= 1.0) return 1.0;
        double k_restarts = static_cast<double>(budget_B) / mean_evals_per_trial;
        return std::max(0.0, std::min(1.0, 1.0 - std::pow(1.0 - success_rate, k_restarts)));
    }
};

struct LargeScaleKickScanResult {
    std::map<int, ClassicalRateResult> results_by_d;
    int best_d = 2;
    int matched_d = 2;
    ClassicalRateResult best_rate;
    ClassicalRateResult matched_rate;
};

/**
 * @brief Runs num_trials (10^4 - 10^5) independent trajectories of Kick(d) + Steepest Descent.
 * Computes exact total evaluations, total successes, per-trial rate, and per-evaluation rate r = S / E.
 */
inline ClassicalRateResult run_large_scale_kick_sd_rate(
    const SpinState& seed_state,
    long long seed_energy,
    int hamming_d,
    size_t num_trials,
    std::mt19937_64& rng)
{
    ClassicalRateResult result;
    result.d = hamming_d;
    result.total_trials = num_trials;
    if (num_trials == 0 || seed_state.num_bits == 0) return result;

    size_t n = seed_state.num_bits;
    int d = std::max(1, std::min(hamming_d, static_cast<int>(n)));
    std::vector<size_t> indices(n);
    double sum_final_energy = 0.0;

    for (size_t t = 0; t < num_trials; ++t) {
        // Partial Fisher-Yates shuffle to pick d distinct bits uniformly at random
        std::iota(indices.begin(), indices.end(), 0);
        for (int i = 0; i < d; ++i) {
            std::uniform_int_distribution<size_t> dist(i, n - 1);
            size_t j = dist(rng);
            std::swap(indices[i], indices[j]);
        }

        SpinState kicked_state = seed_state;
        for (int i = 0; i < d; ++i) {
            kicked_state = kicked_state.flip_bit(indices[i]);
        }

        SerialDescentResult sd = steepest_descent_serial(kicked_state);
        result.total_evals += sd.evals_used;
        sum_final_energy += static_cast<double>(sd.final_energy);
        if (sd.final_energy < result.min_energy) {
            result.min_energy = sd.final_energy;
        }

        if (sd.final_energy < seed_energy) {
            result.total_successes++;
        }
    }

    result.success_rate = static_cast<double>(result.total_successes) / static_cast<double>(num_trials);
    result.per_eval_rate = (result.total_evals > 0) 
        ? (static_cast<double>(result.total_successes) / static_cast<double>(result.total_evals)) 
        : 0.0;
    result.mean_evals_per_trial = static_cast<double>(result.total_evals) / static_cast<double>(num_trials);
    result.mean_final_energy = sum_final_energy / static_cast<double>(num_trials);

    return result;
}

/**
 * @brief Evaluates each distance d in d_values over num_trials_per_d independent trials.
 * Selects d* from the macroscopic average per-evaluation rates: d* = argmax_d (r_d).
 */
inline LargeScaleKickScanResult run_large_scale_kick_scan(
    const SpinState& seed_state,
    long long seed_energy,
    const std::vector<int>& d_values,
    double mean_sample_flips,
    size_t num_trials_per_d,
    std::mt19937_64& rng)
{
    LargeScaleKickScanResult scan_res;
    if (d_values.empty()) return scan_res;

    // Find the d that is closest to round(mean_sample_flips)
    int target_match = std::max(1, static_cast<int>(std::round(mean_sample_flips)));
    int min_diff = std::numeric_limits<int>::max();
    for (int d : d_values) {
        int diff = std::abs(d - target_match);
        if (diff < min_diff) {
            min_diff = diff;
            scan_res.matched_d = d;
        }
    }

    int best_d = d_values[0];
    double best_per_eval_rate = -1.0;
    double best_succ_rate = -1.0;
    double best_mean_e = std::numeric_limits<double>::max();

    for (int d : d_values) {
        std::mt19937_64 d_rng(rng() ^ (0xdeadbeefULL + static_cast<uint64_t>(d) * 10007ULL));
        ClassicalRateResult rate_res = run_large_scale_kick_sd_rate(
            seed_state, seed_energy, d, num_trials_per_d, d_rng);

        scan_res.results_by_d[d] = rate_res;

        // Choose d* from the macroscopic average rates across the trials
        bool is_better = false;
        if (rate_res.per_eval_rate > best_per_eval_rate) {
            is_better = true;
        } else if (std::abs(rate_res.per_eval_rate - best_per_eval_rate) < 1e-12) {
            // Primary tie-break: higher per-trial success rate
            if (rate_res.success_rate > best_succ_rate) {
                is_better = true;
            } else if (std::abs(rate_res.success_rate - best_succ_rate) < 1e-12) {
                // Secondary tie-break: lower mean final energy
                if (rate_res.mean_final_energy < best_mean_e) {
                    is_better = true;
                } else if (std::abs(rate_res.mean_final_energy - best_mean_e) < 1e-6 && d < best_d) {
                    // Final tie-break: prefer smaller distance d
                    is_better = true;
                }
            }
        }

        if (is_better) {
            best_per_eval_rate = rate_res.per_eval_rate;
            best_succ_rate = rate_res.success_rate;
            best_mean_e = rate_res.mean_final_energy;
            best_d = d;
        }
    }

    scan_res.best_d = best_d;
    scan_res.best_rate = scan_res.results_by_d[best_d];
    scan_res.matched_rate = scan_res.results_by_d[scan_res.matched_d];

    return scan_res;
}

/**
 * @brief Runs num_trials (10^4 - 10^5) independent trajectories of Random Restart + Steepest Descent.
 * Computes exact total evaluations, total successes, per-trial rate, and per-evaluation rate r = S / E.
 */
inline ClassicalRateResult run_large_scale_restart_sd_rate(
    size_t n,
    long long seed_energy,
    size_t num_trials,
    std::mt19937_64& rng)
{
    ClassicalRateResult result;
    result.d = -1; // -1 indicates random restart
    result.total_trials = num_trials;
    if (num_trials == 0 || n == 0) return result;

    size_t num_words = (n + 63) / 64;
    double sum_final_energy = 0.0;

    for (size_t t = 0; t < num_trials; ++t) {
        SpinState s;
        s.num_bits = n;
        s.bits.resize(num_words, 0);
        for (size_t w = 0; w < num_words; ++w) {
            s.bits[w] = rng();
        }
        if (n % 64 != 0) {
            s.bits.back() &= (1ULL << (n % 64)) - 1;
        }

        SerialDescentResult sd = steepest_descent_serial(s);
        result.total_evals += sd.evals_used;
        sum_final_energy += static_cast<double>(sd.final_energy);
        if (sd.final_energy < result.min_energy) {
            result.min_energy = sd.final_energy;
        }

        if (sd.final_energy < seed_energy) {
            result.total_successes++;
        }
    }

    result.success_rate = static_cast<double>(result.total_successes) / static_cast<double>(num_trials);
    result.per_eval_rate = (result.total_evals > 0) 
        ? (static_cast<double>(result.total_successes) / static_cast<double>(result.total_evals)) 
        : 0.0;
    result.mean_evals_per_trial = static_cast<double>(result.total_evals) / static_cast<double>(num_trials);
    result.mean_final_energy = sum_final_energy / static_cast<double>(num_trials);

    return result;
}

#endif // BUDGETED_BASELINES_HPP
