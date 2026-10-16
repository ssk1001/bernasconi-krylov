#pragma once

#include <vector>
#include <unordered_set>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include "symmetry.hpp"
#include "steepest_descent.cuh"

/**
 * Collects >= target_count distinct local minima of the LABS energy landscape
 * for system size n_spins, deduplicated by the 3 symmetry generators:
 *   1. Global spin negation: s -> -s
 *   2. Spatial reversal: (s_1, ..., s_N) -> (s_N, ..., s_1)
 *   3. Alternating spin flip: s_i -> (-1)^i s_i
 * 
 * Conducted in parallel GPU batches using multi_start_steepest_descent.
 * Logs real-time discovery rate, best energies, and cumulative classical evaluations.
 */
inline std::vector<SDMinimumRecord> collect_distinct_sd_minima(
    size_t n_spins,
    size_t target_count,
    SpinCache& cache,
    size_t batch_size = 256)
{
    std::cout << "\n======================================================================\n";
    std::cout << " [SD Discovery] Starting Collection of >= " << target_count 
              << " Symmetry-Distinct Minima (N=" << n_spins << ")\n";
    std::cout << "======================================================================\n";
    std::cout << "  Target Minima Count : " << target_count << "\n";
    std::cout << "  GPU Batch Size      : " << batch_size << " starts per launch\n";
    std::cout << "  Symmetry Generators : Negation, Reversal, Alternating Flip (Orbit size <= 8)\n";
    std::cout << "  Deduplication Mode  : In-memory canonical representative hashing\n";
    std::cout << "----------------------------------------------------------------------\n" << std::endl;

    std::vector<SDMinimumRecord> distinct_minima;
    std::unordered_set<SpinState, SpinStateHash> seen_canonical;

    long long total_starts = 0;
    long long total_evals = 0;
    int batch_num = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (distinct_minima.size() < target_count) {
        batch_num++;
        auto batch_start = std::chrono::high_resolution_clock::now();

        DescentResult batch_res = multi_start_steepest_descent(n_spins, batch_size, cache);

        auto batch_end = std::chrono::high_resolution_clock::now();
        double batch_time = std::chrono::duration<double>(batch_end - batch_start).count();

        size_t new_in_batch = 0;
        long long batch_min_energy = std::numeric_limits<long long>::max();

        for (size_t i = 0; i < batch_res.all_states.size(); ++i) {
            total_starts++;
            long long evals = (i < batch_res.all_evals.size()) 
                              ? batch_res.all_evals[i] 
                              : (1 + 20 * (long long)n_spins);
            total_evals += evals;

            const SpinState& s = batch_res.all_states[i];
            long long energy = batch_res.all_energies[i];
            batch_min_energy = std::min(batch_min_energy, energy);

            SpinState canon = SpinSymmetry::get_canonical_state(s);

            if (seen_canonical.find(canon) == seen_canonical.end()) {
                seen_canonical.insert(canon);

                SDMinimumRecord rec;
                rec.id = distinct_minima.size();
                rec.state = s;
                rec.canonical_state = canon;
                rec.energy = energy;
                rec.evals_to_find = evals;
                rec.start_index = total_starts;

                distinct_minima.push_back(rec);
                new_in_batch++;

                if (distinct_minima.size() >= target_count) {
                    break;
                }
            }
        }

        double elapsed_so_far = std::chrono::duration<double>(batch_end - start_time).count();
        long long current_best = distinct_minima.empty() ? batch_min_energy : distinct_minima[0].energy;
        for (const auto& m : distinct_minima) {
            current_best = std::min(current_best, m.energy);
        }

        std::cout << "[SD-Batch " << std::setw(3) << batch_num << "] "
                  << "Distinct: " << std::setw(4) << distinct_minima.size() << "/" << target_count
                  << " (+" << std::setw(3) << new_in_batch << " new) | "
                  << "Starts: " << std::setw(6) << total_starts << " | "
                  << "Evals: " << std::setw(9) << total_evals << " | "
                  << "Current Best E: " << std::setw(3) << current_best << " | "
                  << "Batch: " << std::fixed << std::setprecision(2) << batch_time << "s "
                  << "(Elapsed: " << std::setprecision(1) << elapsed_so_far << "s)"
                  << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_elapsed = std::chrono::duration<double>(end_time - start_time).count();

    // Sort distinct minima by energy ascending (lowest energy first)
    std::sort(distinct_minima.begin(), distinct_minima.end(), [](const SDMinimumRecord& a, const SDMinimumRecord& b) {
        return a.energy < b.energy;
    });

    // Re-assign sorted IDs
    for (size_t i = 0; i < distinct_minima.size(); ++i) {
        distinct_minima[i].id = i;
    }

    // Print full summary table and energy histogram
    SDLogging::print_minima_summary(distinct_minima, n_spins, total_starts, total_evals, total_elapsed);

    return distinct_minima;
}
