#ifndef SPIN_SYMMETRY_HPP
#define SPIN_SYMMETRY_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>
#include <unordered_set>
#include <sstream>
#include "../labs_energy/labs_energy.hpp"

namespace SpinSymmetry {

    /**
     * Operation 1: Global spin negation (s -> -s)
     * Negates all spins, which corresponds to inverting every bit (0 <-> 1).
     * Preserves LABS energy since C_k(-s) = C_k(s) -> C_k^2 is invariant.
     */
    inline SpinState negate(const SpinState& state) {
        SpinState res = state;
        size_t num_words = res.bits.size();
        for (size_t w = 0; w < num_words; ++w) {
            res.bits[w] = ~res.bits[w];
        }
        if (state.num_bits % 64 != 0) {
            uint64_t mask = (1ULL << (state.num_bits % 64)) - 1;
            res.bits[num_words - 1] &= mask;
        }
        return res;
    }

    /**
     * Operation 2: Spatial reversal ((s_1, ..., s_N) -> (s_N, ..., s_1))
     * Reverses the ordering of the spins.
     * Preserves LABS energy since C_k(rev(s)) = C_k(s) -> C_k^2 is invariant.
     */
    inline SpinState reverse(const SpinState& state) {
        SpinState res;
        res.num_bits = state.num_bits;
        res.bits.assign((state.num_bits + 63) / 64, 0ULL);
        for (size_t i = 0; i < state.num_bits; ++i) {
            size_t src_word = i / 64;
            size_t src_bit = i % 64;
            uint64_t bit_val = (state.bits[src_word] >> src_bit) & 1ULL;

            size_t dst_idx = state.num_bits - 1 - i;
            size_t dst_word = dst_idx / 64;
            size_t dst_bit = dst_idx % 64;
            res.bits[dst_word] |= (bit_val << dst_bit);
        }
        return res;
    }

    /**
     * Operation 3: Alternating spin flip (s_i -> (-1)^i s_i)
     * Flips alternating spins (in 1-indexed notation, odd positions i=1,3,5,... are flipped).
     * In 0-indexed bit representation, bits at even indices j=0,2,4,... are flipped.
     * Preserves LABS energy since C_k -> (-1)^k C_k -> C_k^2 is invariant.
     */
    inline SpinState alternating_flip(const SpinState& state) {
        SpinState res = state;
        size_t num_words = res.bits.size();
        // Mask with 1s at positions 0, 2, 4, ..., 62
        const uint64_t alt_mask = 0x5555555555555555ULL;
        for (size_t w = 0; w < num_words; ++w) {
            res.bits[w] ^= alt_mask;
        }
        if (state.num_bits % 64 != 0) {
            uint64_t mask = (1ULL << (state.num_bits % 64)) - 1;
            res.bits[num_words - 1] &= mask;
        }
        return res;
    }

    /**
     * Generates all up to 8 states in the symmetry orbit:
     * O(s) = { T1^a * T2^b * T3^c (s) | a,b,c in {0, 1} }
     * where T1 = negate, T2 = reverse, T3 = alternating_flip.
     */
    inline std::vector<SpinState> get_symmetry_orbit(const SpinState& state) {
        std::vector<SpinState> orbit;
        orbit.reserve(8);
        for (int a = 0; a < 2; ++a) {
            SpinState s_a = (a == 1) ? negate(state) : state;
            for (int b = 0; b < 2; ++b) {
                SpinState s_b = (b == 1) ? reverse(s_a) : s_a;
                for (int c = 0; c < 2; ++c) {
                    SpinState s_c = (c == 1) ? alternating_flip(s_b) : s_b;
                    orbit.push_back(s_c);
                }
            }
        }
        return orbit;
    }

    /**
     * Computes the unique canonical representative of the symmetry orbit.
     * Defined as the lexicographical minimum across all states in the orbit.
     */
    inline SpinState get_canonical_state(const SpinState& state) {
        auto orbit = get_symmetry_orbit(state);
        return *std::min_element(orbit.begin(), orbit.end());
    }

    /**
     * Returns true if two states belong to the same symmetry orbit.
     */
    inline bool are_symmetry_equivalent(const SpinState& a, const SpinState& b) {
        if (a.num_bits != b.num_bits) return false;
        return get_canonical_state(a) == get_canonical_state(b);
    }

    /**
     * Format a SpinState as a human-readable bitstring (e.g. "01101...").
     */
    inline std::string to_bitstring(const SpinState& s) {
        std::string str(s.num_bits, '0');
        for (size_t i = 0; i < s.num_bits; ++i) {
            size_t w = i / 64;
            size_t b = i % 64;
            if ((s.bits[w] >> b) & 1ULL) {
                str[s.num_bits - 1 - i] = '1';
            }
        }
        return str;
    }

} // namespace SpinSymmetry

// Record representing a discovered local minimum deduplicated by symmetry
struct SDMinimumRecord {
    size_t id;                      // Index: 0, 1, 2, ...
    SpinState state;                // The actual converged state found by descent
    SpinState canonical_state;      // Canonical representative of the symmetry orbit
    long long energy;               // Classical LABS energy E(s)
    long long evals_to_find;        // Classical energy evaluations spent to find it
    size_t start_index;             // Global start index that found this minimum
};

namespace SDLogging {

    /**
     * Prints a rich statistical summary of the collected distinct minima.
     */
    inline void print_minima_summary(
        const std::vector<SDMinimumRecord>& minima,
        size_t n_spins,
        long long total_starts,
        long long total_evals,
        double elapsed_seconds)
    {
        if (minima.empty()) {
            std::cout << "[SD Summary] No minima collected.\n";
            return;
        }

        long long min_e = minima[0].energy;
        long long max_e = minima[0].energy;
        double sum_e = 0.0;
        std::map<long long, size_t> hist;

        for (const auto& rec : minima) {
            min_e = std::min(min_e, rec.energy);
            max_e = std::max(max_e, rec.energy);
            sum_e += rec.energy;
            hist[rec.energy]++;
        }
        double mean_e = sum_e / minima.size();

        std::cout << "\n"
                  << "======================================================================\n"
                  << "           STEEPEST DESCENT MINIMA COLLECTION SUMMARY (N=" << n_spins << ")\n"
                  << "======================================================================\n"
                  << "  Distinct Symmetry Orbits : " << minima.size() << "\n"
                  << "  Total Random Starts Run  : " << total_starts << "\n"
                  << "  Unique Discovery Rate    : " << std::fixed << std::setprecision(2)
                  << (100.0 * minima.size() / std::max(1LL, total_starts)) << "%\n"
                  << "  Total Classical Evals    : " << total_evals << "\n"
                  << "  Mean Evals per Minimum   : " << std::fixed << std::setprecision(1)
                  << ((double)total_evals / minima.size()) << "\n"
                  << "  Search Elapsed Time      : " << std::fixed << std::setprecision(3)
                  << elapsed_seconds << " s\n"
                  << "----------------------------------------------------------------------\n"
                  << "  Energy Statistics:\n"
                  << "    Global Best in Set (Min) : " << min_e << "\n"
                  << "    Mean Energy              : " << std::fixed << std::setprecision(2) << mean_e << "\n"
                  << "    Maximum Energy           : " << max_e << "\n"
                  << "----------------------------------------------------------------------\n"
                  << "  Energy Distribution (Histogram):\n"
                  << "    " << std::setw(10) << "Energy" << " | " << std::setw(8) << "Count" << " | " << "Bar\n"
                  << "    -----------+----------+-------------------------------------------\n";

        for (const auto& [energy, count] : hist) {
            std::string bar(std::min((size_t)40, count * 40 / minima.size() + 1), '#');
            std::cout << "    " << std::setw(10) << energy << " | "
                      << std::setw(8) << count << " | "
                      << bar << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * count / minima.size()) << "%)\n";
        }

        std::cout << "----------------------------------------------------------------------\n"
                  << "  Sample of Best Discovered Minima (Lowest Energy):\n";

        // Sort by energy to show best ones
        std::vector<SDMinimumRecord> sorted_minima = minima;
        std::sort(sorted_minima.begin(), sorted_minima.end(), [](const SDMinimumRecord& a, const SDMinimumRecord& b) {
            return a.energy < b.energy;
        });

        size_t show_count = std::min((size_t)5, sorted_minima.size());
        for (size_t i = 0; i < show_count; ++i) {
            const auto& rec = sorted_minima[i];
            std::cout << "    [" << i + 1 << "] ID=" << rec.id
                      << " | Energy=" << rec.energy
                      << " | Evals=" << rec.evals_to_find
                      << " | StartIdx=" << rec.start_index;
            if (rec.state.bits.size() == 1) {
                std::cout << " | Bits=0x" << std::hex << rec.state.bits[0] << std::dec;
            }
            std::cout << "\n";
        }
        std::cout << "======================================================================\n\n";
    }

} // namespace SDLogging

namespace Statistics {

    struct BinomialStats {
        size_t trials = 0;
        size_t successes = 0;
        double rate = 0.0;
        double std_error = 0.0;
        double ci_lower = 0.0;
        double ci_upper = 0.0;
    };

    /**
     * Computes empirical success rate, binomial standard error,
     * and Wilson score 95% confidence interval bounds.
     */
    inline BinomialStats compute_binomial_stats(size_t successes, size_t trials, double z = 1.96) {
        BinomialStats stats;
        stats.trials = trials;
        stats.successes = successes;
        if (trials == 0) return stats;

        stats.rate = static_cast<double>(successes) / trials;
        stats.std_error = std::sqrt(stats.rate * (1.0 - stats.rate) / trials);

        // Wilson score confidence interval (recommended for binomial proportions)
        double z2 = z * z;
        double denom = 1.0 + z2 / trials;
        double center = (stats.rate + z2 / (2.0 * trials)) / denom;
        double spread = (z * std::sqrt((stats.rate * (1.0 - stats.rate) / trials) + (z2 / (4.0 * trials * trials)))) / denom;

        stats.ci_lower = std::max(0.0, center - spread);
        stats.ci_upper = std::min(1.0, center + spread);
        return stats;
    }

} // namespace Statistics

#endif // SPIN_SYMMETRY_HPP
