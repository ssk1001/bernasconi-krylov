#ifndef SPIN_CACHE_HPP
#define SPIN_CACHE_HPP

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <optional>

// Configuration representing the binary spin state (0 -> -1, 1 -> +1)
struct SpinState {
    size_t num_bits;                // Total number of spins (N)
    std::vector<uint64_t> bits;     // Bit-packed representation

    bool operator==(const SpinState& other) const {
        return num_bits == other.num_bits && bits == other.bits;
    }

    inline int get_spin(size_t index) const {
        size_t word_idx = index / 64;
        size_t bit_idx = index % 64;
        uint64_t bit = (bits[word_idx] >> bit_idx) & 1ULL;
        return bit ? 1 : -1;
    }

    // Creates a copy with the bit at `index` inverted
    SpinState flip_bit(size_t index) const {
        SpinState neighbor = *this;
        size_t word_idx = index / 64;
        size_t bit_idx = index % 64;
        neighbor.bits[word_idx] ^= (1ULL << bit_idx);
        return neighbor;
    }

};

// Custom Hash operator for SpinState
struct SpinStateHash {
    std::size_t operator()(const SpinState& state) const {
        std::size_t seed = state.num_bits;
        for (uint64_t chunk : state.bits) {
            seed ^= std::hash<uint64_t>{}(chunk) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

using SpinCache = std::unordered_map<SpinState, std::vector<int64_t>, SpinStateHash>;

namespace SpinLib {
    namespace detail {

        struct SpinNeighborHit {
            size_t flipped_bit_index;
            const SpinState* state_ptr;
            const std::vector<int64_t>* C_k_ptr;
        };

        // Full O(N^2) recalculation from scratch[cite: 1]
        inline std::vector<int64_t> calculate_C_k_scratch(const SpinState& state) {
            size_t N = state.num_bits;
            std::vector<int64_t> C(N, 0);

            for (size_t k = 1; k < N; ++k) {
                int64_t sum = 0;
                for (size_t i = 0; i < N - k; ++i) {
                    sum += state.get_spin(i) * state.get_spin(i + k);
                }
                C[k] = sum;
            }
            return C;
        }

        // O(N) spin flip update from a 1-flip neighbor[cite: 1]
        inline std::vector<int64_t> update_C_k_from_neighbor(
            const SpinState& target_state,
            const std::vector<int64_t>& neighbor_C_k,
            size_t flipped_bit_index)
        {
            size_t N = target_state.num_bits;
            std::vector<int64_t> C(N, 0);

            int s_m_prime = target_state.get_spin(flipped_bit_index); 

            for (size_t k = 1; k < N; ++k) {
                int left_neighbor = (flipped_bit_index >= k) 
                    ? target_state.get_spin(flipped_bit_index - k) : 0;
                int right_neighbor = (flipped_bit_index + k < N) 
                    ? target_state.get_spin(flipped_bit_index + k) : 0;

                C[k] = neighbor_C_k[k] + 2 * s_m_prime * (left_neighbor + right_neighbor);
            }

            return C;
        }

        // Calculates overall system energy E(s) = sum C_k(s)^2[cite: 1]
        inline int64_t calculate_energy(const std::vector<int64_t>& C_k) {
            int64_t energy = 0;
            for (size_t k = 1; k < C_k.size(); ++k) {
                energy += C_k[k] * C_k[k];
            }
            return energy;
        }

        // Searches cache for any 1-spin-flip neighbor
        inline std::optional<SpinNeighborHit> search_cached_neighbor(
            const SpinCache& cache, 
            const SpinState& state) 
        {
            for (size_t i = 0; i < state.num_bits; ++i) {
                SpinState neighbor = state.flip_bit(i);
                auto it = cache.find(neighbor);

                if (it != cache.end()) {
                    return SpinNeighborHit{
                        i,
                        &(it->first),
                        &(it->second)
                    };
                }
            }
            return std::nullopt;
        }

        // Pipeline: Exact Match -> Spin Flip Update -> Scratch Calculation
        inline const std::vector<int64_t>& fetch_or_compute_C_k(
            SpinCache& cache, 
            const SpinState& state) 
        {
            auto it = cache.find(state);
            if (it != cache.end()) {
                return it->second;
            }

            auto neighbor_hit = search_cached_neighbor(cache, state);
            std::vector<int64_t> C_k;

            if (neighbor_hit.has_value()) {
                C_k = update_C_k_from_neighbor(
                    state, 
                    *(neighbor_hit->C_k_ptr), 
                    neighbor_hit->flipped_bit_index
                );
            } else {
                C_k = calculate_C_k_scratch(state);
            }

            auto inserted = cache.emplace(state, std::move(C_k));
            return inserted.first->second;
        }

    } // namespace detail

    // Primary Public Interface
    inline int64_t get_energy(SpinCache& cache, const SpinState& state) {
        const auto& C_k = detail::fetch_or_compute_C_k(cache, state);
        return detail::calculate_energy(C_k);
    }


} // namespace SpinLib

#endif // SPIN_CACHE_HPP