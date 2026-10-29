#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>

struct SpinState {
    uint64_t bits = 0;

    // Constructors
    constexpr SpinState() = default;
    constexpr explicit SpinState(uint64_t b) : bits(b) {}

    // 1. Bit & Spin access
    inline bool get_bit(size_t i) const {
        return (bits >> i) & 1ULL;
    }

    inline int get_spin(size_t i) const {
        // bit 0 -> +1, bit 1 -> -1 (or vice versa under global negation symmetry)
        return ((bits >> i) & 1ULL) ? -1 : 1;
    }

    // 2. Bit flips (in-place and copy)
    inline void flip(size_t i) {
        bits ^= (1ULL << i);
    }

    inline SpinState flipped(size_t i) const {
        return SpinState{bits ^ (1ULL << i)};
    }

    // 3. Hamming distance (XOR + popcount)
    inline int hamming_distance(const SpinState& other) const {
        return __builtin_popcountll(bits ^ other.bits);
    }

    // Fast 64-bit reverse of the lowest n bits
    static inline uint64_t reverse_bits(uint64_t x, int n) {
        x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
        x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
        x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0FULL) << 4);
        x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x & 0x00FF00FF00FF00FFULL) << 8);
        x = ((x >> 16) & 0x0000FFFF0000FFFFULL) | ((x & 0x0000FFFF0000FFFFULL) << 16);
        x = (x >> 32) | (x << 32);
        return x >> (64 - n);
    }

    // 4. Canonical representative under the order-8 symmetry group
    inline SpinState canonical(int n) const {
        const uint64_t mask = (n >= 64) ? ~0ULL : ((1ULL << n) - 1ULL);
        const uint64_t alt_mask = 0x5555555555555555ULL & mask;

        const uint64_t s0 = bits & mask;
        const uint64_t s_alt = s0 ^ alt_mask;

        // Base 4 orientations: [original, alternating, reversed, reversed-alternating]
        uint64_t c[8];
        c[0] = s0;
        c[1] = s_alt;
        c[2] = reverse_bits(s0, n);
        c[3] = reverse_bits(s_alt, n);

        // Global negations (T1) of all 4
        c[4] = (~c[0]) & mask;
        c[5] = (~c[1]) & mask;
        c[6] = (~c[2]) & mask;
        c[7] = (~c[3]) & mask;

        uint64_t min_bits = c[0];
        for (int i = 1; i < 8; ++i) {
            if (c[i] < min_bits) {
                min_bits = c[i];
            }
        }
        return SpinState{min_bits};
    }

    // 5. Comparators (for std::set, std::map)
    inline bool operator==(const SpinState& other) const { return bits == other.bits; }
    inline bool operator!=(const SpinState& other) const { return bits != other.bits; }
    inline bool operator<(const SpinState& other) const { return bits < other.bits; }
};

// 6. Hasher (for std::unordered_set, std::unordered_map)
struct SpinStateHash {
    inline size_t operator()(const SpinState& s) const noexcept {
        uint64_t x = s.bits;
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return static_cast<size_t>(x);
    }
};

namespace std {
    template <>
    struct hash<SpinState> {
        size_t operator()(const SpinState& s) const noexcept {
            return SpinStateHash{}(s);
        }
    };
}

class LabsEvaluator {
private:
    int n;
    uint64_t full_mask;

public:
    explicit LabsEvaluator(int system_size)
        : n(system_size),
          full_mask((system_size >= 64) ? ~0ULL : ((1ULL << system_size) - 1ULL)) {}

    int get_n() const { return n; }

    // Direct, zero-allocation energy calculation via bit-parallel Popcount (~20ns)
    int64_t calcEnergy(const SpinState& state) const {
        uint64_t b = state.bits & full_mask;
        int64_t total_energy = 0;

        for (int k = 1; k < n; ++k) {
            uint64_t mask_k = (1ULL << (n - k)) - 1ULL;
            int diffs = __builtin_popcountll((b ^ (b >> k)) & mask_k);
            int64_t c_k = static_cast<int64_t>(n - k) - 2 * diffs;
            total_energy += c_k * c_k;
        }
        return total_energy;
    }

    inline int64_t energy(const SpinState& state) const {
        return calcEnergy(state);
    }

    // Autocorrelation vector C_k calculation
    std::vector<int64_t> calcCk(const SpinState& state) const {
        uint64_t b = state.bits & full_mask;
        std::vector<int64_t> C(n, 0);

        for (int k = 1; k < n; ++k) {
            uint64_t mask_k = (1ULL << (n - k)) - 1ULL;
            int diffs = __builtin_popcountll((b ^ (b >> k)) & mask_k);
            C[k] = static_cast<int64_t>(n - k) - 2 * diffs;
        }
        return C;
    }

    // Canonical representative delegate
    SpinState get_canonical(const SpinState& state) const {
        return state.canonical(n);
    }

    /**
     * @brief Computes d_min: the smallest Hamming distance from s0 to any state with E(s) < target_energy.
     * Evaluates Hamming spheres d = 1, 2, 3, ... using bit-parallel popcount.
     * Halts immediately upon finding the first lower-energy state.
     */
    int compute_d_min(const SpinState& s0, int64_t target_energy = -1, int max_d = -1) const {
        if (target_energy < 0) {
            target_energy = calcEnergy(s0);
        }
        uint64_t s_bits = s0.bits & full_mask;
        int search_limit = (max_d > 0 && max_d <= n) ? max_d : n;

        for (int d = 1; d <= search_limit; ++d) {
            if (d == n) {
                SpinState candidate{s_bits ^ full_mask};
                if (calcEnergy(candidate) < target_energy) return n;
                break;
            }
            uint64_t mask = (1ULL << d) - 1ULL;
            const uint64_t limit = 1ULL << n;

            while (mask < limit && mask != 0) {
                SpinState candidate{s_bits ^ mask};
                if (calcEnergy(candidate) < target_energy) {
                    return d;
                }
                uint64_t c = mask & -mask;
                uint64_t r = mask + c;
                if (c == 0) break;
                mask = (((r ^ mask) >> 2) / c) | r;
            }
        }
        return (max_d > 0) ? (max_d + 1) : n;
    }
};