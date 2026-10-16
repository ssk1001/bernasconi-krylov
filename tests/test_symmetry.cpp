#include <iostream>
#include <cassert>
#include <random>
#include <unordered_set>
#include "../src/labs_energy/labs_energy.hpp"
#include "../src/classical/symmetry.hpp"

void test_symmetry_energy_invariance() {
    std::cout << "[Test 1] Testing symmetry energy invariance across random configurations...\n";
    SpinCache cache;
    std::mt19937_64 rng(12345);

    for (size_t n : {8, 12, 16, 20, 24}) {
        uint64_t mask = (1ULL << n) - 1;
        for (int trial = 0; trial < 100; ++trial) {
            SpinState s;
            s.num_bits = n;
            s.bits = {rng() & mask};

            long long e_orig = SpinLib::get_energy(cache, s);

            SpinState s_neg = SpinSymmetry::negate(s);
            SpinState s_rev = SpinSymmetry::reverse(s);
            SpinState s_alt = SpinSymmetry::alternating_flip(s);

            long long e_neg = SpinLib::get_energy(cache, s_neg);
            long long e_rev = SpinLib::get_energy(cache, s_rev);
            long long e_alt = SpinLib::get_energy(cache, s_alt);

            assert(e_neg == e_orig && "Negation must preserve LABS energy");
            assert(e_rev == e_orig && "Reversal must preserve LABS energy");
            assert(e_alt == e_orig && "Alternating flip must preserve LABS energy");

            // Test entire 8-state orbit
            auto orbit = SpinSymmetry::get_symmetry_orbit(s);
            for (const auto& member : orbit) {
                long long e_member = SpinLib::get_energy(cache, member);
                assert(e_member == e_orig && "Every orbit member must have identical energy");
            }
        }
    }
    std::cout << "  --> PASSED: Energy invariance verified for all 3 symmetry operations!\n";
}

void test_canonical_uniqueness() {
    std::cout << "[Test 2] Testing orbit canonicalization consistency...\n";
    std::mt19937_64 rng(67890);

    for (size_t n : {8, 16, 20}) {
        uint64_t mask = (1ULL << n) - 1;
        for (int trial = 0; trial < 50; ++trial) {
            SpinState s;
            s.num_bits = n;
            s.bits = {rng() & mask};

            SpinState canon_s = SpinSymmetry::get_canonical_state(s);

            // For every state in the orbit, its canonical representative MUST be identical to canon_s
            auto orbit = SpinSymmetry::get_symmetry_orbit(s);
            for (const auto& member : orbit) {
                SpinState canon_member = SpinSymmetry::get_canonical_state(member);
                assert(canon_member == canon_s && "All orbit members must yield identical canonical state");
                assert(SpinSymmetry::are_symmetry_equivalent(s, member) && "Orbit members must be equivalent");
            }
        }
    }
    std::cout << "  --> PASSED: Canonical consistency verified!\n";
}

void test_deduplication_reduction() {
    std::cout << "[Test 3] Testing orbit sizes and deduplication count...\n";
    // For N = 6, there are 2^6 = 64 total states.
    // Let's count how many distinct symmetry classes exist in the entire 2^6 space!
    size_t n = 6;
    std::unordered_set<SpinState, SpinStateHash> unique_canonical;
    SpinCache cache;

    for (uint64_t i = 0; i < (1ULL << n); ++i) {
        SpinState s{n, {i}};
        unique_canonical.insert(SpinSymmetry::get_canonical_state(s));
    }

    std::cout << "  For N=6: Total 2^6 = 64 states reduced to " 
              << unique_canonical.size() << " symmetry-distinct classes.\n";
    // A reduction by a factor of ~4 to 8 is expected!
    assert(unique_canonical.size() <= 64 / 4);
    std::cout << "  --> PASSED: Substantial reduction confirmed!\n";
}

void test_binomial_statistics() {
    std::cout << "[Test 4] Testing binomial error bars and Wilson confidence intervals...\n";
    // 80 successes out of 100 trials
    auto stats = Statistics::compute_binomial_stats(80, 100);
    assert(std::abs(stats.rate - 0.80) < 1e-9);
    // SE should be sqrt(0.8 * 0.2 / 100) = sqrt(0.0016) = 0.04
    assert(std::abs(stats.std_error - 0.04) < 1e-6);
    // Wilson CI should contain 0.80 and have valid bounds
    assert(stats.ci_lower < 0.80 && stats.ci_upper > 0.80);
    assert(stats.ci_lower >= 0.0 && stats.ci_upper <= 1.0);

    // Boundary cases: 0 / 100 and 100 / 100
    auto zero_stats = Statistics::compute_binomial_stats(0, 100);
    assert(std::abs(zero_stats.rate - 0.0) < 1e-9 && zero_stats.ci_lower >= 0.0 && zero_stats.ci_upper > 0.0);
    auto one_stats = Statistics::compute_binomial_stats(100, 100);
    assert(std::abs(one_stats.rate - 1.0) < 1e-9 && std::abs(one_stats.ci_upper - 1.0) < 1e-6 && one_stats.ci_lower < 1.0);

    std::cout << "  --> PASSED: Binomial error bars and Wilson CI verified!\n";
}

int main() {
    std::cout << "=========================================\n";
    std::cout << "       Testing SpinSymmetry Module       \n";
    std::cout << "=========================================\n";
    test_symmetry_energy_invariance();
    test_canonical_uniqueness();
    test_deduplication_reduction();
    test_binomial_statistics();
    std::cout << "\nALL SYMMETRY & STATISTICAL TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
