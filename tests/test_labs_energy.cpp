#include "../src/labs_energy/labs_energy.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <random>

// Test 1: Verify known values for N = 2..10 matching configuration bits = N
void test_known_config_energies() {
    std::cout << "[Test 1] Testing known energies for configs with bits = N (N=2..10)..." << std::endl;
    const int expected[] = {0, 0, 1, 1, 2, 6, 7, 35, 32, 24, 45};
    for (int n = 2; n <= 10; ++n) {
        LabsEvaluator eval(n);
        SpinState state(static_cast<uint64_t>(n));
        int64_t energy = eval.calcEnergy(state);
        std::cout << "  N = " << n << ", config = " << n 
                  << " -> energy = " << energy 
                  << " (expected = " << expected[n] << ")" << std::endl;
        assert(energy == expected[n]);
    }
    std::cout << "  --> PASSED: All reference energies matched perfectly!\n" << std::endl;
}

// Test 2: Verify famous Barker sequences (optimal aperiodic autocorrelations |C_k| <= 1)
void test_barker_sequences() {
    std::cout << "[Test 2] Testing optimal Barker sequences..." << std::endl;

    // Barker N=5: +1 +1 +1 -1 +1 -> in bits (0 for +1, 1 for -1): 0b01000 = 8
    // C = [0, 0, 1, 0, 1] -> E = 0^2 + 1^2 + 0^2 + 1^2 = 2
    {
        LabsEvaluator eval(5);
        // Bit 0, 1, 2 = 0 (+1), Bit 3 = 1 (-1), Bit 4 = 0 (+1) -> bits = (1 << 3) = 8
        SpinState s(8ULL);
        assert(eval.calcEnergy(s) == 2);
    }

    // Barker N=7: +1 +1 +1 -1 -1 +1 -1 -> bits at 3, 4, 6 are 1 -> 0b1011000 = 88
    // |C_k| <= 1, exactly three non-zero -> E = 3
    {
        LabsEvaluator eval(7);
        SpinState s((1ULL << 3) | (1ULL << 4) | (1ULL << 6));
        assert(eval.calcEnergy(s) == 3);
    }

    // Barker N=11: +1 +1 +1 -1 -1 -1 +1 -1 -1 +1 -1 -> bits at 3, 4, 5, 7, 8, 10
    // 5 non-zero C_k of magnitude 1 -> E = 5
    {
        LabsEvaluator eval(11);
        SpinState s((1ULL << 3) | (1ULL << 4) | (1ULL << 5) | (1ULL << 7) | (1ULL << 8) | (1ULL << 10));
        assert(eval.calcEnergy(s) == 5);
    }

    // Barker N=13: +1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 -> bits at 5, 6, 9, 11
    // 6 non-zero C_k of magnitude 1 -> E = 6
    {
        LabsEvaluator eval(13);
        SpinState s((1ULL << 5) | (1ULL << 6) | (1ULL << 9) | (1ULL << 11));
        assert(eval.calcEnergy(s) == 6);
    }

    std::cout << "  --> PASSED: Barker sequences N=5, 7, 11, 13 energies verified!\n" << std::endl;
}

// Test 3: Consistency between calcCk() sum of squares and calcEnergy()
void test_ck_sum_consistency() {
    std::cout << "[Test 3] Testing consistency between calcCk() and calcEnergy()..." << std::endl;
    std::mt19937_64 rng(1337);

    for (int n = 4; n <= 30; n += 2) {
        LabsEvaluator eval(n);
        uint64_t mask = (n >= 64) ? ~0ULL : ((1ULL << n) - 1ULL);

        for (int trial = 0; trial < 20; ++trial) {
            SpinState s(rng() & mask);
            int64_t energy = eval.calcEnergy(s);
            std::vector<int64_t> ck = eval.calcCk(s);

            assert(ck.size() == static_cast<size_t>(n));
            int64_t sum_squares = 0;
            for (size_t k = 1; k < ck.size(); ++k) {
                sum_squares += ck[k] * ck[k];
            }

            assert(energy == sum_squares);
        }
    }
    std::cout << "  --> PASSED: calcCk() squared sum exactly equals calcEnergy()!\n" << std::endl;
}

// Test 4: Symmetry invariance: E(s) == E(canonical(s)) across random states
void test_symmetry_energy_invariance() {
    std::cout << "[Test 4] Testing symmetry invariance of energy (E(s) == E(canonical(s)))..." << std::endl;
    std::mt19937_64 rng(42);

    for (int n : {8, 12, 16, 20, 24}) {
        LabsEvaluator eval(n);
        uint64_t mask = (n >= 64) ? ~0ULL : ((1ULL << n) - 1ULL);

        for (int trial = 0; trial < 50; ++trial) {
            SpinState s(rng() & mask);
            SpinState c = eval.get_canonical(s);

            int64_t e_orig = eval.calcEnergy(s);
            int64_t e_canon = eval.calcEnergy(c);
            assert(e_orig == e_canon);

            // Canonical must be idempotent: canonical(canonical(s)) == canonical(s)
            SpinState c2 = eval.get_canonical(c);
            assert(c == c2);
        }
    }
    std::cout << "  --> PASSED: Energy is strictly invariant under symmetry canonicalization!\n" << std::endl;
}

// Test 5: SpinState operations (bits, spins, flips, Hamming distance, hashing)
void test_spin_state_operations() {
    std::cout << "[Test 5] Testing SpinState bit manipulations and Hamming distance..." << std::endl;

    SpinState s(0ULL);
    // Initially all 0s -> get_bit(i) == 0, get_spin(i) == 1
    for (size_t i = 0; i < 64; ++i) {
        assert(!s.get_bit(i));
        assert(s.get_spin(i) == 1);
    }

    // In-place flip
    s.flip(3);
    assert(s.get_bit(3));
    assert(s.get_spin(3) == -1);
    assert(s.bits == (1ULL << 3));

    // Out-of-place flip
    SpinState s2 = s.flipped(5);
    assert(s2.get_bit(3) && s2.get_bit(5));
    assert(!s.get_bit(5));

    // Hamming distance
    assert(s.hamming_distance(s2) == 1);
    SpinState all_ones(~0ULL);
    assert(SpinState(0ULL).hamming_distance(all_ones) == 64);

    // Hasher consistency
    SpinStateHash hasher;
    assert(hasher(s) == hasher(SpinState(s.bits)));
    assert(s == SpinState(s.bits));

    std::cout << "  --> PASSED: SpinState bit manipulation methods verified!\n" << std::endl;
}

int main() {
    std::cout << "=====================================================\n";
    std::cout << "       Testing LabsEnergy & SpinState Library        \n";
    std::cout << "=====================================================\n";

    test_known_config_energies();
    test_barker_sequences();
    test_ck_sum_consistency();
    test_symmetry_energy_invariance();
    test_spin_state_operations();

    std::cout << "=====================================================\n";
    std::cout << "    ALL LABS ENERGY UNIT TESTS PASSED SUCCESSFULLY!  \n";
    std::cout << "=====================================================" << std::endl;
    return 0;
}