// ============================================================================
// Unit Test for d_min computation (src/labs_energy/labs_energy.hpp)
// ============================================================================

#include <iostream>
#include <cassert>
#include <vector>
#include <fstream>
#include <sstream>
#include "../src/labs_energy/labs_energy.hpp"

int main() {
    std::cout << "=====================================================\n";
    std::cout << "             Testing compute_d_min                   \n";
    std::cout << "=====================================================\n";

    // 1. Test for small N=6
    {
        int n = 6;
        LabsEvaluator eval(n);
        // Test all states
        int total_states = 1 << n;
        for (int s = 0; s < total_states; ++s) {
            SpinState s0{static_cast<uint64_t>(s)};
            int64_t e0 = eval.calcEnergy(s0);

            // Compute ground truth by brute force
            int expected_d_min = n;
            for (int other = 0; other < total_states; ++other) {
                SpinState s_other{static_cast<uint64_t>(other)};
                if (eval.calcEnergy(s_other) < e0) {
                    int d = __builtin_popcount(s ^ other);
                    if (d < expected_d_min) expected_d_min = d;
                }
            }

            int actual_d_min = eval.compute_d_min(s0, e0);
            assert(actual_d_min == expected_d_min);
        }
        std::cout << "  --> PASSED: Verified all 2^6 = 64 states against exhaustive search!\n";
    }

    // 2. Test for N=10 against exhaustive states
    {
        int n = 10;
        LabsEvaluator eval(n);
        int total_states = 1 << n;
        std::vector<int64_t> energies(total_states);
        for (int s = 0; s < total_states; ++s) {
            energies[s] = eval.calcEnergy(SpinState{static_cast<uint64_t>(s)});
        }

        // Test 50 random states
        for (int step = 0; step < 50; ++step) {
            int s = (step * 37 + 13) % total_states;
            SpinState s0{static_cast<uint64_t>(s)};
            int64_t e0 = energies[s];

            int expected_d_min = n;
            for (int other = 0; other < total_states; ++other) {
                if (energies[other] < e0) {
                    int d = __builtin_popcount(s ^ other);
                    if (d < expected_d_min) expected_d_min = d;
                }
            }

            int actual_d_min = eval.compute_d_min(s0, e0);
            assert(actual_d_min == expected_d_min);
        }
        std::cout << "  --> PASSED: Verified N=10 sampled states against exhaustive search!\n";
    }

    // 3. Test for N=16 genuine local minima
    {
        int n = 16;
        LabsEvaluator eval(n);
        // Find 10 genuine local minima via steepest descent from different starts
        for (int seed_idx = 0; seed_idx < 10; ++seed_idx) {
            uint64_t start_state = (0x1234ULL * (seed_idx + 1) + 0xABCDULL) & ((1ULL << n) - 1ULL);
            SpinState cur{start_state};
            int64_t cur_e = eval.calcEnergy(cur);

            // Simple steepest descent to reach a genuine local minimum
            while (true) {
                SpinState best_nbr = cur;
                int64_t best_e = cur_e;
                for (int i = 0; i < n; ++i) {
                    SpinState nbr = cur.flipped(i);
                    int64_t e = eval.calcEnergy(nbr);
                    if (e < best_e) {
                        best_e = e;
                        best_nbr = nbr;
                    }
                }
                if (best_e < cur_e) {
                    cur = best_nbr;
                    cur_e = best_e;
                } else {
                    break;
                }
            }

            int d_min = eval.compute_d_min(cur, cur_e);
            // If it is not the global optimum (E* = 24 for N=16), d_min must be >= 2!
            if (cur_e > 24) {
                assert(d_min >= 2);
            }
            std::cout << "  Local Min #" << (seed_idx + 1) 
                      << ": 0x" << std::hex << cur.bits << std::dec 
                      << " (E0 = " << cur_e << ") -> d_min = " << d_min << "\n";
        }
        std::cout << "  --> PASSED: Verified N=16 genuine local minima have d_min >= 2!\n";
    }

    std::cout << "=====================================================\n";
    std::cout << "        ALL d_min UNIT TESTS PASSED!                 \n";
    std::cout << "=====================================================\n";
    return 0;
}
