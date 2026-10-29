#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include "../labs_energy/labs_energy.hpp"

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <N> <state_hex_or_dec> <target_energy_E0>\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    std::string state_str = argv[2];
    uint64_t state_bits = 0;
    if (state_str.rfind("0x", 0) == 0 || state_str.rfind("0X", 0) == 0) {
        state_bits = std::stoull(state_str, nullptr, 16);
    } else {
        state_bits = std::stoull(state_str, nullptr, 10);
    }
    int64_t target_e = std::atoll(argv[3]);

    LabsEvaluator eval(n);
    int d_min = eval.compute_d_min(SpinState{state_bits}, target_e);
    std::cout << d_min << "\n";
    return 0;
}
