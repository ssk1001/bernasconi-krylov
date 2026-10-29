// ============================================================================
// Bernasconi-Krylov Modern Benchmark Executor
// ============================================================================

#include <iostream>
#include <string>
#include "connector/connector.cuh"

int main(int argc, char* argv[]) {
    std::string config_file = "configs/benchmark.ini";
    if (argc > 1) {
        config_file = argv[1];
    }

    std::cout << "======================================================================\n";
    std::cout << "               Bernasconi-Krylov Benchmark Pipeline                   \n";
    std::cout << "======================================================================\n";
    std::cout << "Configuration File: " << config_file << "\n";

    try {
        BenchmarkConnector connector;
        connector.run(config_file);
    } catch (const std::exception& e) {
        std::cerr << "\n[Fatal Benchmark Error] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
