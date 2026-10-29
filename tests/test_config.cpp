// ============================================================================
// Modern Unit Tests for Configuration Parser (src/config/config.hpp)
// ============================================================================

#include <iostream>
#include <cassert>
#include <cmath>
#include "../src/config/config.hpp"

void test_default_config() {
    std::cout << "[Test 1] Testing default configuration values...\n";
    BenchmarkConfig cfg;

    // System
    assert(cfg.system.n_values.size() == 3);
    assert(cfg.system.n_values[0] == 16 && cfg.system.n_values[1] == 20 && cfg.system.n_values[2] == 24);
    assert(cfg.system.seed_base == 42);
    assert(cfg.system.device == "auto");

    // Traps
    assert(cfg.traps.pool_minima_count == 100);
    assert(cfg.traps.benchmark_trap_count == 5);
    assert(cfg.traps.minima_cache_dir == "seeds");
    assert(!cfg.traps.force_regenerate_traps);

    // Quantum
    assert(cfg.quantum.g_values.size() == 3);
    assert(std::abs(cfg.quantum.g_values[0] - 0.25) < 1e-9);
    assert(std::abs(cfg.quantum.g_values[1] - 0.5) < 1e-9);
    assert(std::abs(cfg.quantum.g_values[2] - 0.75) < 1e-9);
    assert(std::abs(cfg.quantum.dt - 0.05) < 1e-9);
    assert(std::abs(cfg.quantum.t_start - 10.0) < 1e-9);
    assert(std::abs(cfg.quantum.dt_sample - 1.5) < 1e-9);
    assert(cfg.quantum.k_sample_points == 20);
    assert(cfg.quantum.m_shots_per_point == 100);
    assert(cfg.quantum.track_p_less);
    assert(!cfg.quantum.record_trajectory);
    assert(std::abs(cfg.quantum.dt_trajectory - 0.0) < 1e-9);
    assert(cfg.quantum.run_classical);

    // Classical
    assert(cfg.classical.kick_trials_per_d == 50000);
    assert(cfg.classical.kick_d_values.size() == 6);
    assert(cfg.classical.kick_d_values[0] == 1 && cfg.classical.kick_d_values[5] == 6);
    assert(cfg.classical.restart_trials == 50000);
    assert(cfg.classical.max_evals_per_start == 0);
    assert(cfg.classical.batch_size == 10000);

    // Output
    assert(cfg.output.output_dir == "data_raw");
    assert(cfg.output.results_csv == "benchmark_results_n{N}.csv");
    assert(cfg.output.time_series_csv == "p_less_dynamics_n{N}.csv");
    assert(cfg.output.summary_csv == "benchmark_summary.csv");
    assert(cfg.output.verbose);

    std::cout << "  --> PASSED: Default configuration verified!\n";
}

void test_derived_quantum_helpers() {
    std::cout << "[Test 2] Testing derived quantum helpers...\n";
    BenchmarkConfig cfg;
    cfg.quantum.t_start = 10.0;
    cfg.quantum.dt_sample = 1.5;
    cfg.quantum.k_sample_points = 20;
    cfg.quantum.dt = 0.05;

    // t_max = 10.0 + (20 - 1) * 1.5 = 10.0 + 28.5 = 38.5
    assert(std::abs(cfg.quantum.t_max() - 38.5) < 1e-9);

    auto times = cfg.quantum.sample_times();
    assert(times.size() == 20);
    assert(std::abs(times[0] - 10.0) < 1e-9);
    assert(std::abs(times[1] - 11.5) < 1e-9);
    assert(std::abs(times[19] - 38.5) < 1e-9);

    // 1.5 / 0.05 = 30 Trotter steps
    assert(cfg.quantum.trotter_steps_per_sample() == 30);

    std::cout << "  --> PASSED: Derived quantum helpers verified!\n";
}

void test_path_resolution() {
    std::cout << "[Test 3] Testing path template resolution...\n";
    BenchmarkConfig cfg;
    cfg.output.output_dir = "data_raw";
    cfg.output.results_csv = "results_n{N}.csv";
    cfg.output.time_series_csv = "dynamics_n{n}.csv";
    cfg.output.summary_csv = "summary.csv";

    assert(cfg.output.get_results_path(24) == "data_raw/results_n24.csv");
    assert(cfg.output.get_time_series_path(24) == "data_raw/dynamics_n24.csv");
    assert(cfg.output.get_summary_path() == "data_raw/summary.csv");

    // Test absolute path resolution
    cfg.output.results_csv = "/tmp/results_n{N}.csv";
    assert(cfg.output.get_results_path(30) == "/tmp/results_n30.csv");

    std::cout << "  --> PASSED: Path template resolution verified!\n";
}

void test_ini_string_parsing() {
    std::cout << "[Test 4] Testing INI string parsing with comments...\n";
    std::string ini_content = R"(
# Benchmark experiment definition
[system]
n_values = 20, 24, 28 ; System sizes
seed_base = 12345
device = gpu

[traps]
pool_minima_count = 200
benchmark_trap_count = 10
minima_cache_dir = data_raw/custom_minima
force_regenerate_traps = true

[quantum]
g_values = 0.35, 0.50
dt = 0.02
t_start = 12.0
dt_sample = 2.0
k_sample_points = 15
m_shots_per_point = 250
track_p_less = true

[classical]
kick_trials_per_d = 100000
kick_d_values = 2, 4, 6, 8
restart_trials = 80000
max_evals_per_start = 500
batch_size = 20000

[output]
output_dir = data_raw/runs
results_csv = run_n{N}.csv
time_series_csv = p_n{N}.csv
summary_csv = all_summary.csv
verbose = false
)";

    BenchmarkConfig cfg = BenchmarkConfig::load_from_string(ini_content);

    assert(cfg.system.n_values.size() == 3);
    assert(cfg.system.n_values[0] == 20 && cfg.system.n_values[2] == 28);
    assert(cfg.system.seed_base == 12345);
    assert(cfg.system.device == "gpu");

    assert(cfg.traps.pool_minima_count == 200);
    assert(cfg.traps.benchmark_trap_count == 10);
    assert(cfg.traps.minima_cache_dir == "data_raw/custom_minima");
    assert(cfg.traps.force_regenerate_traps);

    assert(cfg.quantum.g_values.size() == 2);
    assert(std::abs(cfg.quantum.g_values[0] - 0.35) < 1e-9);
    assert(std::abs(cfg.quantum.dt - 0.02) < 1e-9);
    assert(std::abs(cfg.quantum.t_start - 12.0) < 1e-9);
    assert(std::abs(cfg.quantum.dt_sample - 2.0) < 1e-9);
    assert(cfg.quantum.k_sample_points == 15);
    assert(cfg.quantum.m_shots_per_point == 250);
    assert(cfg.quantum.track_p_less);

    assert(cfg.classical.kick_trials_per_d == 100000);
    assert(cfg.classical.kick_d_values.size() == 4);
    assert(cfg.classical.kick_d_values[3] == 8);
    assert(cfg.classical.restart_trials == 80000);
    assert(cfg.classical.max_evals_per_start == 500);
    assert(cfg.classical.batch_size == 20000);

    assert(cfg.output.output_dir == "data_raw/runs");
    assert(cfg.output.get_results_path(24) == "data_raw/runs/run_n24.csv");
    assert(!cfg.output.verbose);

    std::cout << "  --> PASSED: INI string parsing verified!\n";
}

void test_legacy_config_backward_compatibility() {
    std::cout << "[Test 5] Testing backward compatibility with legacy .cfg files...\n";

    // 1. configs/n24_delayed_window.cfg
    {
        BenchmarkConfig cfg = BenchmarkConfig::load_from_file("configs/n24_delayed_window.cfg");
        assert(cfg.system.n_values.size() == 1 && cfg.system.n_values[0] == 24);
        assert(cfg.traps.pool_minima_count == 200);
        assert(cfg.traps.benchmark_trap_count == 10);
        assert(cfg.quantum.g_values.size() == 5);
        assert(std::abs(cfg.quantum.t_start - 10.0) < 1e-9);
        assert(std::abs(cfg.quantum.dt_sample - 1.5) < 1e-9);
        assert(std::abs(cfg.quantum.dt - 0.05) < 1e-9);
        assert(cfg.quantum.k_sample_points == 20);
        assert(cfg.quantum.m_shots_per_point == 100);
        assert(cfg.classical.kick_d_values.size() == 6);
        assert(cfg.classical.kick_trials_per_d == 50000);
        assert(cfg.classical.restart_trials == 50000);
        std::cout << "  --> PASSED: configs/n24_delayed_window.cfg compatibility verified!\n";
    }

    // 2. configs/default_benchmark.cfg
    {
        BenchmarkConfig cfg = BenchmarkConfig::load_from_file("configs/default_benchmark.cfg");
        assert(cfg.system.n_values.size() == 3);
        assert(cfg.system.n_values[0] == 16 && cfg.system.n_values[2] == 24);
        assert(cfg.traps.pool_minima_count == 100);
        assert(cfg.traps.benchmark_trap_count == 5);
        assert(cfg.quantum.g_values.size() == 1 && std::abs(cfg.quantum.g_values[0] - 0.5) < 1e-9);
        std::cout << "  --> PASSED: configs/default_benchmark.cfg compatibility verified!\n";
    }
}

void test_round_trip_serialization() {
    std::cout << "[Test 6] Testing round-trip INI serialization and re-parsing...\n";
    BenchmarkConfig cfg_orig;
    cfg_orig.system.n_values = {18, 22, 26};
    cfg_orig.system.seed_base = 99999;
    cfg_orig.system.device = "cpu";
    cfg_orig.traps.pool_minima_count = 150;
    cfg_orig.traps.benchmark_trap_count = 7;
    cfg_orig.quantum.g_values = {0.15, 0.45};
    cfg_orig.quantum.dt = 0.025;
    cfg_orig.quantum.t_start = 8.0;
    cfg_orig.quantum.dt_sample = 1.0;
    cfg_orig.quantum.k_sample_points = 25;
    cfg_orig.quantum.m_shots_per_point = 500;
    cfg_orig.classical.kick_trials_per_d = 75000;
    cfg_orig.classical.kick_d_values = {1, 3, 5};
    cfg_orig.classical.restart_trials = 75000;
    cfg_orig.output.output_dir = "data_raw/test_out";

    std::string ini_str = cfg_orig.to_ini_string();
    BenchmarkConfig cfg_parsed = BenchmarkConfig::load_from_string(ini_str);

    assert(cfg_parsed.system.n_values == cfg_orig.system.n_values);
    assert(cfg_parsed.system.seed_base == cfg_orig.system.seed_base);
    assert(cfg_parsed.system.device == cfg_orig.system.device);
    assert(cfg_parsed.traps.pool_minima_count == cfg_orig.traps.pool_minima_count);
    assert(cfg_parsed.traps.benchmark_trap_count == cfg_orig.traps.benchmark_trap_count);
    assert(cfg_parsed.quantum.g_values == cfg_orig.quantum.g_values);
    assert(std::abs(cfg_parsed.quantum.dt - cfg_orig.quantum.dt) < 1e-9);
    assert(std::abs(cfg_parsed.quantum.t_start - cfg_orig.quantum.t_start) < 1e-9);
    assert(std::abs(cfg_parsed.quantum.dt_sample - cfg_orig.quantum.dt_sample) < 1e-9);
    assert(cfg_parsed.quantum.k_sample_points == cfg_orig.quantum.k_sample_points);
    assert(cfg_parsed.quantum.m_shots_per_point == cfg_orig.quantum.m_shots_per_point);
    assert(cfg_parsed.classical.kick_trials_per_d == cfg_orig.classical.kick_trials_per_d);
    assert(cfg_parsed.classical.kick_d_values == cfg_orig.classical.kick_d_values);
    assert(cfg_parsed.classical.restart_trials == cfg_orig.classical.restart_trials);
    assert(cfg_parsed.output.output_dir == cfg_orig.output.output_dir);

    std::cout << "  --> PASSED: Round-trip INI serialization verified!\n";
}

void test_trajectory_config_flags() {
    std::cout << "[Test 7] Testing trajectory and classical execution config flags...\n";
    std::string ini_content = R"(
[quantum]
record_trajectory = true
dt_trajectory = 0.1
t_max = 60.0
run_classical = false
)";
    BenchmarkConfig cfg = BenchmarkConfig::load_from_string(ini_content);
    assert(cfg.quantum.record_trajectory);
    assert(std::abs(cfg.quantum.dt_trajectory - 0.1) < 1e-9);
    assert(std::abs(cfg.quantum.t_max() - 60.0) < 1e-9);
    assert(!cfg.quantum.run_classical);

    auto times = cfg.quantum.sample_times();
    assert(times.size() == 601);
    assert(std::abs(times[0] - 0.0) < 1e-9);
    assert(std::abs(times[600] - 60.0) < 1e-9);

    std::cout << "  --> PASSED: Trajectory config flags verified!\n";
}

int main() {
    std::cout << "=====================================================\n";
    std::cout << "       Testing Modern Benchmark Config Parser        \n";
    std::cout << "=====================================================\n";

    test_default_config();
    test_derived_quantum_helpers();
    test_path_resolution();
    test_ini_string_parsing();
    test_legacy_config_backward_compatibility();
    test_round_trip_serialization();
    test_trajectory_config_flags();

    std::cout << "=====================================================\n";
    std::cout << "  ALL CONFIG PARSER TESTS PASSED SUCCESSFULLY!       \n";
    std::cout << "=====================================================\n";
    return 0;
}
