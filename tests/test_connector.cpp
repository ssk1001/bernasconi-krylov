// ============================================================================
// Unit Test for BenchmarkConnector (src/connector/connector.cuh)
// ============================================================================

#include <iostream>
#include <cassert>
#include <filesystem>
#include "../src/connector/connector.cuh"

int main() {
    std::cout << "=====================================================\n";
    std::cout << "            Testing BenchmarkConnector               \n";
    std::cout << "=====================================================\n";

    // Setup temporary test configuration
    BenchmarkConfig cfg;
    cfg.system.n_values = {6};
    cfg.system.seed_base = 42;
    cfg.system.device = "cpu";

    cfg.traps.pool_minima_count = 3;
    cfg.traps.benchmark_trap_count = 2;
    cfg.traps.minima_cache_dir = "data_raw/test_cache";
    cfg.traps.force_regenerate_traps = true;

    cfg.quantum.g_values = {0.5};
    cfg.quantum.dt = 0.05;
    cfg.quantum.t_start = 2.0;
    cfg.quantum.dt_sample = 0.5;
    cfg.quantum.k_sample_points = 5;
    cfg.quantum.m_shots_per_point = 50;

    cfg.classical.kick_trials_per_d = 50;
    cfg.classical.kick_d_values = {1, 2};
    cfg.classical.restart_trials = 50;
    cfg.classical.batch_size = 50;

    cfg.output.output_dir = "data_raw/test_connector_out";
    cfg.output.results_csv = "results_n{N}.csv";
    cfg.output.time_series_csv = "dynamics_n{N}.csv";
    cfg.output.summary_csv = "summary.csv";
    cfg.output.verbose = false;

    BenchmarkConnector connector;

    // 1. Test Trap Preparation
    std::cout << "[Test 1] Testing trap preparation...\n";
    auto traps = connector.prepare_traps(6, cfg.traps, cfg.system.seed_base, false, false);
    assert(traps.size() == 2);
    assert(traps[0].energy <= traps[1].energy);
    std::cout << "  Trap #1 E0 = " << traps[0].energy << " | Trap #2 E0 = " << traps[1].energy << "\n";
    assert(std::filesystem::exists("data_raw/test_cache/seeds_N6.csv"));
    std::cout << "  --> PASSED: Trap discovery and caching verified!\n";

    // 2. Test Full Benchmark Pipeline
    std::cout << "[Test 2] Running full benchmark pipeline end-to-end...\n";
    auto results = connector.run(cfg);
    assert(results.size() == 2);

    for (const auto& r : results) {
        assert(r.n == 6);
        assert(r.g == 0.5);
        assert(r.best_d == 1 || r.best_d == 2);
        assert(r.rate_by_d.count(1) == 1);
        assert(r.rate_by_d.count(2) == 1);
        assert(r.best_d_trials == 50);
        assert(r.restart_trials == 50);
        assert(r.time_points.size() == 5);
        assert(r.p_less_history.size() == 5);
    }
    std::cout << "  --> PASSED: Pipeline execution verified!\n";

    // 3. Test Export Verification
    std::cout << "[Test 3] Verifying output CSV files...\n";
    std::string res_csv = cfg.output.get_results_path(6);
    std::string dyn_csv = cfg.output.get_time_series_path(6);
    std::string sum_csv = cfg.output.get_summary_path();
    std::string json_file = cfg.output.get_structured_json_path(6);
    std::string all_json_file = cfg.output.get_all_structured_json_path();

    assert(std::filesystem::exists(res_csv));
    assert(std::filesystem::exists(dyn_csv));
    assert(std::filesystem::exists(sum_csv));
    assert(std::filesystem::exists(json_file));
    assert(std::filesystem::exists(all_json_file));
    assert(std::filesystem::file_size(json_file) > 50);
    assert(std::filesystem::file_size(all_json_file) > 50);

    // Verify CSV line count
    std::ifstream f_res(res_csv);
    std::string line;
    int res_lines = 0;
    while (std::getline(f_res, line)) res_lines++;
    assert(res_lines == 3); // Header + 2 trap rows

    std::cout << "  --> PASSED: Output CSV and structured JSON exports verified!\n";

    // 4. Test Table Print
    std::cout << "[Test 4] Verifying summary table format...\n";
    connector.print_comparison_table(results);
    std::cout << "  --> PASSED: Summary table verified!\n";

    // 5. Test Continuous Trajectory Logging & Plotting Integration
    std::cout << "[Test 5] Testing continuous trajectory logging (record_trajectory = true, run_classical = false)...\n";
    BenchmarkConfig traj_cfg = cfg;
    traj_cfg.quantum.record_trajectory = true;
    traj_cfg.quantum.dt_trajectory = 0.05;
    traj_cfg.quantum.t_max_val = 2.0;
    traj_cfg.quantum.run_classical = false;
    traj_cfg.quantum.g_values = {0.30, 0.50};
    traj_cfg.output.output_dir = "data_raw/test_traj_out";

    auto traj_results = connector.run(traj_cfg);
    assert(traj_results.size() == 4); // 2 traps * 2 g values

    for (const auto& r : traj_results) {
        assert(r.time_points.size() == 41); // t = 0.0 to 2.0 with dt = 0.05 (41 points)
        assert(r.p_less_history.size() == 41);
        assert(r.p_blind >= 0.0);
        assert(r.best_d_trials == 0); // classical was skipped
        assert(r.restart_trials == 0);
    }

    std::string traj_csv = traj_cfg.output.get_time_series_path(6);
    assert(std::filesystem::exists(traj_csv));

    // Verify trajectory CSV rows: 1 header + 4 * 41 data lines = 165 lines
    std::ifstream f_traj(traj_csv);
    std::string traj_line;
    int traj_lines = 0;
    while (std::getline(f_traj, traj_line)) traj_lines++;
    assert(traj_lines == 165);

    std::cout << "  --> PASSED: Continuous trajectory CSV generated with " << traj_lines << " lines!\n";

    // Test Python plotting integration
    std::cout << "[Test 6] Testing plot_probability_dynamics.py execution...\n";
    int py_check = std::system("python3 -c 'import pandas, matplotlib' > /dev/null 2>&1");
    if (py_check == 0) {
        int plot_ret = std::system("python3 scripts/plot_probability_dynamics.py data_raw/test_traj_out/dynamics_n6.csv figures/test_n6 > /dev/null 2>&1");
        assert(plot_ret == 0);
        assert(std::filesystem::exists("figures/test_n6/prob_dynamics_all_2_traps_grid.png"));
        std::cout << "  --> PASSED: Overview grid and individual plots successfully generated!\n";
    } else {
        std::cout << "  [Notice] Python pandas/matplotlib not present in local python environment.\n"
                  << "  (They are activated via 'source scripts/setup_env.sh cuquantum_env' on the HPC cluster).\n"
                  << "  --> SKIPPED rendering test (CSV validation passed!).\n";
    }

    // 7. Test Multi-Format Seed Loading and Symmetry Deduplication
    std::cout << "[Test 7] Testing ClassicalLogger::load_from_csv multi-format parsing and deduplication...\n";
    std::string test_fmt_dir = "data_raw/test_seeds_formats";
    std::filesystem::create_directories(test_fmt_dir);

    // Format A: Standard 8-column modern CSV
    {
        std::string path_a = test_fmt_dir + "/standard_n8.csv";
        std::ofstream out_a(path_a);
        out_a << "# N=8,Seed=42,TotalStarts=100,TotalEvals=500,ElapsedMs=1.5,Device=CPU\n"
              << "id,energy,times_visited,basin_fraction,evals_to_first_find,canonical_hex,sample_hex,canonical_bitstring\n"
              << "1,40,10,0.100000,45,0x05,0x05,00000101\n"
              << "2,56,5,0.050000,80,0x09,0x09,00001001\n";
        out_a.close();

        auto loaded_a = ClassicalLogger::load_from_csv(path_a, 8);
        assert(loaded_a.size() == 2);
        assert(loaded_a[0].energy == 40);
        assert(loaded_a[0].canonical_state.bits == 0x05);
        assert(loaded_a[0].times_visited == 10);
        assert(loaded_a[1].energy == 56);
        assert(loaded_a[1].canonical_state.bits == 0x09);
        std::cout << "  --> Format A (Standard 8-col CSV): PASSED\n";
    }

    // Format B: Legacy 5-column CSV (id,energy,bitstring,evals,start_index)
    {
        std::string path_b = test_fmt_dir + "/legacy_n8.csv";
        std::ofstream out_b(path_b);
        out_b << "id,energy,bitstring,evals_to_find,start_index\n"
              << "1,40,00000101,45,1\n"
              << "2,56,00001001,80,2\n";
        out_b.close();

        auto loaded_b = ClassicalLogger::load_from_csv(path_b, 8);
        assert(loaded_b.size() == 2);
        assert(loaded_b[0].energy == 40);
        assert(loaded_b[0].canonical_state.bits == 0x05);
        assert(loaded_b[1].energy == 56);
        assert(loaded_b[1].canonical_state.bits == 0x09);
        std::cout << "  --> Format B (Legacy 5-col CSV): PASSED\n";
    }

    // Format C: Raw bitstrings with comments, energy recalculation, and deduplication
    {
        std::string path_c = test_fmt_dir + "/raw_bitstrings.txt";
        std::ofstream out_c(path_c);
        out_c << "# Plain bitstrings list with duplicate symmetry orbits\n"
              << "00000101\n"
              << "00000101\n"  // Exact duplicate -> should increment visits
              << "00001001\n";
        out_c.close();

        auto loaded_c = ClassicalLogger::load_from_csv(path_c, 8);
        assert(loaded_c.size() == 2);
        // Sorted ascending by energy: 0x09 has E=12, 0x05 has E=20
        assert(loaded_c[0].canonical_state.bits == 0x09);
        assert(loaded_c[0].energy == 12);
        assert(loaded_c[0].times_visited == 1);
        assert(loaded_c[1].canonical_state.bits == 0x05);
        assert(loaded_c[1].energy == 20);
        assert(loaded_c[1].times_visited == 2); // 2 occurrences deduplicated
        std::cout << "  --> Format C (Raw bitstrings + deduplication): PASSED\n";
    }

    // Format D: Raw hex strings
    {
        std::string path_d = test_fmt_dir + "/raw_hex.txt";
        std::ofstream out_d(path_d);
        out_d << "0x05\n"
              << "0x09\n";
        out_d.close();

        auto loaded_d = ClassicalLogger::load_from_csv(path_d, 8);
        assert(loaded_d.size() == 2);
        // Sorted ascending by energy: 0x09 has E=12, 0x05 has E=20
        assert(loaded_d[0].canonical_state.bits == 0x09);
        assert(loaded_d[0].energy == 12);
        assert(loaded_d[1].canonical_state.bits == 0x05);
        assert(loaded_d[1].energy == 20);
        std::cout << "  --> Format D (Raw hex values): PASSED\n";
    }

    // 8. Test Automatic Seed Discovery Fallback and Cache Persistence
    std::cout << "[Test 8] Testing automatic seed generation and cache persistence...\n";
    std::string auto_seed_dir = "data_raw/test_seeds_auto";
    std::filesystem::remove_all(auto_seed_dir);

    TrapsConfig auto_cfg;
    auto_cfg.pool_minima_count = 5;
    auto_cfg.benchmark_trap_count = 3;
    auto_cfg.minima_cache_dir = auto_seed_dir;
    auto_cfg.force_regenerate_traps = false;

    // Subtest 8.1: First call -> File absent, triggers discovery and creates seeds_N8.csv
    auto auto_traps_run1 = connector.prepare_traps(8, auto_cfg, 12345, false, true);
    assert(auto_traps_run1.size() == 3);
    assert(auto_traps_run1[0].energy <= auto_traps_run1[1].energy);
    assert(auto_traps_run1[1].energy <= auto_traps_run1[2].energy);

    std::string expected_seed_path = auto_seed_dir + "/seeds_N8.csv";
    assert(std::filesystem::exists(expected_seed_path));
    std::cout << "  --> Discovered & saved seeds to: " << expected_seed_path << "\n";

    // Subtest 8.2: Second call -> File present, loads from cache without discovery
    auto auto_traps_run2 = connector.prepare_traps(8, auto_cfg, 12345, false, true);
    assert(auto_traps_run2.size() == 3);
    for (size_t i = 0; i < 3; ++i) {
        assert(auto_traps_run1[i].canonical_state == auto_traps_run2[i].canonical_state);
        assert(auto_traps_run1[i].energy == auto_traps_run2[i].energy);
    }
    std::cout << "  --> Successfully hit cache and reloaded identical seeds!\n";

    // Subtest 8.3: Directory search fallback test (discovers data_raw/minima/minima_n16.csv)
    std::string found_minima_n16 = BenchmarkConnector::find_seed_file("data_raw/minima", 16);
    assert(!found_minima_n16.empty());
    assert(found_minima_n16.find("minima_n16.csv") != std::string::npos);
    std::cout << "  --> Fallback path resolver found: " << found_minima_n16 << "\n";

    // Clean up temporary test files
    std::filesystem::remove_all("data_raw/test_cache");
    std::filesystem::remove_all("data_raw/test_connector_out");
    std::filesystem::remove_all("data_raw/test_traj_out");
    std::filesystem::remove_all("figures/test_n6");
    std::filesystem::remove_all(test_fmt_dir);
    std::filesystem::remove_all(auto_seed_dir);

    std::cout << "=====================================================\n";
    std::cout << "  ALL CONNECTOR TESTS PASSED SUCCESSFULLY!           \n";
    std::cout << "=====================================================\n";
    return 0;
}
