#include <iostream>
#include <cassert>
#include <cmath>
#include "../src/benchmark_config.hpp"

int main() {
    std::cout << "Testing BenchmarkConfig parser...\n";

    // 1. Test loading default_benchmark.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/default_benchmark.cfg");
        assert(cfg.target_minima_count == 100);
        assert(cfg.pool_minima_count == 100);
        assert(cfg.top_minima_count == 5);
        assert(cfg.num_sampling_seeds == 3);
        assert(cfg.seed_base == 42);
        assert(cfg.seed_stride == 10007);
        assert(cfg.g > 0.0);
        assert(cfg.g_values.size() == 1 && cfg.g_values[0] == cfg.g);
        assert(cfg.n_values.size() == 3);
        assert(cfg.n_values[0] == 16 && cfg.n_values[1] == 20 && cfg.n_values[2] == 24);
        assert(cfg.scan_params.size() == 4);
        assert(cfg.scan_params[0].k_steps == 5 && cfg.scan_params[0].m_samples == 400 && cfg.scan_params[0].dt == 0.4);
        assert(cfg.csv_results_filename == "skqd_benchmark_results.csv");
        assert(cfg.csv_summary_filename == "skqd_summary_statistics.csv");
        auto seeds = cfg.generate_sampling_seeds();
        assert(seeds.size() == 3);
        assert(seeds[0] == 42 && seeds[1] == 10049 && seeds[2] == 20056);
        std::cout << "  --> PASSED: default_benchmark.cfg\n";
    }

    // 2. Test loading quick_test.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/quick_test.cfg");
        assert(cfg.target_minima_count == 5);
        assert(cfg.pool_minima_count == 5);
        assert(cfg.top_minima_count == 2);
        assert(cfg.num_sampling_seeds == 1);
        assert(std::abs(cfg.g - 2.0) < 1e-9);
        assert(cfg.n_values.size() == 1 && cfg.n_values[0] == 16);
        assert(cfg.scan_params.size() == 1);
        assert(cfg.scan_params[0].k_steps == 5 && cfg.scan_params[0].m_samples == 100 && cfg.scan_params[0].dt == 0.4);
        assert(cfg.csv_results_filename == "test_benchmark_results.csv");
        assert(cfg.csv_summary_filename == "test_summary_statistics.csv");
        std::cout << "  --> PASSED: quick_test.cfg\n";
    }

    // 3. Test loading deep_scan.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/deep_scan.cfg");
        assert(cfg.target_minima_count == 200);
        assert(cfg.num_sampling_seeds == 5);
        assert(std::abs(cfg.g - 2.0) < 1e-9);
        assert(cfg.n_values.size() == 3);
        assert(cfg.scan_params.size() == 4);
        std::cout << "  --> PASSED: deep_scan.cfg\n";
    }

    // 4. Test loading n16_only.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/n16_only.cfg");
        assert(cfg.target_minima_count == 100);
        assert(cfg.n_values.size() == 1 && cfg.n_values[0] == 16);
        assert(cfg.scan_params.size() == 4);
        std::cout << "  --> PASSED: n16_only.cfg\n";
    }

    // 5. Test loading g_scan.cfg (g_values array)
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/g_scan.cfg");
        assert(cfg.target_minima_count == 20);
        assert(cfg.num_sampling_seeds == 2);
        assert(cfg.g_values.size() == 4);
        assert(std::abs(cfg.g_values[0] - 0.5) < 1e-9);
        assert(std::abs(cfg.g_values[1] - 1.0) < 1e-9);
        assert(std::abs(cfg.g_values[2] - 1.5) < 1e-9);
        assert(std::abs(cfg.g_values[3] - 2.0) < 1e-9);
        std::cout << "  --> PASSED: g_scan.cfg (g_values list)\n";
    }

    // 6. Test loading n24_n20_budget2000.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/n24_n20_budget2000.cfg");
        assert(cfg.target_minima_count == 200);
        assert(cfg.pool_minima_count == 200);
        assert(cfg.top_minima_count == 10);
        assert(cfg.mandatory_energy == 52);
        assert(cfg.num_sampling_seeds == 3);
        assert(cfg.evaluation_budget == 2000);
        assert(cfg.conditional_n_timeout_seconds == 1800.0);
        assert(cfg.n_values.size() == 2 && cfg.n_values[0] == 24 && cfg.n_values[1] == 20);
        assert(cfg.scan_params.size() == 4);
        assert(cfg.scan_params[0].k_steps == 5 && cfg.scan_params[0].m_samples == 400);
        std::cout << "  --> PASSED: n24_n20_budget2000.cfg (N=24/20, budget=2000, E=52 mandatory)\n";
    }

    // 7. Test loading n24_delayed_window.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/n24_delayed_window.cfg");
        assert(cfg.target_minima_count == 200);
        assert(cfg.pool_minima_count == 200);
        assert(cfg.top_minima_count == 10);
        assert(cfg.mandatory_energy == -1);
        assert(cfg.num_sampling_seeds == 3);
        assert(cfg.evaluation_budget == 2000);
        assert(cfg.n_values.size() == 1 && cfg.n_values[0] == 24);
        assert(cfg.g_values.size() == 5);
        assert(std::abs(cfg.g_values[0] - 0.15) < 1e-9);
        assert(std::abs(cfg.g_values[1] - 0.25) < 1e-9);
        assert(std::abs(cfg.g_values[2] - 0.35) < 1e-9);
        assert(std::abs(cfg.g_values[3] - 0.5) < 1e-9);
        assert(std::abs(cfg.g_values[4] - 0.75) < 1e-9);
        assert(std::abs(cfg.g - 0.15) < 1e-9);
        assert(std::abs(cfg.t_start - 10.0) < 1e-9);
        assert(std::abs(cfg.dt_sample - 1.5) < 1e-9);
        assert(std::abs(cfg.dt - 0.05) < 1e-9);
        assert(cfg.k_steps == 20);
        assert(cfg.m_samples == 100);
        assert(cfg.quench_m_samples == 90);
        assert(cfg.quench_top_strings == 5);
        assert(cfg.kick_scan_d.size() == 6);
        assert(cfg.kick_scan_d[0] == 2 && cfg.kick_scan_d[5] == 12);
        assert(cfg.csv_results_filename == "n24_delayed_window_results.csv");
        assert(cfg.csv_summary_filename == "n24_delayed_window_summary.csv");
        std::cout << "  --> PASSED: n24_delayed_window.cfg (N=24, delayed window, 5-g sweep)\n";
    }

    // 8. Test loading n20_delayed_window.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/n20_delayed_window.cfg");
        assert(cfg.target_minima_count == 200);
        assert(cfg.pool_minima_count == 200);
        assert(cfg.top_minima_count == 10);
        assert(cfg.mandatory_energy == -1);
        assert(cfg.num_sampling_seeds == 3);
        assert(cfg.evaluation_budget == 2000);
        assert(cfg.n_values.size() == 1 && cfg.n_values[0] == 20);
        assert(cfg.g_values.size() == 5);
        assert(std::abs(cfg.g - 0.15) < 1e-9);
        assert(std::abs(cfg.t_start - 10.0) < 1e-9);
        assert(std::abs(cfg.dt_sample - 1.5) < 1e-9);
        assert(std::abs(cfg.dt - 0.05) < 1e-9);
        assert(cfg.k_steps == 20);
        assert(cfg.m_samples == 100);
        assert(cfg.quench_m_samples == 90);
        assert(cfg.quench_top_strings == 5);
        assert(cfg.kick_scan_d.size() == 6);
        assert(cfg.csv_results_filename == "n20_delayed_window_results.csv");
        assert(cfg.csv_summary_filename == "n20_delayed_window_summary.csv");
        std::cout << "  --> PASSED: n20_delayed_window.cfg (N=20, delayed window, 5-g sweep)\n";
    }

    // 9. Test loading n20_to_n26_delayed_window.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/n20_to_n26_delayed_window.cfg");
        assert(cfg.minima_selection_mode == "top_distinct_minima");
        assert(cfg.target_minima_count == 250);
        assert(cfg.pool_minima_count == 250);
        assert(cfg.top_minima_count == 10);
        assert(cfg.mandatory_energy == -1);
        assert(cfg.num_sampling_seeds == 3);
        assert(cfg.evaluation_budget == 2000);
        assert(cfg.n_values.size() == 4);
        assert(cfg.n_values[0] == 20 && cfg.n_values[1] == 22 && cfg.n_values[2] == 24 && cfg.n_values[3] == 26);
        assert(cfg.g_values.size() == 5);
        assert(std::abs(cfg.g - 0.15) < 1e-9);
        assert(std::abs(cfg.t_start - 10.0) < 1e-9);
        assert(std::abs(cfg.dt_sample - 1.5) < 1e-9);
        assert(std::abs(cfg.dt - 0.05) < 1e-9);
        assert(cfg.k_steps == 20);
        assert(cfg.m_samples == 100);
        assert(cfg.quench_m_samples == 90);
        assert(cfg.quench_top_strings == 5);
        assert(cfg.kick_scan_d.size() == 6);
        assert(cfg.csv_results_filename == "n20_n26_delayed_window_results.csv");
        assert(cfg.csv_summary_filename == "n20_n26_delayed_window_summary.csv");
        std::cout << "  --> PASSED: n20_to_n26_delayed_window.cfg (N=20..26, top_distinct_minima, 5-g sweep)\n";
    }

    // 10. Test loading n28_delayed_window_g035.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/n28_delayed_window_g035.cfg");
        assert(cfg.n_values.size() == 1 && cfg.n_values[0] == 28);
        assert(cfg.g_values.size() == 3);
        assert(std::abs(cfg.g_values[0] - 0.3) < 1e-9);
        assert(std::abs(cfg.g_values[1] - 0.35) < 1e-9);
        assert(std::abs(cfg.g_values[2] - 0.45) < 1e-9);
        assert(cfg.top_minima_count == 10);
        assert(cfg.pool_minima_count == 250);
        assert(cfg.seeds_dir == "seeds");
        assert(!cfg.force_regenerate_seeds);
        assert(cfg.csv_results_filename == "n28_delayed_window_g035_results.csv");
        assert(cfg.csv_summary_filename == "n28_delayed_window_g035_summary.csv");
        std::cout << "  --> PASSED: n28_delayed_window_g035.cfg (N=28, g in {0.3, 0.35, 0.45})\n";
    }

    // 11. Test loading n30_delayed_window_g035.cfg
    {
        auto cfg = BenchmarkConfig::load_from_file("configs/n30_delayed_window_g035.cfg");
        assert(cfg.n_values.size() == 1 && cfg.n_values[0] == 30);
        assert(cfg.g_values.size() == 3);
        assert(std::abs(cfg.g_values[0] - 0.3) < 1e-9);
        assert(std::abs(cfg.g_values[1] - 0.35) < 1e-9);
        assert(std::abs(cfg.g_values[2] - 0.45) < 1e-9);
        assert(cfg.top_minima_count == 10);
        assert(cfg.pool_minima_count == 250);
        assert(cfg.seeds_dir == "seeds");
        assert(!cfg.force_regenerate_seeds);
        assert(cfg.csv_results_filename == "n30_delayed_window_g035_results.csv");
        assert(cfg.csv_summary_filename == "n30_delayed_window_g035_summary.csv");
        std::cout << "  --> PASSED: n30_delayed_window_g035.cfg (N=30, g in {0.3, 0.35, 0.45})\n";
    }

    std::cout << "\nALL CONFIG PARSER TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
