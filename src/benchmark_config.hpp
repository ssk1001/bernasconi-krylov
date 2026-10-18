#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <iomanip>

struct ScanConfig {
    int k_steps = 5;
    int m_samples = 400;
    double dt = 0.4;
    double g = 0.5; // Transverse field strength for this scan
};

struct BenchmarkConfig {
    // Experiment parameters
    size_t pool_minima_count = 100;
    size_t target_minima_count = 100; // alias for backwards compatibility
    size_t top_minima_count = 5;      // number of distinct energy levels (traps) to evaluate individually
    size_t num_sampling_seeds = 3;
    uint64_t seed_base = 42;
    uint64_t seed_stride = 10007;

    // Seed caching and persistence parameters
    std::string seeds_dir = "seeds";
    bool force_regenerate_seeds = false;

    // Physical & sampling parameters
    double g = 0.15;                                                  // Default / global transverse field strength
    std::vector<double> g_values = {0.15, 0.25, 0.35, 0.5, 0.75};    // List of g values to evaluate (if multiple)
    size_t top_k_augmentation = 0;

    // System sizes to benchmark
    std::vector<size_t> n_values = {16, 20, 24};

    // Trotter parameter scans
    std::vector<ScanConfig> scan_params = {
        {5, 400, 0.4, 0.5},
        {10, 200, 0.2, 0.5},
        {20, 100, 0.1, 0.5},
        {40, 50, 0.05, 0.5}
    };

    // Output files
    std::string csv_results_filename = "skqd_benchmark_results.csv";
    std::string csv_summary_filename = "skqd_summary_statistics.csv";

    // Evaluation budget parameters (0 = dynamic subspace_dim fallback for backwards compatibility)
    size_t evaluation_budget = 0;

    // Specific mandatory energy level to guarantee in selected minima (e.g. 52; -1 = disabled)
    long long mandatory_energy = -1;

    // Minima selection strategy: "top_distinct_minima" (lowest-energy symmetry orbits, default) or legacy "distinct_energies"
    std::string minima_selection_mode = "top_distinct_minima";

    // Delayed sampling window parameters
    double t_start = 0.0;             // Delay before sampling starts (e.g. 10.0; 0 = start at t=0)
    double dt_sample = 0.0;           // Sampling interval (e.g. 1.5; 0 = sample at dt)
    double dt = 0.05;                 // Simulation Trotter step size
    size_t k_steps = 20;              // Number of measurement points (e.g. 20)
    size_t m_samples = 100;           // Shots per point for Method 1 (e.g. 100 -> 2000 shots)

    // Method 2 (5-string quench) parameters
    size_t quench_m_samples = 90;     // Shots per point for Method 2 (e.g. 90 -> 1800 shots)
    size_t quench_top_strings = 5;    // Number of lowest sampled strings to quench with SD (5)

    // Method 3 (Random kick scan) parameters
    std::vector<int> kick_scan_d = {2, 4, 6, 8, 10, 12}; // Kick distances to scan

    // Large-scale classical ensemble trial counts (10^4 - 10^5)
    size_t classical_trials_per_d = 50000; // Trials per kick distance d
    size_t restart_trials = 50000;         // Trials for Random Restart SD

    // Conditional execution threshold for subsequent system sizes (seconds; 0 = unconditional)
    double conditional_n_timeout_seconds = 0.0;

    // Track the source config path (empty if using defaults)
    std::string source_filepath = "built-in defaults";

    std::vector<uint64_t> generate_sampling_seeds() const {
        std::vector<uint64_t> seeds(num_sampling_seeds);
        for (size_t s = 0; s < num_sampling_seeds; ++s) {
            seeds[s] = seed_base + s * seed_stride;
        }
        return seeds;
    }

    void print_summary() const {
        std::cout << "======================================================================\n";
        std::cout << "                 SKQD BENCHMARK CONFIGURATION\n";
        std::cout << "======================================================================\n";
        std::cout << "  Config Source          : " << source_filepath << "\n";
        std::cout << "  Minima Pool Size       : >= " << pool_minima_count << " (deduplicated by symmetry)\n";
        std::cout << "  Evaluated Traps (Top-x): " << top_minima_count << " symmetry-distinct orbits (mode=" << minima_selection_mode << ")\n";
        std::cout << "  Sampling Seeds per Trap: " << num_sampling_seeds 
                  << " (base=" << seed_base << ", stride=" << seed_stride << ")\n";
        if (g_values.size() > 1) {
            std::cout << "  Coupling strength(s) g : ";
            for (size_t i = 0; i < g_values.size(); ++i) {
                std::cout << g_values[i] << (i + 1 < g_values.size() ? ", " : "\n");
            }
        } else {
            std::cout << "  Coupling strength g    : " << g << "\n";
        }
        std::cout << "  Top-K Augmentation     : " << top_k_augmentation << "\n";
        std::cout << "  System Sizes N         : ";
        for (size_t i = 0; i < n_values.size(); ++i) {
            std::cout << n_values[i] << (i + 1 < n_values.size() ? ", " : "\n");
        }
        std::cout << "  Trotter Scans (K, M, dt, g):\n";
        for (size_t i = 0; i < scan_params.size(); ++i) {
            std::cout << "    [" << i + 1 << "] K=" << scan_params[i].k_steps 
                      << ", M=" << scan_params[i].m_samples 
                      << ", dt=" << scan_params[i].dt 
                      << ", g=" << scan_params[i].g
                      << " (Shots=" << (scan_params[i].k_steps * scan_params[i].m_samples) << ")\n";
        }
        if (t_start > 0.0 || dt_sample > 0.0) {
            std::cout << "  Delayed Window Sampling: t_start=" << t_start << ", dt_sample=" << dt_sample 
                      << ", K=" << k_steps << ", M=" << m_samples << " (Method 1 shots=" << (k_steps * m_samples) << ")\n";
            std::cout << "  Method 2 Quench Setup  : M=" << quench_m_samples << " (shots=" << (k_steps * quench_m_samples) 
                      << "), Top Strings=" << quench_top_strings << "\n";
            std::cout << "  Method 3 Kick Grid (d) : ";
            for (size_t i = 0; i < kick_scan_d.size(); ++i) {
                std::cout << kick_scan_d[i] << (i + 1 < kick_scan_d.size() ? ", " : "\n");
            }
        }
        std::cout << "  Classical Ensemble Size: " << classical_trials_per_d << " trials/d (Kick), "
                  << restart_trials << " trials (Restart)\n";
        std::cout << "  Detailed CSV Output    : " << csv_results_filename << "\n";
        std::cout << "  Statistical CSV Output : " << csv_summary_filename << "\n";
        if (evaluation_budget > 0) {
            std::cout << "  Evaluation Budget B    : " << evaluation_budget << " evals per trial (fixed)\n";
        }
        if (mandatory_energy >= 0) {
            std::cout << "  Mandatory Trap Energy  : E = " << mandatory_energy << " (guaranteed inclusion)\n";
        }
        if (conditional_n_timeout_seconds > 0) {
            std::cout << "  Conditional N Timeout  : " << conditional_n_timeout_seconds << " s\n";
        }
        std::cout << "======================================================================\n" << std::endl;
    }

    static BenchmarkConfig load_from_file(const std::string& filepath) {
        BenchmarkConfig cfg;
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open configuration file: " + filepath);
        }

        cfg.source_filepath = filepath;
        bool scan_cleared = false;
        bool n_cleared = false;
        bool g_vals_cleared = false;

        auto trim = [](const std::string& s) -> std::string {
            size_t start = s.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) return "";
            size_t end = s.find_last_not_of(" \t\r\n");
            return s.substr(start, end - start + 1);
        };

        std::string line;
        size_t line_no = 0;

        while (std::getline(file, line)) {
            line_no++;
            size_t comment_pos = line.find_first_of("#;");
            if (comment_pos != std::string::npos) {
                line = line.substr(0, comment_pos);
            }
            line = trim(line);
            if (line.empty()) continue;

            size_t eq_pos = line.find('=');
            if (eq_pos == std::string::npos) {
                std::cerr << "[Config Warning] Line " << line_no << " missing '=': " << line << "\n";
                continue;
            }

            std::string key = trim(line.substr(0, eq_pos));
            std::string val = trim(line.substr(eq_pos + 1));
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);

            if (key == "pool_minima_count" || key == "target_minima_count" || key == "minima_count") {
                cfg.pool_minima_count = std::stoul(val);
                cfg.target_minima_count = cfg.pool_minima_count;
            } else if (key == "top_minima_count" || key == "eval_top_x_minima" || key == "num_seeds") {
                cfg.top_minima_count = std::stoul(val);
            } else if (key == "num_sampling_seeds" || key == "sampling_seeds") {
                cfg.num_sampling_seeds = std::stoul(val);
            } else if (key == "seed_base") {
                cfg.seed_base = std::stoull(val);
            } else if (key == "seed_stride") {
                cfg.seed_stride = std::stoull(val);
            } else if (key == "g" || key == "g_transverse") {
                cfg.g = std::stod(val);
                if (!g_vals_cleared) {
                    cfg.g_values = {cfg.g};
                }
                for (auto& sc : cfg.scan_params) {
                    sc.g = cfg.g;
                }
            } else if (key == "g_values") {
                if (!g_vals_cleared) {
                    cfg.g_values.clear();
                    g_vals_cleared = true;
                }
                std::stringstream ss(val);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item = trim(item);
                    if (!item.empty()) {
                        cfg.g_values.push_back(std::stod(item));
                    }
                }
                if (!cfg.g_values.empty()) {
                    cfg.g = cfg.g_values[0];
                }
            } else if (key == "top_k_augmentation") {
                cfg.top_k_augmentation = std::stoul(val);
            } else if (key == "csv_results" || key == "output_results_csv") {
                cfg.csv_results_filename = val;
            } else if (key == "csv_summary" || key == "output_summary_csv") {
                cfg.csv_summary_filename = val;
            } else if (key == "n_values" || key == "n") {
                if (!n_cleared) {
                    cfg.n_values.clear();
                    n_cleared = true;
                }
                std::stringstream ss(val);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item = trim(item);
                    if (!item.empty()) {
                        cfg.n_values.push_back(std::stoul(item));
                    }
                }
            } else if (key == "scan") {
                if (!scan_cleared) {
                    cfg.scan_params.clear();
                    scan_cleared = true;
                }
                // Parse comma-separated "K, M, dt [, g]"
                std::stringstream ss(val);
                std::string token;
                std::vector<std::string> tokens;
                while (std::getline(ss, token, ',')) {
                    token = trim(token);
                    if (!token.empty()) tokens.push_back(token);
                }
                if (tokens.size() >= 3) {
                    ScanConfig sc;
                    sc.k_steps = std::stoi(tokens[0]);
                    sc.m_samples = std::stoi(tokens[1]);
                    sc.dt = std::stod(tokens[2]);
                    sc.g = (tokens.size() >= 4) ? std::stod(tokens[3]) : cfg.g;
                    cfg.scan_params.push_back(sc);
                } else {
                    std::cerr << "[Config Warning] Malformed 'scan' format on line " << line_no 
                              << ". Expected: scan = K, M, dt [, g]\n";
                }
            } else if (key == "evaluation_budget" || key == "budget" || key == "budget_b") {
                cfg.evaluation_budget = std::stoul(val);
            } else if (key == "mandatory_energy" || key == "include_energy" || key == "target_energy") {
                cfg.mandatory_energy = std::stoll(val);
            } else if (key == "minima_selection_mode" || key == "selection_mode") {
                cfg.minima_selection_mode = val;
                if (cfg.minima_selection_mode == "lowest_energy_orbits" ||
                    cfg.minima_selection_mode == "lowest_energy_minima") {
                    cfg.minima_selection_mode = "top_distinct_minima";
                }
            } else if (key == "conditional_n_timeout_seconds" || key == "n_timeout_seconds" || key == "conditional_timeout") {
                cfg.conditional_n_timeout_seconds = std::stod(val);
            } else if (key == "t_start" || key == "window_start" || key == "t_delay") {
                cfg.t_start = std::stod(val);
            } else if (key == "dt_sample" || key == "sample_interval" || key == "delta_t_sample") {
                cfg.dt_sample = std::stod(val);
            } else if (key == "dt") {
                cfg.dt = std::stod(val);
            } else if (key == "k_steps" || key == "k") {
                cfg.k_steps = std::stoul(val);
            } else if (key == "m_samples" || key == "m") {
                cfg.m_samples = std::stoul(val);
            } else if (key == "quench_m_samples" || key == "m_quench") {
                cfg.quench_m_samples = std::stoul(val);
            } else if (key == "quench_top_strings" || key == "top_quench_strings") {
                cfg.quench_top_strings = std::stoul(val);
            } else if (key == "kick_scan_d" || key == "d_scan" || key == "kick_distances") {
                cfg.kick_scan_d.clear();
                std::stringstream ss(val);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item = trim(item);
                    if (!item.empty()) cfg.kick_scan_d.push_back(std::stoi(item));
                }
            } else if (key == "classical_trials_per_d" || key == "classical_trials" || key == "num_classical_trials") {
                cfg.classical_trials_per_d = std::stoul(val);
                cfg.restart_trials = cfg.classical_trials_per_d; // Default restart trials to match unless overridden
            } else if (key == "restart_trials" || key == "num_restart_trials") {
                cfg.restart_trials = std::stoul(val);
            } else if (key == "seeds_dir" || key == "seed_dir") {
                cfg.seeds_dir = val;
            } else if (key == "force_regenerate_seeds" || key == "regenerate_seeds") {
                std::string v = val;
                std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                cfg.force_regenerate_seeds = (v == "1" || v == "true" || v == "yes");
            } else {
                std::cerr << "[Config Warning] Unknown parameter on line " << line_no << ": " << key << "\n";
            }
        }

        if (cfg.n_values.empty()) {
            std::cerr << "[Config Warning] No N values specified; defaulting to 16.\n";
            cfg.n_values = {16};
        }
        if (cfg.scan_params.empty()) {
            std::cerr << "[Config Warning] No scans specified; using default scan (K=5, M=400, dt=0.4, g=" << cfg.g << ").\n";
            cfg.scan_params.push_back({5, 400, 0.4, cfg.g});
        }

        return cfg;
    }
};
