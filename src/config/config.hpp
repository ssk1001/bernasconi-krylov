#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <iomanip>
#include <filesystem>

// Forward declaration of SamplerConfig if sampler.cuh is not included
struct SamplerConfig;

// ============================================================================
// Bernasconi-Krylov Modern Benchmark Configuration
// ============================================================================

namespace config_detail {

inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

inline std::string to_lower(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return lower;
}

inline std::string strip_comments(const std::string& line) {
    bool in_quote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"' || line[i] == '\'') {
            in_quote = !in_quote;
        } else if (!in_quote && (line[i] == '#' || line[i] == ';')) {
            return line.substr(0, i);
        }
    }
    return line;
}

inline bool parse_bool(const std::string& val) {
    std::string s = to_lower(trim(val));
    if (s == "1" || s == "true" || s == "yes" || s == "on") return true;
    if (s == "0" || s == "false" || s == "no" || s == "off") return false;
    throw std::invalid_argument("Cannot parse boolean value: '" + val + "'");
}

template <typename T>
inline std::vector<T> parse_list(const std::string& val) {
    std::vector<T> items;
    std::stringstream ss(val);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim(item);
        if (!item.empty()) {
            std::stringstream item_ss(item);
            T v;
            if (item_ss >> v) {
                items.push_back(v);
            }
        }
    }
    return items;
}

} // namespace config_detail

// ----------------------------------------------------------------------------
// 1. System & Global Settings
// ----------------------------------------------------------------------------
struct SystemConfig {
    std::vector<int> n_values = {16, 20, 24};
    uint64_t seed_base = 42;
    std::string device = "auto"; // "auto", "gpu", "cpu"
};

// ----------------------------------------------------------------------------
// 2. Trap Basin Collection & Selection
// ----------------------------------------------------------------------------
struct TrapsConfig {
    size_t pool_minima_count = 100;
    size_t benchmark_trap_count = 5;
    std::string minima_cache_dir = "seeds";
    bool force_regenerate_traps = false;
};

// ----------------------------------------------------------------------------
// 3. Quantum Dynamics & Sampling Window
// ----------------------------------------------------------------------------
struct QuantumConfig {
    std::vector<double> g_values = {0.25, 0.5, 0.75};
    double dt = 0.05;              // Trotter simulation integration step
    double t_start = 10.0;         // Sampling start time (burn-in delay)
    double dt_sample = 1.5;        // Sampling delta t (interval between measurement points)
    int k_sample_points = 20;      // Number of measurement points K (times t_k = t_start + k * dt_sample)
    int m_shots_per_point = 100;   // Projective measurement shots M per sampling point
    bool track_p_less = true;      // Track exact noise-free P_<(t) via GPU reduction
    bool verbose = true;           // Log progress / summary

    // Continuous trajectory logging flags
    bool record_trajectory = false; // When true: records continuous P_<(t) from t=0 to t_max
    double dt_trajectory = 0.0;     // Trajectory logging interval (<= 0.0 defaults to dt, i.e. every step)
    double t_max_val = 0.0;         // Explicit t_max duration (if > 0.0, overrides derived t_max)
    bool run_classical = true;      // Run classical baselines (false skips Kick/Restart search for pure quantum runs)

    // Derived: Total duration t_max
    double t_max() const {
        if (t_max_val > 0.0) return t_max_val;
        if (k_sample_points <= 1) return t_start;
        double step = (dt_sample > 0.0) ? dt_sample : dt;
        return t_start + (k_sample_points - 1) * step;
    }

    // Derived: Evaluation time points t_k
    std::vector<double> sample_times() const {
        std::vector<double> times;
        if (record_trajectory) {
            double step = (dt_trajectory > 0.0) ? dt_trajectory : dt;
            if (step <= 0.0) step = 0.05;
            double max_t = t_max();
            int n_pts = static_cast<int>(std::round(max_t / step)) + 1;
            times.reserve(n_pts);
            for (int k = 0; k < n_pts; ++k) {
                times.push_back(k * step);
            }
            return times;
        }
        if (k_sample_points <= 0) return times;
        times.reserve(k_sample_points);
        double step = (dt_sample > 0.0) ? dt_sample : dt;
        for (int k = 0; k < k_sample_points; ++k) {
            times.push_back(t_start + k * step);
        }
        return times;
    }

    // Derived: Trotter steps per sampling interval
    size_t trotter_steps_per_sample() const {
        double step = (dt_sample > 0.0) ? dt_sample : dt;
        if (dt <= 0.0) return 1;
        return std::max<size_t>(1, static_cast<size_t>(std::round(step / dt)));
    }
};

// ----------------------------------------------------------------------------
// 4. Classical Baselines (Trials-Based Protocol)
// ----------------------------------------------------------------------------
struct ClassicalConfig {
    size_t kick_trials_per_d = 50000;            // Trials per kick distance (10^4 - 10^5)
    std::vector<int> kick_d_values = {1, 2, 3, 4, 5, 6}; // Hamming distances to sweep
    size_t restart_trials = 50000;               // Trials for Restart-SD (10^4 - 10^5)
    size_t max_evals_per_start = 0;              // Eval ceiling per trajectory (0 = unlimited until local min)
    size_t batch_size = 10000;                   // Concurrency chunk size for GPU batch descent
};

// ----------------------------------------------------------------------------
// 5. Output & Logging
// ----------------------------------------------------------------------------
struct OutputConfig {
    std::string output_dir = "data_raw";
    std::string results_csv = "benchmark_results_n{N}.csv";
    std::string time_series_csv = "p_less_dynamics_n{N}.csv";
    std::string summary_csv = "benchmark_summary.csv";
    std::string structured_json = "structured_benchmark_n{N}.json";
    std::string all_structured_json = "all_seeds_benchmark.json";
    bool verbose = true;

    // Helper: Substitute {N} or {n} and join with output_dir
    std::string resolve_path(const std::string& pattern, int n = 0) const {
        std::string s = pattern;
        std::string n_str = std::to_string(n);

        // Replace {N} and {n}
        size_t pos;
        while ((pos = s.find("{N}")) != std::string::npos) s.replace(pos, 3, n_str);
        while ((pos = s.find("{n}")) != std::string::npos) s.replace(pos, 3, n_str);

        std::filesystem::path p(s);
        if (p.is_relative() && !output_dir.empty() && p.parent_path().empty()) {
            return (std::filesystem::path(output_dir) / p).string();
        }
        return p.string();
    }

    std::string get_results_path(int n) const {
        return resolve_path(results_csv, n);
    }

    std::string get_time_series_path(int n) const {
        return resolve_path(time_series_csv, n);
    }

    std::string get_summary_path() const {
        return resolve_path(summary_csv, 0);
    }

    std::string get_structured_json_path(int n) const {
        return resolve_path(structured_json, n);
    }

    std::string get_all_structured_json_path() const {
        return resolve_path(all_structured_json, 0);
    }
};

// ============================================================================
// Top-Level Unified BenchmarkConfig
// ============================================================================
struct BenchmarkConfig {
    SystemConfig system;
    TrapsConfig traps;
    QuantumConfig quantum;
    ClassicalConfig classical;
    OutputConfig output;

    // Load from configuration file (.cfg, .ini)
    static BenchmarkConfig load_from_file(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw std::runtime_error("BenchmarkConfig: Cannot open file: " + filepath);
        }
        std::stringstream ss;
        ss << file.rdbuf();
        return load_from_string(ss.str());
    }

    // Load from string content
    static BenchmarkConfig load_from_string(const std::string& content) {
        BenchmarkConfig cfg;
        std::stringstream stream(content);
        std::string line;
        std::string current_section = "";
        int line_no = 0;

        while (std::getline(stream, line)) {
            line_no++;
            std::string cleaned = config_detail::strip_comments(line);
            cleaned = config_detail::trim(cleaned);
            if (cleaned.empty()) continue;

            // Section header: [section]
            if (cleaned.front() == '[' && cleaned.back() == ']') {
                current_section = config_detail::to_lower(
                    config_detail::trim(cleaned.substr(1, cleaned.size() - 2))
                );
                continue;
            }

            // Key = Value
            size_t delim_pos = cleaned.find('=');
            if (delim_pos == std::string::npos) {
                delim_pos = cleaned.find(':');
            }
            if (delim_pos == std::string::npos) {
                std::cerr << "[Config Warning] Malformed line " << line_no << ": " << line << "\n";
                continue;
            }

            std::string key = config_detail::to_lower(config_detail::trim(cleaned.substr(0, delim_pos)));
            std::string val = config_detail::trim(cleaned.substr(delim_pos + 1));

            cfg.apply_key_value(current_section, key, val, line_no);
        }

        return cfg;
    }

    void apply_key_value(const std::string& section, const std::string& key, const std::string& val, int line_no) {
        // 1. System section
        if (key == "n_values" || key == "n" || key == "n_spins") {
            system.n_values = config_detail::parse_list<int>(val);
        } else if (key == "seed_base" || key == "seed" || key == "master_seed") {
            system.seed_base = std::stoull(val);
        } else if (key == "device" || key == "backend") {
            system.device = config_detail::to_lower(val);
        }

        // 2. Traps section
        else if (key == "pool_minima_count" || key == "target_minima_count" || key == "minima_pool_size") {
            traps.pool_minima_count = std::stoul(val);
        } else if (key == "benchmark_trap_count" || key == "top_minima_count" || key == "num_traps") {
            traps.benchmark_trap_count = std::stoul(val);
        } else if (key == "minima_cache_dir" || key == "seeds_dir" || key == "seed_dir") {
            traps.minima_cache_dir = val;
        } else if (key == "force_regenerate_traps" || key == "force_regenerate_seeds" || key == "regenerate_seeds") {
            traps.force_regenerate_traps = config_detail::parse_bool(val);
        }

        // 3. Quantum section
        else if (key == "g_values") {
            quantum.g_values = config_detail::parse_list<double>(val);
        } else if (key == "g") {
            quantum.g_values = {std::stod(val)};
        } else if (key == "dt" || key == "simulation_dt") {
            quantum.dt = std::stod(val);
        } else if (key == "t_start" || key == "sampling_start_time" || key == "burn_in_time") {
            quantum.t_start = std::stod(val);
        } else if (key == "dt_sample" || key == "sample_dt" || key == "sampling_delta_t") {
            quantum.dt_sample = std::stod(val);
        } else if (key == "k_sample_points" || key == "k_steps" || key == "num_sample_points" || key == "k") {
            quantum.k_sample_points = std::stoi(val);
        } else if (key == "m_shots_per_point" || key == "m_samples" || key == "shots_per_point" || key == "m") {
            quantum.m_shots_per_point = std::stoi(val);
        } else if (key == "track_p_less") {
            quantum.track_p_less = config_detail::parse_bool(val);
        } else if (key == "record_trajectory" || key == "track_trajectory" || key == "continuous_trajectory") {
            quantum.record_trajectory = config_detail::parse_bool(val);
        } else if (key == "dt_trajectory" || key == "dt_log" || key == "trajectory_dt") {
            quantum.dt_trajectory = std::stod(val);
        } else if (key == "t_max" || key == "simulation_t_max") {
            quantum.t_max_val = std::stod(val);
        } else if (key == "run_classical" || key == "enable_classical") {
            quantum.run_classical = config_detail::parse_bool(val);
        }

        // 4. Classical section
        else if (key == "kick_trials_per_d" || key == "classical_trials_per_d" || key == "num_classical_trials") {
            classical.kick_trials_per_d = std::stoul(val);
            // Default restart_trials to match kick_trials_per_d if not explicitly specified
            classical.restart_trials = classical.kick_trials_per_d;
        } else if (key == "kick_d_values" || key == "kick_scan_d" || key == "d_scan" || key == "kick_distances") {
            classical.kick_d_values = config_detail::parse_list<int>(val);
        } else if (key == "restart_trials" || key == "num_restart_trials") {
            classical.restart_trials = std::stoul(val);
        } else if (key == "max_evals_per_start" || key == "evaluation_budget") {
            classical.max_evals_per_start = std::stoul(val);
        } else if (key == "batch_size") {
            classical.batch_size = std::stoul(val);
        }

        // 5. Output section
        else if (key == "output_dir") {
            output.output_dir = val;
        } else if (key == "results_csv" || key == "csv_results" || key == "csv_output") {
            output.results_csv = val;
        } else if (key == "time_series_csv" || key == "csv_time_series" || key == "csv_trajectory") {
            output.time_series_csv = val;
        } else if (key == "structured_json" || key == "json_output" || key == "structured_output" || key == "json_benchmark") {
            output.structured_json = val;
        } else if (key == "all_structured_json" || key == "all_json_output" || key == "overall_json") {
            output.all_structured_json = val;
        } else if (key == "summary_csv" || key == "csv_summary") {
            output.summary_csv = val;
        } else if (key == "verbose") {
            bool v = config_detail::parse_bool(val);
            output.verbose = v;
            quantum.verbose = v;
        } else {
            std::cerr << "[Config Warning] Unknown parameter on line " << line_no << ": '" << key << "'\n";
        }
    }

    // Print summary to an output stream
    void print(std::ostream& os = std::cout) const {
        os << "======================================================================\n";
        os << "                  Bernasconi-Krylov Configuration                     \n";
        os << "======================================================================\n";
        os << "[System]\n";
        os << "  N values                 : ";
        for (size_t i = 0; i < system.n_values.size(); ++i) {
            os << system.n_values[i] << (i + 1 < system.n_values.size() ? ", " : "\n");
        }
        os << "  Master Seed              : " << system.seed_base << "\n";
        os << "  Device                   : " << system.device << "\n";

        os << "[Traps]\n";
        os << "  Pool Minima Count        : " << traps.pool_minima_count << "\n";
        os << "  Benchmark Trap Count     : " << traps.benchmark_trap_count << "\n";
        os << "  Minima Cache Dir         : " << traps.minima_cache_dir << "\n";
        os << "  Force Regenerate         : " << (traps.force_regenerate_traps ? "true" : "false") << "\n";

        os << "[Quantum]\n";
        os << "  g values                 : ";
        for (size_t i = 0; i < quantum.g_values.size(); ++i) {
            os << quantum.g_values[i] << (i + 1 < quantum.g_values.size() ? ", " : "\n");
        }
        os << "  Trotter dt               : " << quantum.dt << "\n";
        os << "  Sampling Start Time      : " << quantum.t_start << "\n";
        os << "  Sampling Delta t         : " << quantum.dt_sample << "\n";
        os << "  K sample points          : " << quantum.k_sample_points << "\n";
        os << "  M shots per point        : " << quantum.m_shots_per_point << "\n";
        os << "  Calculated t_max         : " << quantum.t_max() << "\n";
        os << "  Track P_< (noise-free)   : " << (quantum.track_p_less ? "true" : "false") << "\n";
        os << "  Record Trajectory        : " << (quantum.record_trajectory ? "true" : "false") << "\n";
        if (quantum.record_trajectory) {
            os << "  Trajectory dt            : " << ((quantum.dt_trajectory > 0.0) ? quantum.dt_trajectory : quantum.dt) << "\n";
        }
        os << "  Run Classical Baselines  : " << (quantum.run_classical ? "true" : "false") << "\n";

        os << "[Classical]\n";
        os << "  Kick Trials per d        : " << classical.kick_trials_per_d << "\n";
        os << "  Kick d values            : ";
        for (size_t i = 0; i < classical.kick_d_values.size(); ++i) {
            os << classical.kick_d_values[i] << (i + 1 < classical.kick_d_values.size() ? ", " : "\n");
        }
        os << "  Restart Trials           : " << classical.restart_trials << "\n";
        os << "  Max Evals per Trajectory : " << classical.max_evals_per_start << " (0 = unlimited)\n";
        os << "  Batch Size               : " << classical.batch_size << "\n";

        os << "[Output]\n";
        os << "  Output Dir               : " << output.output_dir << "\n";
        os << "  Results CSV Template     : " << output.results_csv << "\n";
        os << "  Time Series CSV Template : " << output.time_series_csv << "\n";
        os << "  Summary CSV Template     : " << output.summary_csv << "\n";
        os << "======================================================================\n";
    }

    // Convert to canonical INI string
    std::string to_ini_string() const {
        std::ostringstream ss;
        ss << "# Bernasconi-Krylov Benchmark Configuration\n\n";
        ss << "[system]\n";
        ss << "n_values = ";
        for (size_t i = 0; i < system.n_values.size(); ++i)
            ss << system.n_values[i] << (i + 1 < system.n_values.size() ? ", " : "\n");
        ss << "seed_base = " << system.seed_base << "\n";
        ss << "device = " << system.device << "\n\n";

        ss << "[traps]\n";
        ss << "pool_minima_count = " << traps.pool_minima_count << "\n";
        ss << "benchmark_trap_count = " << traps.benchmark_trap_count << "\n";
        ss << "minima_cache_dir = " << traps.minima_cache_dir << "\n";
        ss << "force_regenerate_traps = " << (traps.force_regenerate_traps ? "true" : "false") << "\n\n";

        ss << "[quantum]\n";
        ss << "g_values = ";
        for (size_t i = 0; i < quantum.g_values.size(); ++i)
            ss << quantum.g_values[i] << (i + 1 < quantum.g_values.size() ? ", " : "\n");
        ss << "dt = " << quantum.dt << "\n";
        ss << "t_start = " << quantum.t_start << "\n";
        ss << "dt_sample = " << quantum.dt_sample << "\n";
        ss << "k_sample_points = " << quantum.k_sample_points << "\n";
        ss << "m_shots_per_point = " << quantum.m_shots_per_point << "\n";
        ss << "track_p_less = " << (quantum.track_p_less ? "true" : "false") << "\n";
        ss << "record_trajectory = " << (quantum.record_trajectory ? "true" : "false") << "\n";
        if (quantum.record_trajectory) {
            ss << "dt_trajectory = " << quantum.dt_trajectory << "\n";
        }
        if (quantum.t_max_val > 0.0) {
            ss << "t_max = " << quantum.t_max_val << "\n";
        }
        ss << "run_classical = " << (quantum.run_classical ? "true" : "false") << "\n\n";

        ss << "[classical]\n";
        ss << "kick_trials_per_d = " << classical.kick_trials_per_d << "\n";
        ss << "kick_d_values = ";
        for (size_t i = 0; i < classical.kick_d_values.size(); ++i)
            ss << classical.kick_d_values[i] << (i + 1 < classical.kick_d_values.size() ? ", " : "\n");
        ss << "restart_trials = " << classical.restart_trials << "\n";
        ss << "max_evals_per_start = " << classical.max_evals_per_start << "\n";
        ss << "batch_size = " << classical.batch_size << "\n\n";

        ss << "[output]\n";
        ss << "output_dir = " << output.output_dir << "\n";
        ss << "results_csv = " << output.results_csv << "\n";
        ss << "time_series_csv = " << output.time_series_csv << "\n";
        ss << "summary_csv = " << output.summary_csv << "\n";
        ss << "verbose = " << (output.verbose ? "true" : "false") << "\n";

        return ss.str();
    }
};
