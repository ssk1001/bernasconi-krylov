#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <memory>
#include <filesystem>
#include <algorithm>
#include <cmath>

#include "../config/config.hpp"
#include "../labs_energy/labs_energy.hpp"
#include "../classical/classical.cuh"
#include "../sampling/sampler.cuh"
#include "../subspace/diagonalizer.cuh"

// ============================================================================
// Data Structures for Connector Results & Logging
// ============================================================================

struct SingleTrapBenchmark {
    int n = 0;
    double g = 0.0;
    size_t trap_rank = 0;
    SpinState trap_state;
    int64_t trap_energy = 0; // E_0
    int d_min = 0;           // Barrier distance to nearest state with E < E_0

    // Quantum dynamics metrics
    double window_avg_p_less = 0.0;     // Noise-free window-averaged P_<
    double window_avg_dh = 0.0;         // Quantum diffusion window-averaged <dH>
    double initial_rise_rate_avg = 0.0; // Mean finite difference slope while increasing: P_<(t_peak1) / t_peak1
    double initial_rise_rate_max = 0.0; // Max instantaneous slope: max delta P_< / delta t
    size_t quantum_improved_shots = 0;
    size_t quantum_total_shots = 0;
    double quantum_success_rate = 0.0;
    double quantum_per_eval_rate = 0.0;
    size_t quantum_basis_size = 0;
    double quantum_time_sec = 0.0;
    std::vector<double> time_points;
    std::vector<double> p_less_history;
    std::vector<double> dh_history;     // Instantaneous <dH(t)> trajectory
    double p_blind = 0.0;
    double peak_p_less = 0.0;
    double peak_time = 0.0;

    // Classical Kick+SD metrics (Method 3)
    int best_d = 0; // d* = argmax_d r(d)
    double best_d_rate = 0.0; // r(d*) = successes(d*) / evaluations(d*)
    size_t best_d_successes = 0;
    size_t best_d_trials = 0;
    size_t best_d_evals = 0;
    std::map<int, double> rate_by_d;
    std::map<int, size_t> successes_by_d;
    std::map<int, size_t> trials_by_d;
    std::map<int, size_t> evals_by_d;
    double kick_time_sec = 0.0;

    // Classical Restart-SD metrics (Method 4)
    double restart_rate = 0.0; // r_restart = successes / evaluations
    size_t restart_successes = 0;
    size_t restart_trials = 0;
    size_t restart_evals = 0;
    double restart_time_sec = 0.0;
};

// ============================================================================
// BenchmarkConnector: Bridges Configurations to Execution Pipelines
// ============================================================================

class BenchmarkConnector {
public:
    BenchmarkConnector() = default;

    /// Resolves the directory for seed files, checking configured dir, current working dir, and parent dir
    static std::string resolve_seeds_dir(const std::string& configured_dir = "seeds") {
        namespace fs = std::filesystem;
        if (!configured_dir.empty() && configured_dir != "seeds") {
            return configured_dir;
        }
        try {
            if (fs::exists("seeds") && fs::is_directory("seeds")) {
                return "seeds";
            }
            if (fs::exists("../seeds") && fs::is_directory("../seeds")) {
                return "../seeds";
            }
            if (!fs::exists("CMakeLists.txt") && fs::exists("../CMakeLists.txt")) {
                return "../seeds";
            }
        } catch (...) {}
        return "seeds";
    }

    /// Searches for an existing seed file for system size n across known naming patterns and directories
    static std::string find_seed_file(const std::string& configured_dir, int n) {
        namespace fs = std::filesystem;
        std::string n_str = std::to_string(n);

        // Candidate search directories in priority order
        std::vector<std::string> search_dirs;
        if (!configured_dir.empty()) {
            search_dirs.push_back(configured_dir);
            search_dirs.push_back("../" + configured_dir);
        }
        search_dirs.push_back("seeds");
        search_dirs.push_back("../seeds");
        search_dirs.push_back("data_raw/minima");
        search_dirs.push_back("../data_raw/minima");

        // Filename patterns to test for system size n
        std::vector<std::string> patterns = {
            "seeds_N" + n_str + ".csv",
            "seeds_n" + n_str + ".csv",
            "minima_n" + n_str + ".csv",
            "minima_N" + n_str + ".csv",
            "seeds_N" + n_str + ".txt",
            "seeds_n" + n_str + ".txt",
            "N" + n_str + ".csv",
            "n" + n_str + ".csv",
            n_str + ".csv",
            "N" + n_str + ".txt",
            "n" + n_str + ".txt",
            n_str + ".txt"
        };

        for (const auto& dir : search_dirs) {
            try {
                if (!fs::exists(dir) || !fs::is_directory(dir)) continue;
                for (const auto& pattern : patterns) {
                    fs::path candidate = fs::path(dir) / pattern;
                    if (fs::exists(candidate) && fs::is_regular_file(candidate) && fs::file_size(candidate) > 0) {
                        return candidate.string();
                    }
                }
            } catch (...) {
                continue;
            }
        }
        return "";
    }

    // ------------------------------------------------------------------------
    // Step 1: Discover or Load Traps (Local Minima Basins)
    // ------------------------------------------------------------------------
    std::vector<MinimumFamily> prepare_traps(
        int n,
        const TrapsConfig& traps_cfg,
        uint64_t seed,
        bool use_gpu = true,
        bool verbose = true) const
    {
        std::vector<MinimumFamily> families;
        bool cache_loaded = false;

        std::string seed_file = (!traps_cfg.force_regenerate_traps)
            ? find_seed_file(traps_cfg.minima_cache_dir, n)
            : "";

        if (!seed_file.empty()) {
            if (verbose) {
                std::cout << "[Connector] Loading cached local minima from: " << seed_file << "\n";
            }
            families = ClassicalLogger::load_from_csv(seed_file, n);
            if (families.size() >= traps_cfg.benchmark_trap_count) {
                cache_loaded = true;
            } else if (verbose) {
                std::cout << "[Connector] Cache has only " << families.size()
                          << " traps, but " << traps_cfg.benchmark_trap_count
                          << " requested. Regenerating...\n";
            }
        }

        if (!cache_loaded) {
            size_t target_pool = (n <= 16) 
                ? std::min<size_t>(traps_cfg.pool_minima_count, 100) 
                : traps_cfg.pool_minima_count;

            if (verbose) {
                std::cout << "[Connector] Discovering " << target_pool
                          << " distinct symmetry-reduced minima (N = " << n << ")...\n";
            }

            ClassicalOptimizer opt(n, seed);
            size_t max_starts = (n < 20) ? std::min<size_t>(20000, 1ULL << n) : 100000;
#if HAS_CUDA_CLASSICAL
            if (use_gpu) {
                families = opt.collect_distinct_minima_gpu(
                    target_pool, 4096, max_starts,
                    verbose ? LogLevel::Summary : LogLevel::Silent
                );
            } else
#endif
            {
                (void)use_gpu;
                families = opt.collect_distinct_minima_cpu(
                    target_pool, max_starts,
                    verbose ? LogLevel::Summary : LogLevel::Silent
                );
            }

            std::string seeds_dir = resolve_seeds_dir(traps_cfg.minima_cache_dir);
            try {
                std::filesystem::create_directories(seeds_dir);
            } catch (...) {}

            std::string save_path = (std::filesystem::path(seeds_dir) /
                                     ("seeds_N" + std::to_string(n) + ".csv")).string();

            ClassicalRunMetadata meta;
            meta.n = n;
            meta.seed = seed;
            meta.algorithm_name = "Minima Discovery";
            meta.total_starts = families.size();
            meta.total_evals = 0;
            meta.device_name = use_gpu ? "GPU" : "CPU";
            ClassicalLogger::export_to_csv(save_path, families, meta);
            if (verbose) {
                std::cout << "[Connector] Exported discovered minima to: " << save_path << "\n";
            }
        }

        // Sort ascending by energy (deepest traps first)
        std::sort(families.begin(), families.end(), [](const MinimumFamily& a, const MinimumFamily& b) {
            if (a.energy != b.energy) return a.energy < b.energy;
            return a.times_visited > b.times_visited;
        });

        size_t take_count = std::min(traps_cfg.benchmark_trap_count, families.size());
        families.resize(take_count);
        return families;
    }

    // ------------------------------------------------------------------------
    // Step 2: Run Classical Baselines (Fixed-Trial Protocol: Kick+SD & Restart-SD)
    // ------------------------------------------------------------------------
    void run_classical_baselines(
        SingleTrapBenchmark& bench,
        ClassicalOptimizer& opt,
        const ClassicalConfig& c_cfg,
        bool use_gpu = true,
        bool verbose = true) const
    {
        auto t_start_kick = std::chrono::high_resolution_clock::now();

        int best_d = -1;
        double max_rate = -1.0;

        // Sweep d in kick_d_values
        for (int d : c_cfg.kick_d_values) {
            MultiStartResult kick_res = opt.run_kick_trials(
                bench.trap_state, d, c_cfg.kick_trials_per_d,
                c_cfg.batch_size, use_gpu, c_cfg.max_evals_per_start, verbose
            );

            bench.rate_by_d[d] = kick_res.per_eval_rate;
            bench.successes_by_d[d] = kick_res.improved_hits;
            bench.trials_by_d[d] = kick_res.total_starts;
            bench.evals_by_d[d] = kick_res.total_evals;

            // Choose d* from these population averages
            if (kick_res.per_eval_rate > max_rate) {
                max_rate = kick_res.per_eval_rate;
                best_d = d;
            }
        }

        bench.best_d = (best_d != -1) ? best_d : (!c_cfg.kick_d_values.empty() ? c_cfg.kick_d_values[0] : 1);
        bench.best_d_rate = (best_d != -1) ? bench.rate_by_d[best_d] : 0.0;
        bench.best_d_successes = (best_d != -1) ? bench.successes_by_d[best_d] : 0;
        bench.best_d_trials = (best_d != -1) ? bench.trials_by_d[best_d] : 0;
        bench.best_d_evals = (best_d != -1) ? bench.evals_by_d[best_d] : 0;

        auto t_end_kick = std::chrono::high_resolution_clock::now();
        bench.kick_time_sec = std::chrono::duration<double>(t_end_kick - t_start_kick).count();

        // Restart-SD
        auto t_start_restart = std::chrono::high_resolution_clock::now();
        MultiStartResult restart_res = opt.run_restart_trials(
            bench.trap_state, c_cfg.restart_trials,
            c_cfg.batch_size, use_gpu, c_cfg.max_evals_per_start, verbose
        );

        bench.restart_rate = restart_res.per_eval_rate;
        bench.restart_successes = restart_res.improved_hits;
        bench.restart_trials = restart_res.total_starts;
        bench.restart_evals = restart_res.total_evals;

        auto t_end_restart = std::chrono::high_resolution_clock::now();
        bench.restart_time_sec = std::chrono::duration<double>(t_end_restart - t_start_restart).count();
    }

    // ------------------------------------------------------------------------
    // Step 3: Run Quantum Dynamics & Noise-Free Window-Averaged P_<
    // ------------------------------------------------------------------------
    void run_quantum_simulation(
        SingleTrapBenchmark& bench,
        const QuantumConfig& q_cfg,
        int n,
        double g,
        uint64_t seed,
        bool verbose = true) const
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        SamplerConfig s_cfg;
        s_cfg.n = n;
        s_cfg.g = g;
        s_cfg.dt = q_cfg.dt;
        s_cfg.t_max = q_cfg.t_max();
        s_cfg.num_sample_points = q_cfg.k_sample_points;
        s_cfg.shots_per_point = q_cfg.m_shots_per_point;
        s_cfg.seed = seed;
        s_cfg.t_start = q_cfg.t_start;
        s_cfg.dt_sample = q_cfg.dt_sample;
        s_cfg.track_p_less = q_cfg.track_p_less;
        s_cfg.verbose = verbose;
        s_cfg.record_trajectory = q_cfg.record_trajectory;
        s_cfg.dt_trajectory = q_cfg.dt_trajectory;

#if HAS_CUSTATEVEC
        auto sampler = create_sampler(n);
        SamplingResult res = sampler->sample(bench.trap_state, s_cfg);

        bench.time_points = std::move(res.time_points);
        bench.p_less_history = std::move(res.p_less_history);
        bench.dh_history = std::move(res.dh_history);
        bench.window_avg_dh = res.window_avg_dh;
        bench.initial_rise_rate_avg = res.initial_rise_rate_avg;
        bench.initial_rise_rate_max = res.initial_rise_rate_max;
        if (bench.d_min == 0) bench.d_min = res.d_min;
        bench.quantum_improved_shots = res.improved_shots;
        bench.quantum_total_shots = res.total_shots_collected;
        bench.quantum_success_rate = res.success_rate;
        bench.quantum_per_eval_rate = res.per_eval_rate;
        bench.quantum_basis_size = res.unique_basis.size();
        bench.p_blind = res.p_blind;
        bench.peak_p_less = res.peak_p_less;
        bench.peak_time = res.peak_time;
#else
        // CPU fallback for testing / simulation without GPU cuQuantum
        if (n <= 12) {
            Diagonalizer diag;
            DiagConfig d_cfg;
            int64_t dim = 1LL << n;
            d_cfg.method = DiagMethod::ExactDense;
            d_cfg.num_eigenpairs = static_cast<int>(dim);
            d_cfg.normalize_by_n = false;

            Eigensolution sol = diag.solve_fullspace_dense(n, g, d_cfg);
            LabsEvaluator eval(n);

            if (bench.d_min == 0) {
                bench.d_min = eval.compute_d_min(bench.trap_state, bench.trap_energy);
            }

            // Precompute LABS energies for all basis states
            std::vector<int64_t> energies(dim);
            int64_t count_improved = 0;
            for (int64_t s = 0; s < dim; ++s) {
                energies[s] = eval.calcEnergy(SpinState{static_cast<uint64_t>(s)});
                if (energies[s] < bench.trap_energy) {
                    count_improved++;
                }
            }
            bench.p_blind = static_cast<double>(count_improved) / static_cast<double>(dim);

            int64_t s0 = static_cast<int64_t>(bench.trap_state.bits);
            std::vector<double> c_coeff(dim, 0.0);
            for (int64_t j = 0; j < dim && j < static_cast<int64_t>(sol.pairs.size()); ++j) {
                c_coeff[j] = sol.pairs[j].state_vector[s0];
            }

            bench.time_points = q_cfg.sample_times();
            bench.p_less_history.resize(bench.time_points.size(), 0.0);
            bench.dh_history.resize(bench.time_points.size(), 0.0);
            double peak_p = 0.0;
            double peak_t = 0.0;

            for (size_t ti = 0; ti < bench.time_points.size(); ++ti) {
                double t = bench.time_points[ti];
                double p_less = 0.0;
                double dh_val = 0.0;

                for (int64_t s = 0; s < dim; ++s) {
                    double re = 0.0, im = 0.0;
                    for (int64_t j = 0; j < dim && j < static_cast<int64_t>(sol.pairs.size()); ++j) {
                        double angle = -sol.pairs[j].energy * t;
                        double v_sj = sol.pairs[j].state_vector[s];
                        double ampl = c_coeff[j] * v_sj;
                        re += ampl * std::cos(angle);
                        im += ampl * std::sin(angle);
                    }
                    double ampl_sq = (re * re + im * im);
                    if (energies[s] < bench.trap_energy) {
                        p_less += ampl_sq;
                    }
                    int flips = __builtin_popcountll(static_cast<uint64_t>(s ^ s0));
                    dh_val += ampl_sq * static_cast<double>(flips);
                }
                bench.p_less_history[ti] = p_less;
                bench.dh_history[ti] = dh_val;
                if (p_less > peak_p) {
                    peak_p = p_less;
                    peak_t = t;
                }
            }
            bench.peak_p_less = peak_p;
            bench.peak_time = peak_t;

            // Compute initial rise rates:
            // average finite difference slope while monotonically increasing: P_<(t_peak1) / t_peak1
            // and max instantaneous slope
            if (bench.p_less_history.size() >= 2) {
                size_t first_drop = bench.p_less_history.size() - 1;
                for (size_t i = 1; i < bench.p_less_history.size(); ++i) {
                    if (bench.p_less_history[i] < bench.p_less_history[i - 1]) {
                        first_drop = i - 1;
                        break;
                    }
                }
                double t_p1 = bench.time_points[first_drop];
                double p_p1 = bench.p_less_history[first_drop];
                bench.initial_rise_rate_avg = (t_p1 > 0.0) ? (p_p1 / t_p1) : 0.0;

                double max_slope = 0.0;
                for (size_t i = 0; i < first_drop; ++i) {
                    double dt_step = bench.time_points[i + 1] - bench.time_points[i];
                    if (dt_step > 1e-9) {
                        double slope = (bench.p_less_history[i + 1] - bench.p_less_history[i]) / dt_step;
                        if (slope > max_slope) max_slope = slope;
                    }
                }
                bench.initial_rise_rate_max = max_slope;
            }

            // Window avg dH
            double avg_start = q_cfg.record_trajectory ? 0.0 : q_cfg.t_start;
            double sum_dh = 0.0;
            size_t count_dh = 0;
            for (size_t i = 0; i < bench.time_points.size() && i < bench.dh_history.size(); ++i) {
                if (bench.time_points[i] >= avg_start - 1e-9) {
                    sum_dh += bench.dh_history[i];
                    count_dh++;
                }
            }
            bench.window_avg_dh = (count_dh > 0) ? (sum_dh / count_dh) : (bench.dh_history.empty() ? 0.0 : bench.dh_history.back());
        } else {
            if (verbose) {
                std::cout << "[Connector Notice] cuStateVec not available in CPU build. "
                          << "Skipping full-space quantum evolution for N = " << n << ".\n";
            }
            bench.time_points = q_cfg.sample_times();
            bench.p_less_history.assign(bench.time_points.size(), 0.0);
            bench.dh_history.assign(bench.time_points.size(), 0.0);
        }
#endif

        // Exact window-averaged P_<: mean over t in [t_start, t_max] (or [0, t_max] if record_trajectory)
        double avg_start = q_cfg.record_trajectory ? 0.0 : q_cfg.t_start;
        double sum_p = 0.0;
        size_t count = 0;
        for (size_t i = 0; i < bench.time_points.size(); ++i) {
            if (bench.time_points[i] >= avg_start - 1e-9) {
                sum_p += bench.p_less_history[i];
                count++;
            }
        }
        bench.window_avg_p_less = (count > 0) ? (sum_p / count) : 0.0;

        auto t_end = std::chrono::high_resolution_clock::now();
        bench.quantum_time_sec = std::chrono::duration<double>(t_end - t_start).count();
    }

    // ------------------------------------------------------------------------
    // Step 4: CSV Export & Logging
    // ------------------------------------------------------------------------
    void export_results_csv(
        const std::string& path,
        const std::vector<SingleTrapBenchmark>& results) const
    {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("BenchmarkConnector: Cannot open results CSV for writing: " + path);
        }

        file << "N,g,trap_rank,trap_energy,d_min,window_avg_p_less,window_avg_dh,"
             << "initial_rise_rate_avg,initial_rise_rate_max,peak_p_less,peak_time,p_blind,"
             << "best_d,best_d_rate,best_d_successes,best_d_trials,best_d_evals,"
             << "restart_rate,restart_successes,restart_trials,restart_evals,"
             << "quantum_time_sec,kick_time_sec,restart_time_sec,trap_bits\n";

        for (const auto& r : results) {
            file << r.n << ","
                 << r.g << ","
                 << r.trap_rank << ","
                 << r.trap_energy << ","
                 << r.d_min << ","
                 << std::scientific << std::setprecision(6) << r.window_avg_p_less << ","
                 << std::fixed << std::setprecision(4) << r.window_avg_dh << ","
                 << std::scientific << std::setprecision(6) << r.initial_rise_rate_avg << ","
                 << std::scientific << std::setprecision(6) << r.initial_rise_rate_max << ","
                 << r.peak_p_less << ","
                 << std::fixed << std::setprecision(4) << r.peak_time << ","
                 << std::scientific << std::setprecision(6) << r.p_blind << ","
                 << r.best_d << ","
                 << r.best_d_rate << ","
                 << r.best_d_successes << ","
                 << r.best_d_trials << ","
                 << r.best_d_evals << ","
                 << r.restart_rate << ","
                 << r.restart_successes << ","
                 << r.restart_trials << ","
                 << r.restart_evals << ","
                 << std::fixed << std::setprecision(4) << r.quantum_time_sec << ","
                 << r.kick_time_sec << ","
                 << r.restart_time_sec << ","
                 << "0x" << std::hex << std::uppercase << r.trap_state.bits << std::dec << "\n";
        }
    }

    void export_time_series_csv(
        const std::string& path,
        const std::vector<SingleTrapBenchmark>& results) const
    {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("BenchmarkConnector: Cannot open time series CSV: " + path);
        }

        file << "N,g,trap_rank,trap_energy,time,p_less,dH,p_blind\n";
        for (const auto& r : results) {
            for (size_t i = 0; i < r.time_points.size() && i < r.p_less_history.size(); ++i) {
                double dh = (i < r.dh_history.size()) ? r.dh_history[i] : 0.0;
                file << r.n << ","
                     << r.g << ","
                     << r.trap_rank << ","
                     << r.trap_energy << ","
                     << std::fixed << std::setprecision(4) << r.time_points[i] << ","
                     << std::scientific << std::setprecision(6) << r.p_less_history[i] << ","
                     << std::fixed << std::setprecision(4) << dh << ","
                     << std::scientific << std::setprecision(6) << r.p_blind << "\n";
            }
        }
    }

    void export_structured_json(
        const std::string& path,
        const std::vector<SingleTrapBenchmark>& results,
        int n_override = 0) const
    {
        std::filesystem::path p(path);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }

        std::ofstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("BenchmarkConnector: Cannot open structured JSON for writing: " + path);
        }

        // Group results by (n, trap_rank)
        std::map<std::pair<int, size_t>, std::vector<SingleTrapBenchmark>> grouped_traps;
        for (const auto& r : results) {
            grouped_traps[{r.n, r.trap_rank}].push_back(r);
        }

        int sys_n = n_override;
        if (sys_n == 0 && !results.empty()) {
            sys_n = results.front().n;
        }

        file << "{\n";
        file << "  \"system\": {\n";
        file << "    \"N\": " << sys_n << ",\n";
        file << "    \"num_seeds\": " << grouped_traps.size() << "\n";
        file << "  },\n";
        file << "  \"seeds\": [\n";

        size_t trap_idx = 0;
        for (auto it = grouped_traps.begin(); it != grouped_traps.end(); ++it, ++trap_idx) {
            const auto& trap_list = it->second;
            if (trap_list.empty()) continue;
            const auto& first = trap_list.front();

            file << "    {\n";
            file << "      \"N\": " << first.n << ",\n";
            file << "      \"trap_rank\": " << first.trap_rank << ",\n";
            file << "      \"trap_state_hex\": \"0x" << std::hex << std::uppercase << first.trap_state.bits << std::dec << "\",\n";
            file << "      \"trap_energy_E0\": " << first.trap_energy << ",\n";
            file << "      \"d_min\": " << first.d_min << ",\n";

            // Classical baselines
            file << "      \"classical_baselines\": {\n";
            file << "        \"best_d\": " << first.best_d << ",\n";
            file << "        \"best_d_rate\": " << std::scientific << std::setprecision(6) << first.best_d_rate << ",\n";
            file << "        \"best_d_successes\": " << first.best_d_successes << ",\n";
            file << "        \"best_d_trials\": " << first.best_d_trials << ",\n";
            file << "        \"best_d_evals\": " << first.best_d_evals << ",\n";

            file << "        \"rates_by_d\": {";
            size_t di = 0;
            for (const auto& kv : first.rate_by_d) {
                file << (di > 0 ? ", " : "") << "\"" << kv.first << "\": " << std::scientific << std::setprecision(6) << kv.second;
                di++;
            }
            file << "},\n";

            file << "        \"trials_by_d\": {";
            di = 0;
            for (const auto& kv : first.trials_by_d) {
                file << (di > 0 ? ", " : "") << "\"" << kv.first << "\": " << kv.second;
                di++;
            }
            file << "},\n";

            file << "        \"evals_by_d\": {";
            di = 0;
            for (const auto& kv : first.evals_by_d) {
                file << (di > 0 ? ", " : "") << "\"" << kv.first << "\": " << kv.second;
                di++;
            }
            file << "},\n";

            file << "        \"restart_rate\": " << std::scientific << std::setprecision(6) << first.restart_rate << ",\n";
            file << "        \"restart_successes\": " << first.restart_successes << ",\n";
            file << "        \"restart_trials\": " << first.restart_trials << ",\n";
            file << "        \"restart_evals\": " << first.restart_evals << ",\n";
            file << "        \"kick_time_sec\": " << std::fixed << std::setprecision(4) << first.kick_time_sec << ",\n";
            file << "        \"restart_time_sec\": " << std::fixed << std::setprecision(4) << first.restart_time_sec << "\n";
            file << "      },\n";

            // Quantum dynamics by g
            file << "      \"quantum_dynamics\": {\n";
            for (size_t gi = 0; gi < trap_list.size(); ++gi) {
                const auto& q = trap_list[gi];
                std::ostringstream g_str;
                g_str << std::fixed << std::setprecision(2) << q.g;

                file << "        \"" << g_str.str() << "\": {\n";
                file << "          \"g\": " << std::fixed << std::setprecision(2) << q.g << ",\n";
                file << "          \"window_avg_p_less\": " << std::scientific << std::setprecision(6) << q.window_avg_p_less << ",\n";
                file << "          \"window_avg_dH\": " << std::fixed << std::setprecision(4) << q.window_avg_dh << ",\n";
                file << "          \"initial_rise_rate_avg\": " << std::scientific << std::setprecision(6) << q.initial_rise_rate_avg << ",\n";
                file << "          \"initial_rise_rate_max\": " << std::scientific << std::setprecision(6) << q.initial_rise_rate_max << ",\n";
                file << "          \"peak_p_less\": " << std::scientific << std::setprecision(6) << q.peak_p_less << ",\n";
                file << "          \"peak_time\": " << std::fixed << std::setprecision(4) << q.peak_time << ",\n";
                file << "          \"p_blind\": " << std::scientific << std::setprecision(6) << q.p_blind << ",\n";

                // Time array
                file << "          \"time_array\": [";
                for (size_t ti = 0; ti < q.time_points.size(); ++ti) {
                    file << (ti > 0 ? ", " : "") << std::fixed << std::setprecision(4) << q.time_points[ti];
                }
                file << "],\n";

                // P_less array
                file << "          \"p_less_array\": [";
                for (size_t ti = 0; ti < q.p_less_history.size(); ++ti) {
                    file << (ti > 0 ? ", " : "") << std::scientific << std::setprecision(6) << q.p_less_history[ti];
                }
                file << "],\n";

                // dH array
                file << "          \"dH_array\": [";
                for (size_t ti = 0; ti < q.dh_history.size(); ++ti) {
                    file << (ti > 0 ? ", " : "") << std::fixed << std::setprecision(4) << q.dh_history[ti];
                }
                file << "],\n";

                file << "          \"quantum_time_sec\": " << std::fixed << std::setprecision(4) << q.quantum_time_sec << "\n";
                file << "        }" << (gi + 1 < trap_list.size() ? "," : "") << "\n";
            }
            file << "      }\n";

            file << "    }" << (trap_idx + 1 < grouped_traps.size() ? "," : "") << "\n";
        }
        file << "  ]\n";
        file << "}\n";
    }

    void print_comparison_table(
        const std::vector<SingleTrapBenchmark>& results,
        std::ostream& os = std::cout) const
    {
        os << "\n========================================================================================================================================\n";
        os << "                                            BERNASCONI-KRYLOV BENCHMARK SUMMARY TABLE                                                   \n";
        os << "========================================================================================================================================\n";
        os << " N  |  g   | Trap (E0) | Window <P_<> | Best d* | Kick Rate r(d*) | Restart Rate | Quantum Time | Kick Time  | Restart Time\n";
        os << "----+------+-----------+--------------+---------+-----------------+--------------+--------------+------------+-------------\n";

        // Group by (n, g)
        std::map<std::pair<int, double>, std::vector<SingleTrapBenchmark>> grouped;
        for (const auto& r : results) {
            grouped[{r.n, r.g}].push_back(r);
        }

        for (const auto& kv : grouped) {
            int n = kv.first.first;
            double g = kv.first.second;
            const auto& trap_list = kv.second;

            double sum_pless = 0.0;
            double sum_krate = 0.0;
            double sum_rrate = 0.0;
            double sum_qtime = 0.0;
            double sum_ktime = 0.0;
            double sum_rtime = 0.0;

            for (const auto& r : trap_list) {
                sum_pless += r.window_avg_p_less;
                sum_krate += r.best_d_rate;
                sum_rrate += r.restart_rate;
                sum_qtime += r.quantum_time_sec;
                sum_ktime += r.kick_time_sec;
                sum_rtime += r.restart_time_sec;

                std::ostringstream ss_trap, ss_pless, ss_krate, ss_rrate;
                ss_trap << "#" << r.trap_rank << " (" << r.trap_energy << ")";
                ss_pless << std::scientific << std::setprecision(3) << r.window_avg_p_less;
                ss_krate << std::scientific << std::setprecision(3) << r.best_d_rate;
                ss_rrate << std::scientific << std::setprecision(3) << r.restart_rate;

                os << " " << std::setw(2) << r.n << " | "
                   << std::setw(4) << std::fixed << std::setprecision(2) << r.g << " | "
                   << std::setw(9) << ss_trap.str() << " | "
                   << std::setw(12) << ss_pless.str() << " | "
                   << std::setw(7) << r.best_d << " | "
                   << std::setw(15) << ss_krate.str() << " | "
                   << std::setw(12) << ss_rrate.str() << " | "
                   << std::setw(10) << std::fixed << std::setprecision(2) << r.quantum_time_sec << "s | "
                   << std::setw(8) << std::fixed << std::setprecision(2) << r.kick_time_sec << "s | "
                   << std::setw(9) << std::fixed << std::setprecision(2) << r.restart_time_sec << "s\n";
            }

            if (trap_list.size() > 1) {
                double count = static_cast<double>(trap_list.size());
                std::ostringstream ss_mean_pless, ss_mean_krate, ss_mean_rrate;
                ss_mean_pless << std::scientific << std::setprecision(3) << (sum_pless / count);
                ss_mean_krate << std::scientific << std::setprecision(3) << (sum_krate / count);
                ss_mean_rrate << std::scientific << std::setprecision(3) << (sum_rrate / count);

                os << "----+------+-----------+--------------+---------+-----------------+--------------+--------------+------------+-------------\n";
                os << " " << std::setw(2) << n << " | "
                   << std::setw(4) << std::fixed << std::setprecision(2) << g << " | "
                   << std::setw(9) << "MEAN" << " | "
                   << std::setw(12) << ss_mean_pless.str() << " | "
                   << std::setw(7) << "---" << " | "
                   << std::setw(15) << ss_mean_krate.str() << " | "
                   << std::setw(12) << ss_mean_rrate.str() << " | "
                   << std::setw(10) << std::fixed << std::setprecision(2) << (sum_qtime / count) << "s | "
                   << std::setw(8) << std::fixed << std::setprecision(2) << (sum_ktime / count) << "s | "
                   << std::setw(9) << std::fixed << std::setprecision(2) << (sum_rtime / count) << "s\n";
                os << "----+------+-----------+--------------+---------+-----------------+--------------+--------------+------------+-------------\n";
            }
        }
        os << "========================================================================================================================================\n\n";
    }

    // ------------------------------------------------------------------------
    // Step 5: Full Benchmark Execution Pipeline
    // ------------------------------------------------------------------------
    std::vector<SingleTrapBenchmark> run(const BenchmarkConfig& cfg) const {
        if (cfg.output.verbose) {
            cfg.print();
        }

        bool use_gpu = (cfg.system.device == "gpu" || cfg.system.device == "auto");
        std::vector<SingleTrapBenchmark> all_results;

        for (int n : cfg.system.n_values) {
            if (cfg.output.verbose) {
                std::cout << "\n>>> [Connector] Starting Benchmark for System Size N = " << n << " <<<\n";
            }

            // 1. Prepare traps
            std::vector<MinimumFamily> traps = prepare_traps(
                n, cfg.traps, cfg.system.seed_base, use_gpu, cfg.output.verbose
            );

            LabsEvaluator eval(n);
            ClassicalOptimizer opt(n, cfg.system.seed_base);
            std::vector<SingleTrapBenchmark> n_results;

            // 2. Evaluate each trap
            for (size_t rank = 0; rank < traps.size(); ++rank) {
                const auto& trap_fam = traps[rank];

                // Run classical baselines ONCE per trap (independent of quantum field g)
                SingleTrapBenchmark base_bench;
                base_bench.n = n;
                base_bench.trap_rank = rank + 1;
                base_bench.trap_state = trap_fam.sample_state;
                base_bench.trap_energy = trap_fam.energy;
                base_bench.d_min = eval.compute_d_min(base_bench.trap_state, base_bench.trap_energy);

                bool should_run_classical = cfg.quantum.run_classical && 
                    (cfg.classical.kick_trials_per_d > 0 || cfg.classical.restart_trials > 0);

                if (should_run_classical) {
                    if (cfg.output.verbose) {
                        std::cout << "\n======================================================================\n";
                        std::cout << " [Classical Baselines] Benchmarking Trap #" << base_bench.trap_rank
                                  << " (E0 = " << base_bench.trap_energy << ", d_min = " << base_bench.d_min << ") for N = " << n << "\n";
                        std::cout << "======================================================================\n";
                    }
                    run_classical_baselines(base_bench, opt, cfg.classical, use_gpu, cfg.output.verbose);
                } else if (cfg.output.verbose) {
                    std::cout << "\n [Classical Baselines] Skipping for Trap #" << base_bench.trap_rank 
                              << " (run_classical = false)\n";
                }

                // Now evaluate each quantum transverse field g, reusing precomputed classical stats
                for (double g : cfg.quantum.g_values) {
                    SingleTrapBenchmark bench = base_bench;
                    bench.g = g;

                    if (cfg.output.verbose) {
                        std::cout << "\n----------------------------------------------------------------------\n";
                        std::cout << " [Quantum Dynamics] Simulating Trap #" << bench.trap_rank
                                  << " (E0 = " << bench.trap_energy << ", d_min = " << bench.d_min << ") with N = " << n << ", g = " << g << "\n";
                        std::cout << "----------------------------------------------------------------------\n";
                    }

                    // Quantum Dynamics (Exact noise-free window-averaged P_<)
                    run_quantum_simulation(bench, cfg.quantum, n, g, cfg.system.seed_base, cfg.output.verbose);

                    n_results.push_back(bench);
                    all_results.push_back(bench);
                }
            }

            // 3. Export outputs for this system size N
            std::string res_path = cfg.output.get_results_path(n);
            std::string dyn_path = cfg.output.get_time_series_path(n);
            std::string json_path = cfg.output.get_structured_json_path(n);

            if (cfg.output.verbose) {
                std::cout << "[Connector] Exporting N=" << n << " results to: " << res_path << "\n";
                std::cout << "[Connector] Exporting N=" << n << " dynamics to: " << dyn_path << "\n";
                std::cout << "[Connector] Exporting N=" << n << " structured JSON to: " << json_path << "\n";
            }

            export_results_csv(res_path, n_results);
            export_time_series_csv(dyn_path, n_results);
            export_structured_json(json_path, n_results, n);
        }

        // Export overall summary
        std::string summary_path = cfg.output.get_summary_path();
        std::string all_json_path = cfg.output.get_all_structured_json_path();
        if (cfg.output.verbose) {
            std::cout << "[Connector] Exporting overall summary to: " << summary_path << "\n";
            std::cout << "[Connector] Exporting all structured JSON to: " << all_json_path << "\n";
            print_comparison_table(all_results);
        }
        export_results_csv(summary_path, all_results);
        export_structured_json(all_json_path, all_results, 0);

        return all_results;
    }

    // Convenience overload: load from file and run
    std::vector<SingleTrapBenchmark> run(const std::string& config_file) const {
        BenchmarkConfig cfg = BenchmarkConfig::load_from_file(config_file);
        return run(cfg);
    }
};
