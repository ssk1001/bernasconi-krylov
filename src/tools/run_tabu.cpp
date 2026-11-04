// ============================================================================
// Standalone Local Tabu Search Benchmark CLI
// Evaluates Monte Carlo escape probabilities per evaluation across 50,000 trials
// for Low-Autocorrelation Binary Sequence (LABS) / Bernasconi spin model.
// ============================================================================

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>

#include "../labs_energy/labs_energy.hpp"
#include "../classical/tabu_search.hpp"
#include "../classical/classical.cuh"
#include <map>

namespace fs = std::filesystem;

struct CliArgs {
    int n = 20;
    size_t trials = 50000;
    size_t max_steps = 100;
    uint64_t seed = 42;
    std::string seeds_file = "";
    int rank = 1;
    bool all_ranks = false;
    size_t max_ranks = 10;
    std::vector<int> kick_d = {2, 4, 6, 8};
    int fixed_tenure = 0;
    bool compare_sd = true;
    std::string output_csv = "";
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --n <int>           System size N (default: 20)\n"
              << "  --trials <int>      Number of Monte Carlo trials per method (default: 50000)\n"
              << "  --steps <int>       Max Tabu steps per trial (default: 100)\n"
              << "  --seed <int>        RNG seed (default: 42)\n"
              << "  --seeds-file <file> Path to CSV file containing minima seeds\n"
              << "  --rank <int>        Trap rank to benchmark (1 = deepest trap, default: 1)\n"
              << "  --all-ranks         Benchmark all available top ranks (up to --max-ranks)\n"
              << "  --max-ranks <int>   Maximum ranks to benchmark with --all-ranks (default: 10)\n"
              << "  --kick-d <list>     Comma-separated list of kick distances d (default: 2,4,6,8)\n"
              << "  --tenure <int>      Fixed tabu tenure (default: 0 = dynamic randomized tenure)\n"
              << "  --no-compare-sd     Disable comparison with Steepest Descent baselines\n"
              << "  --output-csv <file> Path to export comparative results CSV\n"
              << "  -h, --help          Show this help message\n";
}

CliArgs parse_args(int argc, char* argv[]) {
    CliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) {
            args.n = std::stoi(argv[++i]);
        } else if (arg == "--trials" && i + 1 < argc) {
            args.trials = std::stoull(argv[++i]);
        } else if (arg == "--steps" && i + 1 < argc) {
            args.max_steps = std::stoull(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            args.seed = std::stoull(argv[++i]);
        } else if (arg == "--seeds-file" && i + 1 < argc) {
            args.seeds_file = argv[++i];
        } else if (arg == "--rank" && i + 1 < argc) {
            args.rank = std::stoi(argv[++i]);
        } else if (arg == "--all-ranks") {
            args.all_ranks = true;
        } else if (arg == "--max-ranks" && i + 1 < argc) {
            args.max_ranks = std::stoull(argv[++i]);
        } else if (arg == "--tenure" && i + 1 < argc) {
            args.fixed_tenure = std::stoi(argv[++i]);
        } else if (arg == "--no-compare-sd") {
            args.compare_sd = false;
        } else if (arg == "--output-csv" && i + 1 < argc) {
            args.output_csv = argv[++i];
        } else if (arg == "--kick-d" && i + 1 < argc) {
            std::string d_str = argv[++i];
            std::stringstream ss(d_str);
            std::string item;
            args.kick_d.clear();
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) args.kick_d.push_back(std::stoi(item));
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    return args;
}

struct BenchmarkTrap {
    int rank = 1;
    int n = 0;
    int64_t energy = 0;
    int d_min = 0;
    SpinState state{0};
    std::string source = "";
};

std::string find_seed_file(int n, const std::string& explicit_path) {
    if (!explicit_path.empty() && fs::exists(explicit_path)) return explicit_path;

    std::vector<std::string> candidates = {
        "data_processed/benchmark_results_n" + std::to_string(n) + ".csv",
        "seeds/seeds_N" + std::to_string(n) + ".csv",
        "seeds/seeds_n" + std::to_string(n) + ".csv",
        "../seeds/seeds_N" + std::to_string(n) + ".csv",
        "data_raw/minima/minima_n" + std::to_string(n) + ".csv"
    };

    for (const auto& c : candidates) {
        if (fs::exists(c) && fs::file_size(c) > 0) return c;
    }
    return "";
}

std::vector<BenchmarkTrap> load_benchmark_traps(int n, const std::string& seed_path, const LabsEvaluator& eval, ClassicalOptimizer& opt, uint64_t seed) {
    std::vector<BenchmarkTrap> traps;

    if (!seed_path.empty() && fs::exists(seed_path)) {
        // Check if file is benchmark_results CSV with trap_bits & trap_rank
        std::ifstream in(seed_path);
        std::string first_line;
        if (std::getline(in, first_line)) {
            if (first_line.find("trap_bits") != std::string::npos && first_line.find("trap_rank") != std::string::npos) {
                std::cout << "[Seeds] Loading benchmark traps directly from: " << seed_path << "\n";
                std::stringstream ss(first_line);
                std::string col;
                std::vector<std::string> headers;
                while (std::getline(ss, col, ',')) {
                    // trim
                    col.erase(0, col.find_first_not_of(" \t\r\n"));
                    col.erase(col.find_last_not_of(" \t\r\n") + 1);
                    headers.push_back(col);
                }
                int rank_idx = -1, energy_idx = -1, dmin_idx = -1, bits_idx = -1;
                for (size_t i = 0; i < headers.size(); ++i) {
                    if (headers[i] == "trap_rank") rank_idx = static_cast<int>(i);
                    else if (headers[i] == "trap_energy") energy_idx = static_cast<int>(i);
                    else if (headers[i] == "d_min") dmin_idx = static_cast<int>(i);
                    else if (headers[i] == "trap_bits") bits_idx = static_cast<int>(i);
                }

                std::map<int, BenchmarkTrap> unique_traps;
                std::string line;
                while (std::getline(in, line)) {
                    if (line.empty()) continue;
                    std::stringstream lss(line);
                    std::string val;
                    std::vector<std::string> fields;
                    while (std::getline(lss, val, ',')) {
                        val.erase(0, val.find_first_not_of(" \t\r\n"));
                        val.erase(val.find_last_not_of(" \t\r\n") + 1);
                        fields.push_back(val);
                    }
                    if (rank_idx >= 0 && rank_idx < static_cast<int>(fields.size())) {
                        int rk = std::stoi(fields[rank_idx]);
                        if (unique_traps.find(rk) == unique_traps.end()) {
                            BenchmarkTrap bt;
                            bt.rank = rk;
                            bt.n = n;
                            if (energy_idx >= 0 && energy_idx < static_cast<int>(fields.size())) {
                                bt.energy = std::stoll(fields[energy_idx]);
                            }
                            if (dmin_idx >= 0 && dmin_idx < static_cast<int>(fields.size())) {
                                bt.d_min = std::stoi(fields[dmin_idx]);
                            }
                            if (bits_idx >= 0 && bits_idx < static_cast<int>(fields.size())) {
                                std::string bstr = fields[bits_idx];
                                if (bstr.rfind("0x", 0) == 0 || bstr.rfind("0X", 0) == 0) {
                                    bt.state = SpinState{std::stoull(bstr, nullptr, 16)};
                                } else {
                                    bt.state = SpinState{std::stoull(bstr, nullptr, 2)};
                                }
                            }
                            if (bt.energy == 0) bt.energy = eval.calcEnergy(bt.state);
                            if (bt.d_min <= 0) bt.d_min = eval.compute_d_min(bt.state, bt.energy);
                            bt.source = seed_path;
                            unique_traps[rk] = bt;
                        }
                    }
                }
                for (const auto& kv : unique_traps) {
                    traps.push_back(kv.second);
                }
                std::cout << "[Seeds] Loaded " << traps.size() << " structured benchmark traps from CSV.\n";
                return traps;
            }
        }
        in.close();

        // Standard CSV format (e.g. seeds_N22.csv)
        std::cout << "[Seeds] Loading precomputed minima from: " << seed_path << "\n";
        auto raw_traps = ClassicalLogger::load_from_csv(seed_path, n);
        static const std::unordered_map<int, int64_t> known_opts = {
            {16, 24}, {20, 26}, {22, 39}, {24, 36}, {26, 45}, {28, 50}, {30, 59}
        };
        auto it_opt = known_opts.find(n);
        int64_t opt_e = (it_opt != known_opts.end()) ? it_opt->second : 0;

        int rk = 1;
        for (const auto& t : raw_traps) {
            if (opt_e > 0 && t.energy <= opt_e) continue;
            BenchmarkTrap bt;
            bt.rank = rk++;
            bt.n = n;
            bt.energy = t.energy;
            bt.state = t.sample_state.bits ? t.sample_state : t.canonical_state;
            bt.d_min = eval.compute_d_min(bt.state, bt.energy);
            bt.source = seed_path;
            traps.push_back(bt);
        }
    }

    if (traps.empty()) {
        std::cout << "[Seeds] No precomputed seed file found for N=" << n 
                  << ". Running fast CPU minima collection (target: 20 families)...\n";
        auto discovered = opt.collect_distinct_minima_cpu(20, 50000, LogLevel::Silent);
        if (!discovered.empty()) {
            ClassicalRunMetadata meta;
            meta.n = n;
            meta.seed = seed;
            meta.algorithm_name = "CPU Serial Minima Collector";
            meta.device_name = "Host CPU";
            meta.total_starts = 50000;
            std::string save_path = "seeds/seeds_N" + std::to_string(n) + ".csv";
            ClassicalLogger::export_to_csv(save_path, discovered, meta);
            std::cout << "[Seeds] Saved discovered minima to: " << save_path << "\n";

            int rk = 1;
            for (const auto& t : discovered) {
                BenchmarkTrap bt;
                bt.rank = rk++;
                bt.n = n;
                bt.energy = t.energy;
                bt.state = t.sample_state.bits ? t.sample_state : t.canonical_state;
                bt.d_min = eval.compute_d_min(bt.state, bt.energy);
                bt.source = save_path;
                traps.push_back(bt);
            }
        }
    }
    return traps;
}

int main(int argc, char* argv[]) {
    CliArgs args = parse_args(argc, argv);

    std::cout << "========================================================================================================================\n";
    std::cout << "                    TABU SEARCH LOCAL ESCAPE BENCHMARK (N = " << args.n << ", TRIALS = " << args.trials << ")\n";
    std::cout << "========================================================================================================================\n";

    LabsEvaluator eval(args.n);
    ClassicalOptimizer opt(eval, args.seed);

    // 1. Locate and load traps
    std::string seed_path = find_seed_file(args.n, args.seeds_file);
    std::vector<BenchmarkTrap> traps = load_benchmark_traps(args.n, seed_path, eval, opt, args.seed);

    if (traps.empty()) {
        std::cerr << "[Error] Failed to load or generate local minima traps for N=" << args.n << "\n";
        return 1;
    }

    std::cout << "[Seeds] Available metastable traps for evaluation: " << traps.size() << "\n";

    // Select traps to benchmark
    std::vector<BenchmarkTrap> target_traps;
    if (args.all_ranks) {
        size_t count = std::min(args.max_ranks, traps.size());
        for (size_t i = 0; i < count; ++i) target_traps.push_back(traps[i]);
    } else {
        bool found = false;
        for (const auto& t : traps) {
            if (t.rank == args.rank) {
                target_traps.push_back(t);
                found = true;
                break;
            }
        }
        if (!found && !traps.empty()) {
            size_t r_idx = (args.rank >= 1 && static_cast<size_t>(args.rank) <= traps.size()) ? (args.rank - 1) : 0;
            target_traps.push_back(traps[r_idx]);
        }
    }

    // Setup Tabu configuration
    TabuConfig tabu_cfg;
    tabu_cfg.n = args.n;
    tabu_cfg.max_steps = args.max_steps;
    tabu_cfg.stop_on_escape = true;
    tabu_cfg.seed = args.seed;
    if (args.fixed_tenure > 0) {
        tabu_cfg.dynamic_tenure = false;
        tabu_cfg.tabu_tenure = args.fixed_tenure;
    } else {
        tabu_cfg.dynamic_tenure = true;
    }

    TabuSearch tabu_solver(eval, tabu_cfg);

    std::vector<TabuBenchmarkResult> all_results;
    std::ofstream csv_out;
    if (!args.output_csv.empty()) {
        csv_out.open(args.output_csv);
        csv_out << "trap_rank,n,E0,d_min,method,kick_d,hits,trials,success_rate,total_evals,per_eval_rate,mean_evals_trial,time_sec\n";
    }

    for (size_t t_i = 0; t_i < target_traps.size(); ++t_i) {
        const auto& trap = target_traps[t_i];
        SpinState trap_state = trap.state;
        int64_t e_trap = trap.energy;
        int d_min = trap.d_min;

        std::cout << "\n------------------------------------------------------------------------------------------------------------------------\n";
        std::cout << ">>> BENCHMARKING TRAP #" << trap.rank << " | Energy E_0 = " << e_trap << " | Minimal Barrier d_min = " << d_min << " <<<\n";
        std::cout << "------------------------------------------------------------------------------------------------------------------------\n";

        std::vector<TabuBenchmarkResult> trap_results;

        // 1. Direct Tabu Search (starts at trap s_0, climbs out)
        std::cout << " [1/3] Running Direct Tabu Search (from trap, no kick)... " << std::flush;
        auto res_direct = tabu_solver.run_escape_trials(trap_state, args.trials, &tabu_cfg, false);
        trap_results.push_back(res_direct);
        all_results.push_back(res_direct);
        std::cout << "Done (" << std::fixed << std::setprecision(2) << res_direct.elapsed_sec << " s, "
                  << "Rate r = " << std::scientific << std::setprecision(3) << res_direct.per_eval_rate << ")\n";

        // 2. Kicked Tabu Search across perturbation distances d
        for (int d : args.kick_d) {
            std::cout << " [2/3] Running Kicked Tabu Search (kick d=" << d << ")... " << std::flush;
            auto res_kick = tabu_solver.run_kicked_trials(trap_state, d, args.trials, &tabu_cfg, false);
            trap_results.push_back(res_kick);
            all_results.push_back(res_kick);
            std::cout << "Done (" << std::fixed << std::setprecision(2) << res_kick.elapsed_sec << " s, "
                      << "Rate r = " << std::scientific << std::setprecision(3) << res_kick.per_eval_rate << ")\n";
        }

        // 3. Random Restart Tabu Search
        std::cout << " [3/3] Running Random Restart Tabu Search... " << std::flush;
        auto res_restart = tabu_solver.run_restart_trials(trap_state, args.trials, &tabu_cfg, false);
        trap_results.push_back(res_restart);
        all_results.push_back(res_restart);
        std::cout << "Done (" << std::fixed << std::setprecision(2) << res_restart.elapsed_sec << " s, "
                  << "Rate r = " << std::scientific << std::setprecision(3) << res_restart.per_eval_rate << ")\n";

        // 4. Steepest Descent baselines for side-by-side comparison
        if (args.compare_sd) {
            std::cout << "\n [Comparative Baseline] Running Steepest Descent (SD) Baselines on Host CPU...\n";
            for (int d : args.kick_d) {
                std::cout << "   - SD Kick (d=" << d << ")... " << std::flush;
                auto t0 = std::chrono::high_resolution_clock::now();
                auto sd_res = opt.run_kick_trials(trap_state, d, args.trials, args.trials, false, 0, false);
                auto t1 = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(t1 - t0).count();

                TabuBenchmarkResult sd_bench;
                sd_bench.method_name = "SD Kick (d=" + std::to_string(d) + ")";
                sd_bench.n = args.n;
                sd_bench.d = d;
                sd_bench.og_state = trap_state;
                sd_bench.og_energy = e_trap;
                sd_bench.total_trials = sd_res.total_starts;
                sd_bench.improved_hits = sd_res.improved_hits;
                sd_bench.total_evals = sd_res.total_evals;
                sd_bench.success_rate = sd_res.success_rate;
                sd_bench.per_eval_rate = sd_res.per_eval_rate;
                sd_bench.mean_evals_per_trial = sd_res.total_starts > 0 ? static_cast<double>(sd_res.total_evals) / sd_res.total_starts : 0.0;
                sd_bench.elapsed_sec = sec;
                trap_results.push_back(sd_bench);
                all_results.push_back(sd_bench);
                std::cout << "Done (" << std::fixed << std::setprecision(2) << sec << " s, "
                          << "Rate r = " << std::scientific << std::setprecision(3) << sd_res.per_eval_rate << ")\n";
            }

            // SD Restart
            std::cout << "   - SD Random Restart... " << std::flush;
            auto t0 = std::chrono::high_resolution_clock::now();
            auto sd_restart = opt.run_restart_trials(trap_state, args.trials, args.trials, false, 0, false);
            auto t1 = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(t1 - t0).count();

            TabuBenchmarkResult sd_rest_bench;
            sd_rest_bench.method_name = "SD Random Restart";
            sd_rest_bench.n = args.n;
            sd_rest_bench.d = -1;
            sd_rest_bench.og_state = trap_state;
            sd_rest_bench.og_energy = e_trap;
            sd_rest_bench.total_trials = sd_restart.total_starts;
            sd_rest_bench.improved_hits = sd_restart.improved_hits;
            sd_rest_bench.total_evals = sd_restart.total_evals;
            sd_rest_bench.success_rate = sd_restart.success_rate;
            sd_rest_bench.per_eval_rate = sd_restart.per_eval_rate;
            sd_rest_bench.mean_evals_per_trial = sd_restart.total_starts > 0 ? static_cast<double>(sd_restart.total_evals) / sd_restart.total_starts : 0.0;
            sd_rest_bench.elapsed_sec = sec;
            trap_results.push_back(sd_rest_bench);
            all_results.push_back(sd_rest_bench);
            std::cout << "Done (" << std::fixed << std::setprecision(2) << sec << " s, "
                      << "Rate r = " << std::scientific << std::setprecision(3) << sd_restart.per_eval_rate << ")\n";
        }

        // Print comparative table for this trap
        std::string table_title = "TRAP #" + std::to_string(trap.rank) + " (E0=" + std::to_string(e_trap) + ", d_min=" + std::to_string(d_min) + ") ESCAPE SUMMARY";
        TabuLogger::print_benchmark_table(trap_results, table_title);

        // Export to CSV if requested
        if (csv_out.is_open()) {
            for (const auto& r : trap_results) {
                csv_out << trap.rank << "," << r.n << "," << r.og_energy << "," << d_min << ","
                        << "\"" << r.method_name << "\"," << r.d << ","
                        << r.improved_hits << "," << r.total_trials << ","
                        << r.success_rate << "," << r.total_evals << ","
                        << r.per_eval_rate << "," << r.mean_evals_per_trial << ","
                        << r.elapsed_sec << "\n";
            }
        }
    }

    if (csv_out.is_open()) {
        csv_out.close();
        std::cout << "[Output] Exported comparative results to: " << args.output_csv << "\n";
    }

    std::cout << "========================================================================================================================\n";
    std::cout << " All 50,000-trial benchmarks completed successfully!\n";
    std::cout << "========================================================================================================================\n";
    return 0;
}
