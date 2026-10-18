#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <bitset>
#include <algorithm>
#include <unordered_set>
#include <fstream>
#include <random>
#include <string>
#include <limits>
#include <sstream>
#include <filesystem>

#include "benchmark_config.hpp"
#include "classical/sd_collector.cuh"
#include "classical/seed_manager.hpp"
#include "sampling/sampler.cuh"
#include "subspace/lanczos.cuh"
#include "exact/exact_diag.hpp"
#include "baselines/budgeted_baselines.hpp"

struct MethodStats
{
    long long min_energy = std::numeric_limits<long long>::max();
    size_t global_opt_hits = 0;
    size_t improved_hits = 0;
    double sum_energy = 0.0;
    double sum_evals = 0.0;

    void update(long long energy, size_t evals, long long seed_e, long long known_opt)
    {
        min_energy = std::min(min_energy, energy);
        if (energy <= known_opt) global_opt_hits++;
        if (energy < seed_e) improved_hits++;
        sum_energy += static_cast<double>(energy);
        sum_evals += static_cast<double>(evals);
    }
};

struct TrapSummaryRecord
{
    size_t n_spins;
    double g;
    int k_steps;
    int m_samples;
    double dt;
    size_t trap_rank;
    long long seed_energy;
    long long known_opt;
    double mean_flips;
    double mean_budget;
    MethodStats skqd;
    MethodStats quench;
    ClassicalRateResult kick;
    ClassicalRateResult restart;
    MethodStats blind;
    size_t num_seeds;
    double mean_total_time;
};

void print_trap_comparison_table(
    size_t rank, long long seed_energy, long long known_opt,
    size_t n, double g, int k_steps, int m_samples, double dt,
    double avg_flips, double avg_budget, size_t num_seeds,
    const MethodStats &skqd, const MethodStats &quench,
    const ClassicalRateResult &kick, const ClassicalRateResult &restart,
    const MethodStats &blind)
{
    std::cout << "\n========================================================================================================================================\n";
    std::cout << " [TRAP LEVEL #" << rank << "] Seed Energy E0 = " << seed_energy 
              << " | Known Opt E* = " << known_opt 
              << " | N = " << n << ", g = " << std::fixed << std::setprecision(2) << g 
              << " (K=" << k_steps << ", M=" << m_samples << ", dt=" << std::setprecision(4) << dt << ")\n";
    std::cout << " Quantum Diffusion: <dH> = " << std::fixed << std::setprecision(1) << avg_flips 
              << " flips | Budget B = " << std::setprecision(0) << avg_budget 
              << " evals | " << num_seeds << " Sampling Seeds | " << kick.total_trials << " Classical Trials/d\n";
    std::cout << "========================================================================================================================================\n";
    std::cout << " Algorithm                           | Min E | Global Opt % | Success % (E < " << seed_energy << ") | Per-Eval Rate r (ev^-1) | Mean Evals\n";
    std::cout << "-------------------------------------+-------+--------------+--------------------+-------------------------+------------\n";

    auto print_quantum_row = [&](const std::string &name, const MethodStats &st) {
        double opt_pct = num_seeds > 0 ? (100.0 * st.global_opt_hits / num_seeds) : 0.0;
        double succ_pct = num_seeds > 0 ? (100.0 * st.improved_hits / num_seeds) : 0.0;
        double mean_evals = num_seeds > 0 ? (st.sum_evals / num_seeds) : 0.0;
        double per_eval_r = (st.sum_evals > 0) ? (static_cast<double>(st.improved_hits) / st.sum_evals) : 0.0;
        std::ostringstream ss_r;
        ss_r << std::scientific << std::setprecision(2) << per_eval_r;

        std::cout << " " << std::left << std::setw(35) << name << " | "
                  << std::right << std::setw(5) << st.min_energy << " | "
                  << std::setw(11) << std::fixed << std::setprecision(1) << opt_pct << "% | "
                  << std::setw(17) << std::fixed << std::setprecision(1) << succ_pct << "% | "
                  << std::setw(23) << ss_r.str() << " | "
                  << std::setw(10) << std::fixed << std::setprecision(1) << mean_evals << "\n";
    };

    auto print_classical_row = [&](const std::string &name, const ClassicalRateResult &cr, size_t budget_B) {
        double succ_pct = cr.effective_budget_succ_rate(budget_B) * 100.0;
        std::ostringstream ss_r;
        ss_r << std::scientific << std::setprecision(2) << cr.per_eval_rate;

        std::cout << " " << std::left << std::setw(35) << name << " | "
                  << std::right << std::setw(5) << cr.min_energy << " | "
                  << std::setw(11) << "N/A" << " | "
                  << std::setw(17) << std::fixed << std::setprecision(1) << succ_pct << "% | "
                  << std::setw(23) << ss_r.str() << " | "
                  << std::setw(10) << std::fixed << std::setprecision(1) << cr.mean_evals_per_trial << "\n";
    };

    print_quantum_row("1. SKQD Dynamics (Raw Subspace)", skqd);
    print_quantum_row("2. Dynamics + SD Quench", quench);
    std::string kick_name = "3. Random Kick (" + std::to_string(std::max(1, (int)std::round(avg_flips))) + " flips) + SD";
    size_t b_evals = static_cast<size_t>(std::round(avg_budget));
    print_classical_row(kick_name, kick, b_evals);
    print_classical_row("4. Multi-Start SD (Random Restarts)", restart, b_evals);
    print_quantum_row("5. Blind Random Sampling", blind);
    std::cout << "========================================================================================================================================\n" << std::endl;
}

struct DelayedTrialRecord
{
    uint64_t sample_seed = 0;
    double mean_flips = 0.0;
    double window_pless = 0.0;

    // M1
    bool m1_success = false;
    long long m1_best_energy = 0;
    size_t m1_evals = 0;

    // Timings
    double time_m1 = 0.0;
};

struct StandardTrialRecord
{
    uint64_t sample_seed = 0;
    long long quantum_shots = 0;
    double mean_flips = 0.0;
    int kick_d = 0;
    size_t budget_B = 0;
    size_t subspace_dim = 0;
    long long best_classical_in_subspace = 0;
    long long quench_energy = 0;
    size_t quench_evals = 0;
    long long blind_energy = 0;
    size_t blind_evals = 0;
    double time_sample = 0.0;
    double time_sub = 0.0;
};

struct DelayedTrapSummaryRecord
{
    size_t n_spins;
    double g;
    size_t trap_rank;
    long long seed_energy;
    long long known_opt;
    double mean_flips;
    double window_mean_p_less;
    size_t num_seeds;
    MethodStats m1;
    int best_d;
    ClassicalRateResult m3_best;
    int matched_d;
    ClassicalRateResult m3_matched;
    std::map<int, ClassicalRateResult> m3_by_d;
    ClassicalRateResult restart;
    double mean_total_time;
};

void print_delayed_window_comparison_table(
    size_t rank, long long seed_energy, long long known_opt,
    size_t n, double g,
    double t_start, double dt_sample, double dt, size_t k_steps,
    double avg_flips, double avg_p_less, size_t num_seeds,
    const MethodStats &m1,
    int best_d, const ClassicalRateResult &m3_best,
    int matched_d, const ClassicalRateResult &m3_matched,
    const std::map<int, ClassicalRateResult> &m3_by_d,
    const ClassicalRateResult &restart_res)
{
    std::cout << "\n====================================================================================================================================================\n";
    std::cout << " [DELAYED WINDOW BENCHMARK] Trap Seed #" << rank 
              << " | Seed Energy E0 = " << seed_energy 
              << " | Known E* = " << known_opt 
              << " | N = " << n << ", g = " << std::fixed << std::setprecision(2) << g << "\n";
    double t_end = t_start + (k_steps > 0 ? (k_steps - 1) * dt_sample : 0.0);
    std::cout << " Window: t_k in [" << std::fixed << std::setprecision(1) << t_start 
              << ", " << t_end << "] (K=" << k_steps << ", dt_sample=" << dt_sample 
              << ", dt=" << std::setprecision(4) << dt << ") | Diffusion <dH> = " 
              << std::fixed << std::setprecision(1) << avg_flips << " flips | Budget B = 2000 | Classical Trials = "
              << m3_best.total_trials << "/d\n";
    std::cout << "====================================================================================================================================================\n";
    std::cout << " Method                                          | Success Rate (E < E0) | Per-Eval Rate r (ev^-1) | Min Energy | Statevec Window <P_<> | Mean Evals\n";
    std::cout << "-------------------------------------------------+-----------------------+-------------------------+------------+-----------------------+------------\n";

    auto print_row = [&](const std::string &name, double succ_pct, const std::string &rate_str, long long min_e, const std::string &pless_str, double mean_evals) {
        std::cout << " " << std::left << std::setw(47) << name << " | "
                  << std::right << std::setw(19) << std::fixed << std::setprecision(1) << succ_pct << " % | "
                  << std::setw(23) << rate_str << " | "
                  << std::setw(10) << min_e << " | "
                  << std::setw(21) << pless_str << " | "
                  << std::setw(10) << std::fixed << std::setprecision(1) << mean_evals << "\n";
    };

    auto format_rate = [](double r) {
        std::ostringstream ss;
        ss << std::scientific << std::setprecision(2) << r;
        return ss.str();
    };

    double m1_succ = num_seeds > 0 ? (100.0 * m1.improved_hits / num_seeds) : 0.0;
    double m1_evals = num_seeds > 0 ? (m1.sum_evals / num_seeds) : 0.0;
    std::ostringstream ss_pless;
    ss_pless << std::fixed << std::setprecision(4) << avg_p_less;
    print_row("1. Dynamical Sampling (K=" + std::to_string(k_steps) + ", M=100)", m1_succ, "N/A", m1.min_energy, ss_pless.str(), m1_evals);

    std::string best_d_str = "2a. Random Kick + SD (Best d* = " + std::to_string(best_d) + ")";
    print_row(best_d_str, m3_best.effective_budget_succ_rate(2000) * 100.0, format_rate(m3_best.per_eval_rate), m3_best.min_energy, "N/A", m3_best.mean_evals_per_trial);

    std::string match_d_str = "2b. Random Kick + SD (Matched d = " + std::to_string(matched_d) + " ~ <dH>)";
    print_row(match_d_str, m3_matched.effective_budget_succ_rate(2000) * 100.0, format_rate(m3_matched.per_eval_rate), m3_matched.min_energy, "N/A", m3_matched.mean_evals_per_trial);

    for (const auto &kv : m3_by_d) {
        std::string scan_str = "   -- Scan d = " + std::to_string(kv.first) + " + SD";
        print_row(scan_str, kv.second.effective_budget_succ_rate(2000) * 100.0, format_rate(kv.second.per_eval_rate), kv.second.min_energy, "N/A", kv.second.mean_evals_per_trial);
    }

    print_row("3. Multi-Start SD (Random Restarts)", restart_res.effective_budget_succ_rate(2000) * 100.0, format_rate(restart_res.per_eval_rate), restart_res.min_energy, "N/A", restart_res.mean_evals_per_trial);

    std::cout << "====================================================================================================================================================\n" << std::endl;
}

void print_delayed_window_summary_report(
    const std::vector<DelayedTrapSummaryRecord> &records,
    size_t n_spins,
    double current_g)
{
    std::cout << "\n========================================================================================================================================\n";
    std::cout << "                                  DELAYED WINDOW (g=" << std::fixed << std::setprecision(2) << current_g << ") BENCHMARK SUMMARY (N = " << n_spins << ")\n";
    std::cout << "========================================================================================================================================\n";
    std::cout << " Trap Seed (E0)  | M1 Succ % | Window <P_<> | Best d* | M3(d*) Succ % | M3(d*) Rate | Matched d | M3(match) Succ % | Restart Rate | Known E*\n";
    std::cout << "-----------------+-----------+--------------+---------+---------------+-------------+-----------+------------------+--------------+----------\n";

    for (const auto &rec : records) {
        double m1_succ = rec.num_seeds > 0 ? (100.0 * rec.m1.improved_hits / rec.num_seeds) : 0.0;
        double m3_best_succ = rec.m3_best.effective_budget_succ_rate(2000) * 100.0;
        double m3_match_succ = rec.m3_matched.effective_budget_succ_rate(2000) * 100.0;

        std::ostringstream ss_pless, ss_best_r, ss_rst_r;
        ss_pless << std::fixed << std::setprecision(4) << rec.window_mean_p_less;
        ss_best_r << std::scientific << std::setprecision(2) << rec.m3_best.per_eval_rate;
        ss_rst_r << std::scientific << std::setprecision(2) << rec.restart.per_eval_rate;

        std::string trap_str = "Rank #" + std::to_string(rec.trap_rank) + " (E=" + std::to_string(rec.seed_energy) + ")";
        std::cout << " " << std::left << std::setw(15) << trap_str << " | "
                  << std::right << std::setw(7) << std::fixed << std::setprecision(1) << m1_succ << " % | "
                  << std::setw(12) << ss_pless.str() << " | "
                  << std::setw(7) << rec.best_d << " | "
                  << std::setw(11) << std::fixed << std::setprecision(1) << m3_best_succ << " % | "
                  << std::setw(11) << ss_best_r.str() << " | "
                  << std::setw(9) << rec.matched_d << " | "
                  << std::setw(14) << std::fixed << std::setprecision(1) << m3_match_succ << " % | "
                  << std::setw(12) << ss_rst_r.str() << " | "
                  << std::setw(8) << rec.known_opt << "\n";
    }
    std::cout << "========================================================================================================================================\n" << std::endl;
}

void warmup_gpu(double g = 0.5)
{
    std::cout << "[WARMUP] Initializing GPU contexts, cuStateVec, cuBLAS, cuSOLVER, and device kernels..." << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();

    SpinCache dummy_cache;
    CuStateVecSampler dummy_sampler;

    // 1. Prime cuStateVec and sampling kernels
    SamplerConfig dummy_cfg;
    dummy_cfg.n = 4;
    dummy_cfg.g = g;
    dummy_cfg.dt = 0.1;
    dummy_cfg.k_steps = 1;
    dummy_cfg.m_samples = 2;
    dummy_cfg.top_k_augmentation = 0;

    SpinState dummy_seed{4, {0ULL}};
    std::vector<SpinState> dummy_seeds = {dummy_seed};
    auto dummy_basis = dummy_sampler.generate_subspace(dummy_seeds, dummy_cfg, dummy_cache);

    // 2. Prime cuBLAS, cuSOLVER, and Lanczos kernels
    std::vector<SpinState> tiny_basis = {
        {4, {0ULL}},
        {4, {1ULL}}
    };
    auto dummy_sol = SpinEigensolver<SpinState, double>::compute_ground_state(tiny_basis, g);

    // 3. Ensure full device synchronization so all async initialization finishes
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    double warmup_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "[WARMUP] GPU context primed and synchronized in " 
              << std::fixed << std::setprecision(3) << warmup_s 
              << "s. Warm-up overhead successfully excluded from benchmark metrics.\n" << std::endl;
}

inline long long get_known_optimum(size_t n)
{
    if (n == 16) return 24;
    if (n == 20) return 26;
    if (n == 22) return 39;
    if (n == 24) return 36;
    if (n == 26) return 45;
    if (n == 28) return 50;
    if (n == 30) return 59;
    return -1;
}

int main(int argc, char **argv)
{
    std::cout << "[TRACE] Program started. Initializing CUDA..." << std::endl;
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices == 0)
    {
        std::cerr << "Error: No CUDA-capable devices found.\n";
        return 1;
    }
    cudaSetDevice(0);
    std::cout << "[TRACE] CUDA Device 0 set." << std::endl;

    // Load benchmark configuration from file or defaults
    BenchmarkConfig bench_cfg;
    if (argc > 1) {
        std::string arg1 = argv[1];
        if (std::filesystem::exists(arg1)) {
            bench_cfg = BenchmarkConfig::load_from_file(arg1);
        } else if (std::filesystem::exists("configs/" + arg1)) {
            bench_cfg = BenchmarkConfig::load_from_file("configs/" + arg1);
        } else if (std::filesystem::exists("../configs/" + arg1)) {
            bench_cfg = BenchmarkConfig::load_from_file("../configs/" + arg1);
        } else if (std::filesystem::exists("../" + arg1)) {
            bench_cfg = BenchmarkConfig::load_from_file("../" + arg1);
        } else {
            // Backward compatibility: If arg1 is numeric, treat as target_minima_count
            try {
                bench_cfg.target_minima_count = std::stoul(arg1);
                if (argc > 2) {
                    bench_cfg.num_sampling_seeds = std::stoul(argv[2]);
                }
            } catch (...) {
                throw std::runtime_error("Configuration file not found: " + arg1);
            }
        }
    } else if (std::filesystem::exists("configs/default_benchmark.cfg")) {
        bench_cfg = BenchmarkConfig::load_from_file("configs/default_benchmark.cfg");
    } else if (std::filesystem::exists("../configs/default_benchmark.cfg")) {
        bench_cfg = BenchmarkConfig::load_from_file("../configs/default_benchmark.cfg");
    }

    // Run one-time warm-up to prime libraries and driver kernels using configured g
    warmup_gpu(bench_cfg.g);

    auto sampling_seeds = bench_cfg.generate_sampling_seeds();
    bench_cfg.print_summary();

    bool is_delayed_window = (bench_cfg.t_start > 0.0 || bench_cfg.dt_sample > 0.0);
    std::string csv_filename = bench_cfg.csv_results_filename;
    std::ofstream csv(csv_filename);
    std::string summary_csv_filename = bench_cfg.csv_summary_filename;
    std::ofstream summary_csv(summary_csv_filename);

    if (is_delayed_window)
    {
        csv << "N,g,TrapRank,MinID,SeedEnergy,SamplingSeed,K,M,dt,t_start,dt_sample,MeanFlips,StatevecWindowPless,Budget_B,"
            << "M1_Success,M1_BestEnergy,M1_Evals,"
            << "M3_Best_d,M3_Best_PerEvalRate,M3_Best_BudgetSuccRate,M3_Best_Energy,"
            << "M3_Matched_d,M3_Matched_PerEvalRate,M3_Matched_BudgetSuccRate,M3_Matched_Energy,"
            << "M3_d2_PerEvalRate,M3_d2_BudgetSuccRate,M3_d2_Energy,"
            << "M3_d4_PerEvalRate,M3_d4_BudgetSuccRate,M3_d4_Energy,"
            << "M3_d6_PerEvalRate,M3_d6_BudgetSuccRate,M3_d6_Energy,"
            << "M3_d8_PerEvalRate,M3_d8_BudgetSuccRate,M3_d8_Energy,"
            << "M3_d10_PerEvalRate,M3_d10_BudgetSuccRate,M3_d10_Energy,"
            << "M3_d12_PerEvalRate,M3_d12_BudgetSuccRate,M3_d12_Energy,"
            << "Restart_PerEvalRate,Restart_BudgetSuccRate,Restart_Energy,"
            << "M1_Time_s,ClassicalTime_s,TotalTime_s\n";

        summary_csv << "N,g,TrapRank,SeedEnergy,KnownGlobalOpt,NumSeeds,MeanFlips,StatevecWindowPless,Budget_B,"
                    << "M1_SuccessRate,M1_MinEnergy,M1_OptPct,M1_MeanDrop,M1_MeanEvals,"
                    << "M3_Best_d,M3_Best_SuccessRate,M3_Best_PerEvalRate,M3_Best_MinEnergy,M3_Best_MeanEvals,"
                    << "M3_Matched_d,M3_Matched_SuccessRate,M3_Matched_PerEvalRate,M3_Matched_MinEnergy,M3_Matched_MeanEvals,"
                    << "M3_d2_SuccessRate,M3_d2_PerEvalRate,M3_d2_MinEnergy,"
                    << "M3_d4_SuccessRate,M3_d4_PerEvalRate,M3_d4_MinEnergy,"
                    << "M3_d6_SuccessRate,M3_d6_PerEvalRate,M3_d6_MinEnergy,"
                    << "M3_d8_SuccessRate,M3_d8_PerEvalRate,M3_d8_MinEnergy,"
                    << "M3_d10_SuccessRate,M3_d10_PerEvalRate,M3_d10_MinEnergy,"
                    << "M3_d12_SuccessRate,M3_d12_PerEvalRate,M3_d12_MinEnergy,"
                    << "Restart_SuccessRate,Restart_PerEvalRate,Restart_MinEnergy,Restart_MeanEvals,"
                    << "MeanM1Time_s,ClassicalTime_s,MeanTotalTime_s\n";
    }
    else
    {
        csv << "N,g,TrapRank,MinID,SeedEnergy,SamplingSeed,K,M,dt,QuantumShots,MeanFlips,KickDistance,Budget_B,SubspaceDim,"
            << "SKQD_BestEnergy,SKQD_Evals,SKQD_Improved,"
            << "Quench_BestEnergy,Quench_Evals,Quench_Improved,"
            << "Kick_PerEvalRate,Kick_BudgetSuccRate,Kick_BestEnergy,"
            << "Restart_PerEvalRate,Restart_BudgetSuccRate,Restart_BestEnergy,"
            << "Blind_BestEnergy,Blind_Evals,Blind_Improved,"
            << "SamplingTime_s,LanczosTime_s,ClassicalTime_s,TotalTime_s\n";

        summary_csv << "N,g,K,M,dt,TrapRank,SeedEnergy,KnownGlobalOpt,QuantumShots,MeanFlips,Budget_B,"
                    << "SKQD_MinE,SKQD_OptPct,SKQD_SuccPct,SKQD_MeanDrop,SKQD_MeanEvals,SKQD_PerEvalRate,"
                    << "Quench_MinE,Quench_OptPct,Quench_SuccPct,Quench_MeanDrop,Quench_MeanEvals,Quench_PerEvalRate,"
                    << "Kick_MinE,Kick_OptPct,Kick_SuccPct,Kick_MeanDrop,Kick_MeanEvals,Kick_PerEvalRate,"
                    << "Restart_MinE,Restart_OptPct,Restart_SuccPct,Restart_MeanDrop,Restart_MeanEvals,Restart_PerEvalRate,"
                    << "Blind_MinE,Blind_OptPct,Blind_SuccPct,Blind_MeanDrop,Blind_MeanEvals,Blind_PerEvalRate,"
                    << "MeanSamplingTime_s,MeanLanczosTime_s,ClassicalTime_s,MeanTotalTime_s\n";
    }

    std::vector<TrapSummaryRecord> overall_trap_records;
    std::vector<DelayedTrapSummaryRecord> overall_delayed_records;

    for (size_t n_idx = 0; n_idx < bench_cfg.n_values.size(); ++n_idx)
    {
        size_t n_spins = bench_cfg.n_values[n_idx];
        auto start_n_time = std::chrono::high_resolution_clock::now();

        std::cout << "\n======================================================================" << std::endl;
        std::cout << "               Evaluating System Size N = " << n_spins << std::endl;
        std::cout << "======================================================================" << std::endl;

        SpinCache cache;
        CuStateVecSampler sampler;

        bool seeds_were_cached = false;
        auto all_minima = SeedManager::get_or_collect_minima(
            n_spins, bench_cfg.pool_minima_count, bench_cfg.seeds_dir,
            bench_cfg.force_regenerate_seeds, cache, seeds_were_cached);
        long long known_opt = get_known_optimum(n_spins);

        // Sort ascending by energy (deepest traps first)
        std::sort(all_minima.begin(), all_minima.end(), [](const auto &a, const auto &b) {
            return a.energy < b.energy;
        });

        // Select representative minima
        std::vector<SDMinimumRecord> selected_minima;
        std::unordered_set<long long> seen_energies;

        if (seeds_were_cached && all_minima.size() <= bench_cfg.top_minima_count) {
            // Direct seed mode: cached file provided a tailored list of target seeds
            selected_minima = all_minima;
            std::cout << "[INFO] Using all " << selected_minima.size() 
                      << " seeds directly from cached seed file.\n";
        } else if (bench_cfg.minima_selection_mode == "top_distinct_minima" ||
            bench_cfg.minima_selection_mode == "lowest_energy_orbits") {
            std::unordered_set<SpinState, SpinStateHash> seen_canonical_seeds;
            for (const auto &m : all_minima) {
                if (m.energy > known_opt) {
                    SpinState canon = SpinSymmetry::get_canonical_state(m.state);
                    if (seen_canonical_seeds.find(canon) == seen_canonical_seeds.end()) {
                        seen_canonical_seeds.insert(canon);
                        selected_minima.push_back(m);
                        if (selected_minima.size() >= bench_cfg.top_minima_count) break;
                    }
                }
            }
        } else {
            // Default "distinct_energies": one representative per distinct energy level
            for (const auto &m : all_minima) {
                if (m.energy > known_opt && seen_energies.find(m.energy) == seen_energies.end()) {
                    selected_minima.push_back(m);
                    seen_energies.insert(m.energy);
                    if (selected_minima.size() >= bench_cfg.top_minima_count) break;
                }
            }
        }

        // Fallback if none found strictly above known_opt
        if (selected_minima.empty() && !all_minima.empty()) {
            for (const auto &m : all_minima) {
                selected_minima.push_back(m);
                if (selected_minima.size() >= bench_cfg.top_minima_count) break;
            }
        }

        // Optional mandatory energy target if specified in config (generic search in pool)
        if (bench_cfg.mandatory_energy >= 0) {
            long long target_e = bench_cfg.mandatory_energy;
            bool found_target = false;
            for (const auto &m : selected_minima) {
                if (m.energy == target_e) {
                    found_target = true;
                    break;
                }
            }
            if (!found_target) {
                // Search the broader pool for a state with the target energy
                auto it = std::find_if(all_minima.begin(), all_minima.end(), 
                    [target_e](const auto &m) { return m.energy == target_e; });
                if (it != all_minima.end()) {
                    if (selected_minima.size() >= bench_cfg.top_minima_count) {
                        selected_minima.back() = *it;
                    } else {
                        selected_minima.push_back(*it);
                    }
                    std::cout << "[INFO] Mandatory energy E=" << target_e 
                              << " included from pool into selected traps.\n";
                } else {
                    std::cout << "[WARNING] Mandatory energy E=" << target_e 
                              << " was not found in pool of " << all_minima.size() << " minima.\n";
                }
            }
        }

        // Re-sort selected traps ascending by energy
        std::sort(selected_minima.begin(), selected_minima.end(), [](const auto &a, const auto &b) {
            return a.energy < b.energy;
        });

        std::cout << "[INFO] Selected " << selected_minima.size() 
                  << " lowest-energy symmetry-distinct seeds (strictly above E* = " << known_opt << "):\n";
        std::map<long long, size_t> level_orbit_counts;
        for (const auto &m : selected_minima) level_orbit_counts[m.energy]++;
        std::cout << "  Breakdown across energy levels:\n";
        for (const auto &[e_val, count] : level_orbit_counts) {
            std::cout << "    - Energy E0 = " << std::setw(3) << e_val << " : " 
                      << count << " distinct symmetry orbit" << (count > 1 ? "s" : "") << "\n";
        }
        std::cout << "  Individual Seed Listing:\n";
        for (size_t i = 0; i < selected_minima.size(); ++i) {
            std::cout << "    Seed #" << std::setw(2) << (i + 1) << ": Energy E0 = " << selected_minima[i].energy 
                      << " | Orbit MinID = " << selected_minima[i].id << "\n";
        }

        if (is_delayed_window)
        {
            // Precompute classical ensemble baselines (M3 kick scan & Restarts) once per selected trap minimum (independent of g)
            struct CachedClassicalBaselines {
                std::map<int, ClassicalRateResult> results_by_d;
                int best_d = 2;
                ClassicalRateResult best_rate;
                ClassicalRateResult restart_res;
                double time_classical = 0.0;
            };
            std::vector<CachedClassicalBaselines> classical_cache(selected_minima.size());
            std::cout << "\n[INFO] Precomputing classical ensemble baselines (M3 kick scan & Restarts: " 
                      << bench_cfg.classical_trials_per_d << " trials/d, " 
                      << bench_cfg.restart_trials << " restarts) across " 
                      << selected_minima.size() << " traps..." << std::endl;

            for (size_t trap_idx = 0; trap_idx < selected_minima.size(); ++trap_idx) {
                const auto &min_rec = selected_minima[trap_idx];
                auto start_classical = std::chrono::high_resolution_clock::now();
                std::mt19937_64 classical_rng(bench_cfg.seed_base ^ (trap_idx * 10007ULL));

                // Scan all d in kick_scan_d with dummy flips 0.0; matched_d will be dynamically resolved per g
                LargeScaleKickScanResult raw_scan = run_large_scale_kick_scan(
                    min_rec.state, min_rec.energy, bench_cfg.kick_scan_d, 0.0,
                    bench_cfg.classical_trials_per_d, classical_rng);
                ClassicalRateResult restart_res = run_large_scale_restart_sd_rate(
                    n_spins, min_rec.energy, bench_cfg.restart_trials, classical_rng);
                auto end_classical = std::chrono::high_resolution_clock::now();

                auto &cc = classical_cache[trap_idx];
                cc.results_by_d = std::move(raw_scan.results_by_d);
                cc.best_d = raw_scan.best_d;
                cc.best_rate = raw_scan.best_rate;
                cc.restart_res = std::move(restart_res);
                cc.time_classical = std::chrono::duration<double>(end_classical - start_classical).count();
            }

            for (double current_g : bench_cfg.g_values)
            {
                double t_end = bench_cfg.t_start + (bench_cfg.k_steps > 0 ? (bench_cfg.k_steps - 1) * bench_cfg.dt_sample : 0.0);
                std::cout << "\n>>> DELAYED WINDOW (g=" << std::fixed << std::setprecision(2) << current_g 
                          << ") BENCHMARK: Window t_k in [" << std::setprecision(1) << bench_cfg.t_start 
                          << ", " << t_end << "] (K=" << bench_cfg.k_steps 
                          << ", dt_sample=" << bench_cfg.dt_sample << ", dt=" << std::setprecision(4) << bench_cfg.dt << ")\n"
                          << "    Method 1: K=" << bench_cfg.k_steps << ", M=" << bench_cfg.m_samples << " (2000 evals)\n"
                          << "    Method 3: Kick scan d in {";
                for (size_t di = 0; di < bench_cfg.kick_scan_d.size(); ++di) {
                    std::cout << bench_cfg.kick_scan_d[di] << (di + 1 < bench_cfg.kick_scan_d.size() ? ", " : "");
                }
                std::cout << "} with 2000 evals per d\n"
                          << "    (Evaluating " << selected_minima.size() << " trap depth levels x " 
                          << bench_cfg.num_sampling_seeds << " seeds) <<<\n" << std::endl;

                for (size_t trap_idx = 0; trap_idx < selected_minima.size(); ++trap_idx)
                {
                size_t rank = trap_idx + 1;
                const auto &min_rec = selected_minima[trap_idx];
                SpinState current_seed = min_rec.state;
                long long seed_energy = min_rec.energy;

                MethodStats m1_stats;

                double sum_mean_flips = 0.0;
                double sum_pless = 0.0;
                double sum_m1_time = 0.0;
                size_t valid_trials = 0;
                std::vector<DelayedTrialRecord> trial_records;

                for (uint64_t sample_seed : sampling_seeds)
                {
                    std::vector<SpinState> seeds = {current_seed};

                    // 1. Method 1: Dynamical Sampling (K=20, M=100)
                    SamplerConfig config1;
                    config1.n = n_spins;
                    config1.g = current_g;
                    config1.dt = bench_cfg.dt;
                    config1.k_steps = bench_cfg.k_steps;
                    config1.m_samples = bench_cfg.m_samples; // 100 -> 2000 shots
                    config1.t_start = bench_cfg.t_start;
                    config1.dt_sample = bench_cfg.dt_sample;
                    config1.top_k_augmentation = 0;
                    config1.seed = sample_seed;
                    config1.compute_p_less = true;
                    config1.seed_energy = seed_energy;

                    cudaDeviceSynchronize();
                    auto start_m1 = std::chrono::high_resolution_clock::now();
                    std::vector<SpinState> basis1 = sampler.generate_subspace(seeds, config1, cache);
                    cudaDeviceSynchronize();
                    auto end_m1 = std::chrono::high_resolution_clock::now();
                    double time_m1 = std::chrono::duration<double>(end_m1 - start_m1).count();

                    if (basis1.empty()) {
                        std::cerr << "Warning: Subspace generation failed for M1. Skipping.\n";
                        continue;
                    }

                    double mean_flips = sampler.get_last_mean_flips();
                    double window_pless = sampler.get_last_window_mean_p_less();

                    long long m1_best_energy = seed_energy;
                    for (const auto &s : basis1) {
                        long long e = SpinLib::get_energy(cache, s);
                        if (e < m1_best_energy) m1_best_energy = e;
                    }
                    size_t m1_evals = bench_cfg.k_steps * bench_cfg.m_samples; // 2000
                    bool m1_success = (m1_best_energy < seed_energy);

                    // Update statistics
                    m1_stats.update(m1_best_energy, m1_evals, seed_energy, known_opt);

                    sum_mean_flips += mean_flips;
                    sum_pless += window_pless;
                    sum_m1_time += time_m1;
                    valid_trials++;

                    DelayedTrialRecord rec;
                    rec.sample_seed = sample_seed;
                    rec.mean_flips = mean_flips;
                    rec.window_pless = window_pless;
                    rec.m1_success = m1_success;
                    rec.m1_best_energy = m1_best_energy;
                    rec.m1_evals = m1_evals;
                    rec.time_m1 = time_m1;
                    trial_records.push_back(std::move(rec));
                }

                if (valid_trials == 0) continue;

                double avg_flips = sum_mean_flips / valid_trials;
                double avg_pless = sum_pless / valid_trials;
                double avg_m1_time = sum_m1_time / valid_trials;

                // 2. Classical Baselines: Retrieve from precomputed cache and match d to this g's avg_flips
                const auto &cc = classical_cache[trap_idx];
                LargeScaleKickScanResult kick_scan;
                kick_scan.results_by_d = cc.results_by_d;
                kick_scan.best_d = cc.best_d;
                kick_scan.best_rate = cc.best_rate;

                // Resolve matched_d for this g's average bit flips
                int target_match = std::max(1, static_cast<int>(std::round(avg_flips)));
                int min_diff = std::numeric_limits<int>::max();
                int resolved_matched_d = bench_cfg.kick_scan_d.empty() ? 2 : bench_cfg.kick_scan_d[0];
                for (int d_val : bench_cfg.kick_scan_d) {
                    int diff = std::abs(d_val - target_match);
                    if (diff < min_diff) {
                        min_diff = diff;
                        resolved_matched_d = d_val;
                    }
                }
                kick_scan.matched_d = resolved_matched_d;
                kick_scan.matched_rate = kick_scan.results_by_d.at(resolved_matched_d);

                const ClassicalRateResult &restart_res = cc.restart_res;
                double time_classical = cc.time_classical;
                double avg_total_time = avg_m1_time + (time_classical / valid_trials);

                int consensus_best_d = kick_scan.best_d;
                const ClassicalRateResult &m3_best = kick_scan.best_rate;
                int common_matched_d = kick_scan.matched_d;
                const ClassicalRateResult &m3_matched = kick_scan.matched_rate;

                // Write detailed trial results to CSV
                for (const auto &rec : trial_records) {
                    csv << n_spins << ","
                        << current_g << ","
                        << rank << ","
                        << min_rec.id << ","
                        << seed_energy << ","
                        << rec.sample_seed << ","
                        << bench_cfg.k_steps << ","
                        << bench_cfg.m_samples << ","
                        << bench_cfg.dt << ","
                        << bench_cfg.t_start << ","
                        << bench_cfg.dt_sample << ","
                        << rec.mean_flips << ","
                        << rec.window_pless << ","
                        << 2000 << ","
                        // M1
                        << (rec.m1_success ? 1 : 0) << ","
                        << rec.m1_best_energy << ","
                        << rec.m1_evals << ","
                        // M3 Best
                        << consensus_best_d << ","
                        << m3_best.per_eval_rate << ","
                        << m3_best.effective_budget_succ_rate(2000) << ","
                        << m3_best.min_energy << ","
                        // M3 Matched
                        << common_matched_d << ","
                        << m3_matched.per_eval_rate << ","
                        << m3_matched.effective_budget_succ_rate(2000) << ","
                        << m3_matched.min_energy << ",";

                    // M3 individual d {2, 4, 6, 8, 10, 12}
                    for (int d_val : {2, 4, 6, 8, 10, 12}) {
                        auto it_d = kick_scan.results_by_d.find(d_val);
                        if (it_d != kick_scan.results_by_d.end()) {
                            csv << it_d->second.per_eval_rate << ","
                                << it_d->second.effective_budget_succ_rate(2000) << ","
                                << it_d->second.min_energy << ",";
                        } else {
                            csv << 0.0 << "," << 0.0 << "," << seed_energy << ",";
                        }
                    }

                    // Restart
                    csv << restart_res.per_eval_rate << ","
                        << restart_res.effective_budget_succ_rate(2000) << ","
                        << restart_res.min_energy << ","
                        << rec.time_m1 << ","
                        << (time_classical / valid_trials) << ","
                        << (rec.time_m1 + (time_classical / valid_trials)) << "\n";
                }
                csv.flush();

                // Console table
                print_delayed_window_comparison_table(
                    rank, seed_energy, known_opt, n_spins, current_g,
                    bench_cfg.t_start, bench_cfg.dt_sample, bench_cfg.dt, bench_cfg.k_steps,
                    avg_flips, avg_pless, valid_trials,
                    m1_stats,
                    consensus_best_d, m3_best,
                    common_matched_d, m3_matched,
                    kick_scan.results_by_d,
                    restart_res);

                // Summary CSV logging
                summary_csv << n_spins << ","
                            << current_g << ","
                            << rank << ","
                            << seed_energy << ","
                            << known_opt << ","
                            << valid_trials << ","
                            << avg_flips << ","
                            << avg_pless << ","
                            << 2000 << ","
                            // M1
                            << (100.0 * m1_stats.improved_hits / valid_trials) << ","
                            << m1_stats.min_energy << ","
                            << (100.0 * m1_stats.global_opt_hits / valid_trials) << ","
                            << (seed_energy - (m1_stats.sum_energy / valid_trials)) << ","
                            << (m1_stats.sum_evals / valid_trials) << ","
                            // M3 Best
                            << consensus_best_d << ","
                            << (m3_best.effective_budget_succ_rate(2000) * 100.0) << ","
                            << m3_best.per_eval_rate << ","
                            << m3_best.min_energy << ","
                            << m3_best.mean_evals_per_trial << ","
                            // M3 Matched
                            << common_matched_d << ","
                            << (m3_matched.effective_budget_succ_rate(2000) * 100.0) << ","
                            << m3_matched.per_eval_rate << ","
                            << m3_matched.min_energy << ","
                            << m3_matched.mean_evals_per_trial << ",";

                // M3 individual d
                for (int d_val : {2, 4, 6, 8, 10, 12}) {
                    auto it_d = kick_scan.results_by_d.find(d_val);
                    if (it_d != kick_scan.results_by_d.end()) {
                        double succ = it_d->second.effective_budget_succ_rate(2000) * 100.0;
                        summary_csv << succ << "," << it_d->second.per_eval_rate << "," << it_d->second.min_energy << ",";
                    } else {
                        summary_csv << 0.0 << "," << 0.0 << "," << seed_energy << ",";
                    }
                }

                summary_csv << (restart_res.effective_budget_succ_rate(2000) * 100.0) << ","
                            << restart_res.per_eval_rate << ","
                            << restart_res.min_energy << ","
                            << restart_res.mean_evals_per_trial << ","
                            << avg_m1_time << ","
                            << time_classical << ","
                            << avg_total_time << "\n";
                summary_csv.flush();

                overall_delayed_records.push_back({
                    n_spins, current_g, rank, seed_energy, known_opt,
                    avg_flips, avg_pless, valid_trials,
                    m1_stats,
                    consensus_best_d, m3_best,
                    common_matched_d, m3_matched,
                    kick_scan.results_by_d,
                    restart_res,
                    avg_total_time
                });
            }
            }
        }
        else
        {
            for (double g_val : bench_cfg.g_values)
            {
                for (const auto &scan : bench_cfg.scan_params)
                {
                    double current_g = (bench_cfg.g_values.size() == 1) ? scan.g : g_val;

                std::cout << "\n>>> Scan Parameter: K=" << scan.k_steps 
                          << ", M=" << scan.m_samples << ", dt=" << scan.dt 
                          << ", g=" << current_g
                          << " (Evaluating " << selected_minima.size() << " trap depth levels x " 
                          << bench_cfg.num_sampling_seeds << " seeds) <<<\n" << std::endl;

                for (size_t trap_idx = 0; trap_idx < selected_minima.size(); ++trap_idx)
                {
                    size_t rank = trap_idx + 1;
                    const auto &min_rec = selected_minima[trap_idx];
                    SpinState current_seed = min_rec.state;
                    long long seed_energy = min_rec.energy;

                    MethodStats stats_skqd;
                    MethodStats stats_quench;
                    MethodStats blind_stats;

                    double sum_mean_flips = 0.0;
                    double sum_budget_b = 0.0;
                    double sum_time_samp = 0.0;
                    double sum_time_lanc = 0.0;
                    size_t valid_trials = 0;
                    std::vector<StandardTrialRecord> trial_records;

                    for (uint64_t sample_seed : sampling_seeds)
                    {
                        SamplerConfig config;
                        config.n = n_spins;
                        config.g = current_g;
                        config.dt = scan.dt;
                        config.k_steps = scan.k_steps;
                        config.m_samples = scan.m_samples;
                        config.top_k_augmentation = bench_cfg.top_k_augmentation;
                        config.seed = sample_seed;

                        std::vector<SpinState> seeds = {current_seed};

                        // 1. Quantum Dynamics Sampling
                        cudaDeviceSynchronize();
                        auto start_sample = std::chrono::high_resolution_clock::now();
                        std::vector<SpinState> subspace_basis = sampler.generate_subspace(seeds, config, cache);
                        cudaDeviceSynchronize();
                        auto end_sample = std::chrono::high_resolution_clock::now();
                        double time_sample = std::chrono::duration<double>(end_sample - start_sample).count();

                        size_t subspace_dim = subspace_basis.size();
                        if (subspace_dim == 0)
                        {
                            std::cerr << "Warning: Subspace generation failed. Skipping.\n";
                            continue;
                        }

                        // Exact Lanczos timing in Krylov subspace
                        cudaDeviceSynchronize();
                        auto start_sub = std::chrono::high_resolution_clock::now();
                        auto result_subspace = SpinEigensolver<SpinState, double>::compute_ground_state(subspace_basis, config.g);
                        cudaDeviceSynchronize();
                        auto end_sub = std::chrono::high_resolution_clock::now();
                        double time_sub = std::chrono::duration<double>(end_sub - start_sub).count();

                        long long best_classical_in_subspace = std::numeric_limits<long long>::max();
                        SpinState best_subspace_state = current_seed;
                        for (size_t i = 0; i < subspace_dim; ++i)
                        {
                            long long state_energy = SpinLib::get_energy(cache, subspace_basis[i]);
                            if (state_energy < best_classical_in_subspace)
                            {
                                best_classical_in_subspace = state_energy;
                                best_subspace_state = subspace_basis[i];
                            }
                        }

                        double mean_flips = sampler.get_last_mean_flips();
                        int kick_d = std::max(1, static_cast<int>(std::round(mean_flips)));
                        size_t budget_B = (bench_cfg.evaluation_budget > 0) 
                                          ? bench_cfg.evaluation_budget 
                                          : subspace_dim;

                        // 2. Dynamics + SD Quench
                        SerialDescentResult quench_res = run_serial_sd_quench(best_subspace_state);
                        long long quench_energy = quench_res.final_energy;
                        size_t quench_evals = budget_B + quench_res.evals_used;

                        // 3. Blind Random Sampling
                        std::mt19937_64 classical_rng(sample_seed ^ 0x9e3779b97f4a7c15ULL);
                        BudgetedComparisonResult blind_res = run_serial_blind_sampling(
                            n_spins, seed_energy, budget_B, classical_rng);

                        // Accumulate
                        stats_skqd.update(best_classical_in_subspace, budget_B, seed_energy, known_opt);
                        stats_quench.update(quench_energy, quench_evals, seed_energy, known_opt);
                        blind_stats.update(blind_res.best_energy, blind_res.evals_used, seed_energy, known_opt);

                        sum_mean_flips += mean_flips;
                        sum_budget_b += static_cast<double>(budget_B);
                        sum_time_samp += time_sample;
                        sum_time_lanc += time_sub;
                        valid_trials++;

                        long long quantum_shots = scan.k_steps * scan.m_samples;

                        StandardTrialRecord tr;
                        tr.sample_seed = sample_seed;
                        tr.quantum_shots = quantum_shots;
                        tr.mean_flips = mean_flips;
                        tr.kick_d = kick_d;
                        tr.budget_B = budget_B;
                        tr.subspace_dim = subspace_dim;
                        tr.best_classical_in_subspace = best_classical_in_subspace;
                        tr.quench_energy = quench_energy;
                        tr.quench_evals = quench_evals;
                        tr.blind_energy = blind_res.best_energy;
                        tr.blind_evals = blind_res.evals_used;
                        tr.time_sample = time_sample;
                        tr.time_sub = time_sub;
                        trial_records.push_back(tr);
                    }

                    if (valid_trials == 0) continue;

                    double avg_flips = sum_mean_flips / valid_trials;
                    double avg_budget = sum_budget_b / valid_trials;
                    size_t mean_b = static_cast<size_t>(std::round(avg_budget));
                    int kick_d = std::max(1, static_cast<int>(std::round(avg_flips)));
                    double avg_samp_time = sum_time_samp / valid_trials;
                    double avg_lanc_time = sum_time_lanc / valid_trials;

                    // Large scale classical baseline runs: 10^4 - 10^5 trials
                    auto start_classical = std::chrono::high_resolution_clock::now();
                    std::mt19937_64 classical_rng(bench_cfg.seed_base ^ (trap_idx * 10007ULL));
                    ClassicalRateResult kick_rate = run_large_scale_kick_sd_rate(
                        current_seed, seed_energy, kick_d, bench_cfg.classical_trials_per_d, classical_rng);
                    ClassicalRateResult restart_rate = run_large_scale_restart_sd_rate(
                        n_spins, seed_energy, bench_cfg.restart_trials, classical_rng);
                    auto end_classical = std::chrono::high_resolution_clock::now();
                    double time_classical = std::chrono::duration<double>(end_classical - start_classical).count();
                    double avg_total_time = avg_samp_time + avg_lanc_time + (time_classical / valid_trials);

                    // Log to detailed CSV
                    for (const auto &tr : trial_records)
                    {
                        csv << n_spins << ","
                            << current_g << ","
                            << rank << ","
                            << min_rec.id << ","
                            << seed_energy << ","
                            << tr.sample_seed << ","
                            << scan.k_steps << ","
                            << scan.m_samples << ","
                            << scan.dt << ","
                            << tr.quantum_shots << ","
                            << tr.mean_flips << ","
                            << tr.kick_d << ","
                            << tr.budget_B << ","
                            << tr.subspace_dim << ","
                            << tr.best_classical_in_subspace << ","
                            << tr.budget_B << ","
                            << (tr.best_classical_in_subspace < seed_energy ? 1 : 0) << ","
                            << tr.quench_energy << ","
                            << tr.quench_evals << ","
                            << (tr.quench_energy < seed_energy ? 1 : 0) << ","
                            << kick_rate.per_eval_rate << ","
                            << kick_rate.effective_budget_succ_rate(tr.budget_B) << ","
                            << kick_rate.min_energy << ","
                            << restart_rate.per_eval_rate << ","
                            << restart_rate.effective_budget_succ_rate(tr.budget_B) << ","
                            << restart_rate.min_energy << ","
                            << tr.blind_energy << ","
                            << tr.blind_evals << ","
                            << (tr.blind_energy < seed_energy ? 1 : 0) << ","
                            << tr.time_sample << ","
                            << tr.time_sub << ","
                            << (time_classical / valid_trials) << ","
                            << (tr.time_sample + tr.time_sub + (time_classical / valid_trials)) << "\n";
                    }
                    csv.flush();

                    double skqd_per_eval_r = (stats_skqd.sum_evals > 0) ? (static_cast<double>(stats_skqd.improved_hits) / stats_skqd.sum_evals) : 0.0;
                    double quench_per_eval_r = (stats_quench.sum_evals > 0) ? (static_cast<double>(stats_quench.improved_hits) / stats_quench.sum_evals) : 0.0;
                    double blind_per_eval_r = (blind_stats.sum_evals > 0) ? (static_cast<double>(blind_stats.improved_hits) / blind_stats.sum_evals) : 0.0;

                    // Log to summary CSV
                    summary_csv << n_spins << ","
                                << current_g << ","
                                << scan.k_steps << ","
                                << scan.m_samples << ","
                                << scan.dt << ","
                                << rank << ","
                                << seed_energy << ","
                                << known_opt << ","
                                << (scan.k_steps * scan.m_samples) << ","
                                << avg_flips << ","
                                << avg_budget << ","
                                // SKQD
                                << stats_skqd.min_energy << ","
                                << (100.0 * stats_skqd.global_opt_hits / valid_trials) << ","
                                << (100.0 * stats_skqd.improved_hits / valid_trials) << ","
                                << (seed_energy - (stats_skqd.sum_energy / valid_trials)) << ","
                                << (stats_skqd.sum_evals / valid_trials) << ","
                                << skqd_per_eval_r << ","
                                // Quench
                                << stats_quench.min_energy << ","
                                << (100.0 * stats_quench.global_opt_hits / valid_trials) << ","
                                << (100.0 * stats_quench.improved_hits / valid_trials) << ","
                                << (seed_energy - (stats_quench.sum_energy / valid_trials)) << ","
                                << (stats_quench.sum_evals / valid_trials) << ","
                                << quench_per_eval_r << ","
                                // Kick
                                << kick_rate.min_energy << ","
                                << (kick_rate.min_energy <= known_opt ? 100.0 : 0.0) << ","
                                << (kick_rate.effective_budget_succ_rate(mean_b) * 100.0) << ","
                                << (seed_energy - kick_rate.mean_final_energy) << ","
                                << kick_rate.mean_evals_per_trial << ","
                                << kick_rate.per_eval_rate << ","
                                // Restart
                                << restart_rate.min_energy << ","
                                << (restart_rate.min_energy <= known_opt ? 100.0 : 0.0) << ","
                                << (restart_rate.effective_budget_succ_rate(mean_b) * 100.0) << ","
                                << (seed_energy - restart_rate.mean_final_energy) << ","
                                << restart_rate.mean_evals_per_trial << ","
                                << restart_rate.per_eval_rate << ","
                                // Blind
                                << blind_stats.min_energy << ","
                                << (100.0 * blind_stats.global_opt_hits / valid_trials) << ","
                                << (100.0 * blind_stats.improved_hits / valid_trials) << ","
                                << (seed_energy - (blind_stats.sum_energy / valid_trials)) << ","
                                << (blind_stats.sum_evals / valid_trials) << ","
                                << blind_per_eval_r << ","
                                // Timing
                                << avg_samp_time << ","
                                << avg_lanc_time << ","
                                << time_classical << ","
                                << avg_total_time << "\n";
                    summary_csv.flush();

                    // Print comparative table to console for this trap
                    print_trap_comparison_table(
                        rank, seed_energy, known_opt, n_spins, current_g,
                        scan.k_steps, scan.m_samples, scan.dt,
                        avg_flips, avg_budget, valid_trials,
                        stats_skqd, stats_quench, kick_rate, restart_rate, blind_stats);

                    overall_trap_records.push_back({
                        n_spins, current_g, scan.k_steps, scan.m_samples, scan.dt,
                        rank, seed_energy, known_opt, avg_flips, avg_budget,
                        stats_skqd, stats_quench, kick_rate, restart_rate, blind_stats,
                        valid_trials, avg_total_time
                    });
                }
            }
        }
        }

        auto end_n_time = std::chrono::high_resolution_clock::now();
        double elapsed_n = std::chrono::duration<double>(end_n_time - start_n_time).count();
        std::cout << "\n[TIMING] Completed System Size N=" << n_spins << " in " 
                  << std::fixed << std::setprecision(1) << elapsed_n << " seconds.\n";

        // Conditional timing check for multi-N runs (e.g. run N=20 only if N=24 finished fast enough)
        if (bench_cfg.conditional_n_timeout_seconds > 0.0 && n_idx + 1 < bench_cfg.n_values.size()) {
            if (elapsed_n > bench_cfg.conditional_n_timeout_seconds) {
                std::cout << "\n[CONDITIONAL TIMEOUT] System Size N=" << n_spins 
                          << " elapsed time (" << elapsed_n << "s) exceeded threshold (" 
                          << bench_cfg.conditional_n_timeout_seconds 
                          << "s). Skipping subsequent system size N=" 
                          << bench_cfg.n_values[n_idx + 1] << ".\n" << std::endl;
                break;
            } else {
                std::cout << "[CONDITIONAL TIMING] N=" << n_spins << " finished fast enough (" 
                          << elapsed_n << "s <= " << bench_cfg.conditional_n_timeout_seconds 
                          << "s threshold). Proceeding to N=" << bench_cfg.n_values[n_idx + 1] << "!\n";
            }
        }
    }

    if (is_delayed_window)
    {
        for (size_t n_val : bench_cfg.n_values) {
            for (double g_val : bench_cfg.g_values) {
                std::vector<DelayedTrapSummaryRecord> g_records;
                for (const auto &r : overall_delayed_records) {
                    if (r.n_spins == n_val && std::abs(r.g - g_val) < 1e-6) {
                        g_records.push_back(r);
                    }
                }
                if (!g_records.empty()) {
                    print_delayed_window_summary_report(g_records, n_val, g_val);
                }
            }
        }
    }
    else
    {
        std::cout << "\n========================================================================================================================================\n";
        std::cout << "                                             SKQD BENCHMARK COMPREHENSIVE SUMMARY REPORT\n";
        std::cout << "                         Comparative Analysis by Trap Depth Level (Stratified across Local Traps)\n";
        std::cout << "========================================================================================================================================\n";
        std::cout << "   N |   g   |  Trap Level (E0) | Best E (Quench) | Known E* | Quench Opt % | Kick Opt % | Quench Succ % | Kick Succ % | Budget B\n";
        std::cout << "-----+-------+------------------+-----------------+----------+--------------+------------+---------------+-------------+----------\n";
        for (const auto &rec : overall_trap_records)
        {
            double q_opt = rec.num_seeds > 0 ? (100.0 * rec.quench.global_opt_hits / rec.num_seeds) : 0.0;
            double k_opt = (rec.kick.min_energy <= rec.known_opt) ? 100.0 : 0.0;
            double q_succ = rec.num_seeds > 0 ? (100.0 * rec.quench.improved_hits / rec.num_seeds) : 0.0;
            size_t b_evals = static_cast<size_t>(std::round(rec.mean_budget));
            double k_succ = rec.kick.effective_budget_succ_rate(b_evals) * 100.0;

            std::string trap_str = "Rank #" + std::to_string(rec.trap_rank) + " (E=" + std::to_string(rec.seed_energy) + ")";
            std::cout << "  " << std::setw(2) << rec.n_spins << " | "
                      << std::setw(5) << std::fixed << std::setprecision(2) << rec.g << " | "
                      << std::setw(16) << trap_str << " | "
                      << std::setw(15) << rec.quench.min_energy << " | "
                      << std::setw(8) << rec.known_opt << " | "
                      << std::setw(11) << std::fixed << std::setprecision(1) << q_opt << "% | "
                      << std::setw(9) << std::fixed << std::setprecision(1) << k_opt << "% | "
                      << std::setw(12) << std::fixed << std::setprecision(1) << q_succ << "% | "
                      << std::setw(10) << std::fixed << std::setprecision(1) << k_succ << "% | "
                      << std::setw(8) << std::fixed << std::setprecision(0) << rec.mean_budget << "\n";
        }
        std::cout << "========================================================================================================================================\n" << std::endl;
    }

    csv.close();
    summary_csv.close();
    std::cout << "[TRACE] Benchmark complete. Results written to " << csv_filename 
              << " and " << summary_csv_filename << "." << std::endl;
    return 0;
}