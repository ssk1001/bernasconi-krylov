// ============================================================================
// DEPRECATED / REDUNDANT
// This legacy executor is obsolete and will be replaced by new executors.
// Superseded components:
//   - Classical: src/classical/classical.cuh
//   - Diagonalizer: src/subspace/diagonalizer.cuh
//   - Sampler: src/sampling/sampler.cuh
// ============================================================================

#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <filesystem>
#include <unordered_set>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

#include "classical/sd_collector.cuh"
#include "classical/seed_manager.hpp"
#include "classical/symmetry.hpp"
#include "sampling/sampler.cuh"

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
}

#define CHECK_CUSTATEVEC(call) { \
    custatevecStatus_t status = call; \
    if (status != CUSTATEVEC_STATUS_SUCCESS) \
        throw std::runtime_error("cuStateVec error " + std::to_string(status) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
}

// Warp reduction helper
__inline__ __device__ double warp_reduce_sum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Portable atomicAdd for double on architectures with __CUDA_ARCH__ < 600
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Sum |c_s|^2 strictly where E(s) < seed_energy
static __global__ void dynamics_sum_improved_prob_kernel(
    const cuDoubleComplex* __restrict__ d_state,
    const double* __restrict__ d_energies,
    double seed_energy,
    uint64_t num_states,
    double* d_out_prob)
{
    __shared__ double warp_sums[32];
    double thread_sum = 0.0;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;

    for (uint64_t i = idx; i < num_states; i += stride) {
        if (d_energies[i] < seed_energy) {
            cuDoubleComplex c = d_state[i];
            thread_sum += (c.x * c.x + c.y * c.y);
        }
    }

    double warp_sum = warp_reduce_sum(thread_sum);
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = warp_sum;
    __syncthreads();

    if (warp_id == 0) {
        double block_sum = (lane < (blockDim.x / warpSize)) ? warp_sums[lane] : 0.0;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) atomicAdd(d_out_prob, block_sum);
    }
}

// Count number of configurations with E(s) < seed_energy for P_blind
__global__ void count_improved_states_kernel(
    const double* __restrict__ d_energies,
    double seed_energy,
    uint64_t num_states,
    unsigned long long* d_out_count)
{
    __shared__ unsigned long long warp_sums[32];
    unsigned long long thread_count = 0;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)blockDim.x * gridDim.x;

    for (uint64_t i = idx; i < num_states; i += stride) {
        if (d_energies[i] < seed_energy) thread_count++;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        thread_count += __shfl_down_sync(0xffffffff, thread_count, offset);

    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = thread_count;
    __syncthreads();

    if (warp_id == 0) {
        unsigned long long block_sum = (lane < (blockDim.x / warpSize)) ? warp_sums[lane] : 0ULL;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        if (lane == 0) atomicAdd(d_out_count, block_sum);
    }
}

inline long long get_known_optimum(size_t n) {
    if (n == 16) return 24;
    if (n == 20) return 26;
    if (n == 22) return 39;
    if (n == 24) return 36;
    if (n == 26) return 45;
    if (n == 28) return 50;
    if (n == 30) return 59;
    return -1;
}

struct DynamicsConfig {
    size_t n = 16;
    std::vector<double> g_values = {0.25, 0.5, 1.0, 2.0};
    double dt = 0.05;
    double t_max = 60.0;
    size_t pool_minima_count = 250;
    size_t top_minima_count = 1;
    std::vector<size_t> trap_ranks;
    std::string seeds_dir = "seeds";
    bool compute_transverse = false;
    std::string csv_output = "data_processed/probability_dynamics_g_scan.csv";
    std::string csv_summary = "data_processed/probability_dynamics_summary.csv";

    void load(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) return;
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            k.erase(remove_if(k.begin(), k.end(), ::isspace), k.end());
            v.erase(0, v.find_first_not_of(" \t"));
            v.erase(v.find_last_not_of(" \t\r\n") + 1);

            if (k == "n") n = std::stoul(v);
            else if (k == "dt") dt = std::stod(v);
            else if (k == "t_max") t_max = std::stod(v);
            else if (k == "pool_minima_count") pool_minima_count = std::stoul(v);
            else if (k == "top_minima_count" || k == "num_seeds") top_minima_count = std::stoul(v);
            else if (k == "csv_output" || k == "csv_file") csv_output = v;
            else if (k == "csv_summary" || k == "summary_csv") csv_summary = v;
            else if (k == "seeds_dir" || k == "seed_dir") seeds_dir = v;
            else if (k == "compute_transverse" || k == "eval_transverse") {
                std::string lower_v = v;
                std::transform(lower_v.begin(), lower_v.end(), lower_v.begin(), ::tolower);
                compute_transverse = (lower_v == "1" || lower_v == "true" || lower_v == "yes");
            }
            else if (k == "trap_ranks" || k == "traps") {
                trap_ranks.clear();
                std::stringstream ss(v);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
                    if (!item.empty()) trap_ranks.push_back(std::stoul(item));
                }
            }
            else if (k == "g_values" || k == "g") {
                g_values.clear();
                std::stringstream ss(v);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
                    if (!item.empty()) g_values.push_back(std::stod(item));
                }
            }
        }
    }
};

struct DynamicsSummaryRecord {
    size_t n;
    size_t trap_rank;
    size_t seed_id;
    long long seed_energy;
    long long known_opt;
    double g;
    double t_max;
    double dt;
    double p_blind;
    double mean_p_full;
    double peak_p_full;
    double peak_time;
    double mean_p_transv;
    double peak_p_transv;
    double elapsed_time_s;
};

int main(int argc, char** argv) {
    DynamicsConfig cfg;
    if (argc > 1) {
        std::string a = argv[1];
        if (std::filesystem::exists(a)) cfg.load(a);
        else if (std::filesystem::exists("configs/" + a)) cfg.load("configs/" + a);
        else if (std::filesystem::exists("../configs/" + a)) cfg.load("../configs/" + a);
        else if (std::filesystem::exists("../" + a)) cfg.load("../" + a);
    } else if (std::filesystem::exists("configs/probability_scan.cfg")) {
        cfg.load("configs/probability_scan.cfg");
    } else if (std::filesystem::exists("../configs/probability_scan.cfg")) {
        cfg.load("../configs/probability_scan.cfg");
    }

    std::cout << "======================================================================\n";
    std::cout << "                PROBABILITY DYNAMICS EVOLUTION & SCAN\n";
    std::cout << "======================================================================\n";
    std::cout << "  System Size N       : " << cfg.n << "\n";
    std::cout << "  Time Horizon        : t in [0, " << std::fixed << std::setprecision(1) << cfg.t_max << "] (dt = " << std::setprecision(4) << cfg.dt << ")\n";
    std::cout << "  Transverse g Values : ";
    for (size_t i = 0; i < cfg.g_values.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << cfg.g_values[i] << (i + 1 < cfg.g_values.size() ? ", " : "\n");
    }
    std::cout << "  Seeds Directory     : " << cfg.seeds_dir << "\n";
    std::cout << "  Evaluate Transverse : " << (cfg.compute_transverse ? "Yes (H_X alone enabled)" : "No (Full H only; 2x faster)") << "\n";
    std::cout << "  Output Trajectory   : " << cfg.csv_output << "\n";
    std::cout << "  Output Summary      : " << cfg.csv_summary << "\n";
    std::cout << "======================================================================\n" << std::endl;

    uint64_t num_states = 1ULL << cfg.n;
    size_t state_bytes = num_states * sizeof(cuDoubleComplex);
    size_t energy_bytes = num_states * sizeof(double);
    size_t k_steps = static_cast<size_t>(std::round(cfg.t_max / cfg.dt));

    // Allocate GPU buffers
    cuDoubleComplex *d_state = nullptr;
    double *d_energies = nullptr, *d_out_prob = nullptr;
    unsigned long long *d_out_count = nullptr;
    CHECK_CUDA(cudaMalloc(&d_state, state_bytes));
    CHECK_CUDA(cudaMalloc(&d_energies, energy_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_prob, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_out_count, sizeof(unsigned long long)));

    int threads = 256;
    int blocks = static_cast<int>((num_states + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024; // Cap reduction grid

    // Precompute Bernasconi energies on GPU
    std::cout << "[Dynamics] Precomputing " << num_states << " diagonal energies on GPU..." << std::endl;
    precompute_labs_energies_kernel<<<(num_states + 255) / 256, 256>>>(d_energies, num_states, cfg.n);
    CHECK_CUDA(cudaDeviceSynchronize());

    custatevecHandle_t handle;
    CHECK_CUSTATEVEC(custatevecCreate(&handle));

    // Collect pool of distinct SD minima (or load cached seeds)
    SpinCache cache;
    bool seeds_were_cached = false;
    auto all_minima = SeedManager::get_or_collect_minima(
        cfg.n, cfg.pool_minima_count, cfg.seeds_dir, false, cache, seeds_were_cached);

    // Sort ascending by energy (lowest energy = best minima)
    std::sort(all_minima.begin(), all_minima.end(), [](const auto& a, const auto& b) {
        return a.energy < b.energy;
    });

    long long known_opt = get_known_optimum(cfg.n);
    std::vector<SDMinimumRecord> canonical_orbits;
    std::unordered_set<SpinState, SpinStateHash> seen_canonical_seeds;

    for (const auto &m : all_minima) {
        if (m.energy > known_opt) {
            SpinState canon = SpinSymmetry::get_canonical_state(m.state);
            if (seen_canonical_seeds.find(canon) == seen_canonical_seeds.end()) {
                seen_canonical_seeds.insert(canon);
                canonical_orbits.push_back(m);
            }
        }
    }

    // Fallback if none found strictly above known_opt
    if (canonical_orbits.empty() && !all_minima.empty()) {
        for (const auto &m : all_minima) {
            canonical_orbits.push_back(m);
        }
    }

    struct SelectedTrap {
        size_t rank;
        SDMinimumRecord rec;
    };
    std::vector<SelectedTrap> selected_traps;

    if (!cfg.trap_ranks.empty()) {
        for (size_t rank : cfg.trap_ranks) {
            if (rank >= 1 && rank <= canonical_orbits.size()) {
                selected_traps.push_back({rank, canonical_orbits[rank - 1]});
            } else {
                std::cerr << "[Dynamics Warning] Requested trap rank " << rank 
                          << " is out of range [1, " << canonical_orbits.size() << "]\n";
            }
        }
    } else {
        size_t count = std::min(cfg.top_minima_count, canonical_orbits.size());
        for (size_t i = 0; i < count; ++i) {
            selected_traps.push_back({i + 1, canonical_orbits[i]});
        }
    }

    std::cout << "[Dynamics] Selected " << selected_traps.size() << " symmetry-distinct local minima (traps):\n";
    for (const auto& trap : selected_traps) {
        std::cout << "  Rank #" << std::setw(2) << trap.rank 
                  << ": Energy E0 = " << std::setw(3) << trap.rec.energy 
                  << " | Orbit MinID = " << trap.rec.id << "\n";
    }

    // Ensure output directories exist
    std::filesystem::path out_path(cfg.csv_output);
    if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
    std::ofstream csv(cfg.csv_output);
    csv << "N,trap_rank,seed_id,seed_energy,g,t,P_full,P_transverse,P_blind\n";

    std::vector<DynamicsSummaryRecord> summary_records;

    for (const auto& trap : selected_traps) {
        size_t trap_rank = trap.rank;
        const auto& seed_rec = trap.rec;
        double seed_energy = static_cast<double>(seed_rec.energy);

        // Map seed configuration to statevector basis index
        uint64_t seed_basis_idx = 0;
        for (size_t i = 0; i < cfg.n; ++i) {
            if (seed_rec.state.get_spin(i) == -1) seed_basis_idx |= (1ULL << i);
        }

        // 1. Exact Blind Random baseline: fraction of states with E < seed_energy
        unsigned long long count_improved = 0;
        CHECK_CUDA(cudaMemset(d_out_count, 0, sizeof(unsigned long long)));
        count_improved_states_kernel<<<blocks, threads>>>(d_energies, seed_energy, num_states, d_out_count);
        CHECK_CUDA(cudaMemcpy(&count_improved, d_out_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        double p_blind = static_cast<double>(count_improved) / static_cast<double>(num_states);

        std::cout << "\n========================================================================================\n";
        std::cout << " [DYNAMICS] Trap Rank #" << trap_rank 
                  << " (SD ID: " << seed_rec.id 
                  << ", E0 = " << seed_rec.energy 
                  << ", Known E* = " << known_opt 
                  << ", P_blind = " << std::scientific << std::setprecision(3) << p_blind << ")\n";
        std::cout << "========================================================================================" << std::endl;

        for (double g_val : cfg.g_values) {
            std::cout << "  --> Evolving g = " << std::fixed << std::setprecision(2) << g_val << "..." << std::flush;
            auto start_eval = std::chrono::high_resolution_clock::now();

            double rx_angle = -2.0 * g_val * cfg.dt;
            cuDoubleComplex rx_matrix[4] = {
                make_cuDoubleComplex(cos(rx_angle / 2.0), 0.0),
                make_cuDoubleComplex(0.0, -sin(rx_angle / 2.0)),
                make_cuDoubleComplex(0.0, -sin(rx_angle / 2.0)),
                make_cuDoubleComplex(cos(rx_angle / 2.0), 0.0)
            };

            std::vector<double> p_full_t(k_steps + 1, 0.0);
            std::vector<double> p_transv_t(k_steps + 1, 0.0);

            // A. Full Quantum Evolution: H = H_B / N - g sum(X)
            CHECK_CUDA(cudaMemset(d_state, 0, state_bytes));
            cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
            CHECK_CUDA(cudaMemcpy(d_state + seed_basis_idx, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

            double sum_p_full = 0.0;
            double peak_p_full = 0.0;
            double peak_time = 0.0;

            for (size_t k = 1; k <= k_steps; ++k) {
                apply_precomputed_phase_kernel<<<(num_states + 255) / 256, 256>>>(d_state, d_energies, num_states, cfg.n, cfg.dt);
                for (int32_t q = 0; q < static_cast<int32_t>(cfg.n); ++q) {
                    CHECK_CUSTATEVEC(custatevecApplyMatrix(
                        handle, d_state, CUDA_C_64F, static_cast<uint32_t>(cfg.n),
                        rx_matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                        0, &q, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
                }
                CHECK_CUDA(cudaMemset(d_out_prob, 0, sizeof(double)));
                dynamics_sum_improved_prob_kernel<<<blocks, threads>>>(d_state, d_energies, seed_energy, num_states, d_out_prob);
                double p_val = 0.0;
                CHECK_CUDA(cudaMemcpy(&p_val, d_out_prob, sizeof(double), cudaMemcpyDeviceToHost));
                p_full_t[k] = p_val;
                sum_p_full += p_val;
                if (p_val > peak_p_full) {
                    peak_p_full = p_val;
                    peak_time = k * cfg.dt;
                }
            }
            double mean_p_full = (k_steps > 0) ? (sum_p_full / static_cast<double>(k_steps)) : 0.0;

            // B. Transverse Field Alone: H_X = -g sum(X) (Optional)
            double sum_p_transv = 0.0;
            double peak_p_transv = 0.0;
            if (cfg.compute_transverse) {
                CHECK_CUDA(cudaMemset(d_state, 0, state_bytes));
                CHECK_CUDA(cudaMemcpy(d_state + seed_basis_idx, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

                for (size_t k = 1; k <= k_steps; ++k) {
                    for (int32_t q = 0; q < static_cast<int32_t>(cfg.n); ++q) {
                        CHECK_CUSTATEVEC(custatevecApplyMatrix(
                            handle, d_state, CUDA_C_64F, static_cast<uint32_t>(cfg.n),
                            rx_matrix, CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW,
                            0, &q, 1, nullptr, nullptr, 0, CUSTATEVEC_COMPUTE_64F, nullptr, 0));
                    }
                    CHECK_CUDA(cudaMemset(d_out_prob, 0, sizeof(double)));
                    dynamics_sum_improved_prob_kernel<<<blocks, threads>>>(d_state, d_energies, seed_energy, num_states, d_out_prob);
                    double p_val = 0.0;
                    CHECK_CUDA(cudaMemcpy(&p_val, d_out_prob, sizeof(double), cudaMemcpyDeviceToHost));
                    p_transv_t[k] = p_val;
                    sum_p_transv += p_val;
                    if (p_val > peak_p_transv) peak_p_transv = p_val;
                }
            }
            double mean_p_transv = (k_steps > 0 && cfg.compute_transverse) ? (sum_p_transv / static_cast<double>(k_steps)) : 0.0;

            auto end_eval = std::chrono::high_resolution_clock::now();
            double elapsed_s = std::chrono::duration<double>(end_eval - start_eval).count();

            // Write trajectory to CSV
            for (size_t k = 0; k <= k_steps; ++k) {
                double t = k * cfg.dt;
                csv << cfg.n << "," << trap_rank << "," << seed_rec.id << "," << seed_rec.energy << ","
                    << g_val << "," << std::fixed << std::setprecision(4) << t << ","
                    << std::setprecision(8) << p_full_t[k] << "," << p_transv_t[k] << "," << p_blind << "\n";
            }
            csv.flush();

            std::cout << " Done. (" << std::fixed << std::setprecision(1) << elapsed_s << "s) | Mean P_<[0," 
                      << std::setprecision(0) << cfg.t_max << "]: " << std::scientific << std::setprecision(4) << mean_p_full 
                      << " | Peak P_<: " << peak_p_full 
                      << " (at t=" << std::fixed << std::setprecision(2) << peak_time << "s)" << std::endl;

            summary_records.push_back({
                cfg.n, trap_rank, seed_rec.id, seed_rec.energy, known_opt,
                g_val, cfg.t_max, cfg.dt, p_blind,
                mean_p_full, peak_p_full, peak_time,
                mean_p_transv, peak_p_transv, elapsed_s
            });
        }
    }

    // Write summary CSV
    std::filesystem::path sum_path(cfg.csv_summary);
    if (sum_path.has_parent_path()) std::filesystem::create_directories(sum_path.parent_path());
    std::ofstream sum_csv(cfg.csv_summary);
    sum_csv << "N,trap_rank,seed_id,seed_energy,known_opt,g,t_max,dt,P_blind,Mean_P_less_60,Peak_P_less,Peak_Time_s,Mean_P_transv,Peak_P_transv,Elapsed_s\n";
    for (const auto& r : summary_records) {
        sum_csv << r.n << "," << r.trap_rank << "," << r.seed_id << "," << r.seed_energy << "," << r.known_opt << ","
                << r.g << "," << r.t_max << "," << r.dt << ","
                << r.p_blind << "," << r.mean_p_full << "," << r.peak_p_full << "," << r.peak_time << ","
                << r.mean_p_transv << "," << r.peak_p_transv << "," << r.elapsed_time_s << "\n";
    }
    sum_csv.flush();

    // Print Consolidated Summary Table
    std::cout << "\n========================================================================================================================\n";
    std::cout << "               PROBABILITY DYNAMICS SUMMARY: System Size N = " << cfg.n 
              << " (t in [0.0, " << std::fixed << std::setprecision(1) << cfg.t_max 
              << "], dt = " << std::setprecision(4) << cfg.dt << ")\n";
    std::cout << "========================================================================================================================\n";
    std::cout << " Trap Rank | Seed E0 | Known E* |   g   |   P_blind   | Mean P_<(t) [0," 
              << std::setprecision(0) << cfg.t_max << "] |  Peak P_<(t) | Peak Time (s) | Elapsed (s)\n";
    std::cout << "-----------+---------+----------+-------+-------------+-----------------------+--------------+---------------+------------\n";

    for (const auto& r : summary_records) {
        std::ostringstream ss_blind, ss_mean, ss_peak;
        ss_blind << std::scientific << std::setprecision(3) << r.p_blind;
        ss_mean << std::scientific << std::setprecision(4) << r.mean_p_full;
        ss_peak << std::scientific << std::setprecision(4) << r.peak_p_full;

        std::string trap_str = "Rank #" + std::to_string(r.trap_rank);
        std::cout << " " << std::left << std::setw(9) << trap_str << " | "
                  << std::right << std::setw(7) << r.seed_energy << " | "
                  << std::setw(8) << r.known_opt << " | "
                  << std::setw(5) << std::fixed << std::setprecision(2) << r.g << " | "
                  << std::setw(11) << ss_blind.str() << " | "
                  << std::setw(21) << ss_mean.str() << " | "
                  << std::setw(12) << ss_peak.str() << " | "
                  << std::setw(13) << std::fixed << std::setprecision(2) << r.peak_time << " | "
                  << std::setw(10) << std::fixed << std::setprecision(1) << r.elapsed_time_s << "\n";
    }
    std::cout << "========================================================================================================================\n";
    std::cout << "[Dynamics] Trajectory written to : " << cfg.csv_output << "\n";
    std::cout << "[Dynamics] Summary written to    : " << cfg.csv_summary << "\n\n";

    CHECK_CUSTATEVEC(custatevecDestroy(handle));
    CHECK_CUDA(cudaFree(d_state));
    CHECK_CUDA(cudaFree(d_energies));
    CHECK_CUDA(cudaFree(d_out_prob));
    CHECK_CUDA(cudaFree(d_out_count));

    return 0;
}
