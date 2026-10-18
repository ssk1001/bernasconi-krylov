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
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <custatevec.h>

#include "classical/sd_collector.cuh"
#include "classical/seed_manager.hpp"
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

// Sum |c_s|^2 strictly where E(s) < seed_energy
static __global__ void dynamics_sum_improved_prob_kernel(
    const cuDoubleComplex* __restrict__ d_state,
    const double* __restrict__ d_energies,
    double seed_energy,
    uint64_t num_states,
    double* __restrict__ d_out_prob)
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
    unsigned long long* __restrict__ d_out_count)
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
    return -1;
}

struct DynamicsConfig {
    size_t n = 16;
    std::vector<double> g_values = {0.25, 0.5, 1.0, 2.0};
    double dt = 0.05;
    double t_max = 10.0;
    size_t pool_minima_count = 50;  // Pool of SD minima to search over
    size_t top_minima_count = 1;   // Number of energy-wise best minima to evaluate
    std::string csv_output = "data_processed/probability_dynamics_g_scan.csv";
    std::string seeds_dir = "seeds";

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
            else if (k == "csv_output") csv_output = v;
            else if (k == "seeds_dir" || k == "seed_dir") seeds_dir = v;
            else if (k == "g_values") {
                g_values.clear();
                std::stringstream ss(v);
                std::string item;
                while (std::getline(ss, item, ','))
                    if (!item.empty()) g_values.push_back(std::stod(item));
            }
        }
    }
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

    std::cout << "[Dynamics] Starting g-scan for top " << cfg.top_minima_count 
              << " energy-wise best minima (pool=" << cfg.pool_minima_count 
              << ", N=" << cfg.n << ", dt=" << cfg.dt << ", t_max=" << cfg.t_max << ")...\n";

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
    std::vector<SDMinimumRecord> selected_minima;
    for (const auto& m : all_minima) {
        // Exclude exact global optimum if found, since no states exist with E < known_opt
        if (m.energy > known_opt) {
            selected_minima.push_back(m);
            if (selected_minima.size() >= cfg.top_minima_count) break;
        }
    }
    // Fallback if none found with E > known_opt
    if (selected_minima.empty() && !all_minima.empty()) {
        for (size_t i = 0; i < std::min(cfg.top_minima_count, all_minima.size()); ++i)
            selected_minima.push_back(all_minima[i]);
    }

    std::cout << "\n[Dynamics] Selected top " << selected_minima.size() << " energy-wise best local minima:\n";
    for (size_t i = 0; i < selected_minima.size(); ++i) {
        std::cout << "  Rank #" << (i + 1) << ": Energy = " << selected_minima[i].energy 
                  << " (SD ID: " << selected_minima[i].id << ")\n";
    }

    // Ensure output directory exists
    std::filesystem::path out_path(cfg.csv_output);
    if (out_path.has_parent_path()) std::filesystem::create_directories(out_path.parent_path());
    std::ofstream csv(cfg.csv_output);
    csv << "N,seed_id,seed_energy,g,t,P_full,P_transverse,P_blind\n";

    for (size_t s_idx = 0; s_idx < selected_minima.size(); ++s_idx) {
        const auto& seed_rec = selected_minima[s_idx];
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

        std::cout << "\n--- Minimum " << (s_idx + 1) << "/" << selected_minima.size() 
                  << " (E=" << seed_rec.energy << ", P_blind=" << std::scientific << std::setprecision(3) << p_blind << ") ---" << std::endl;

        for (double g_val : cfg.g_values) {
            std::cout << "  Evolving g = " << std::fixed << std::setprecision(2) << g_val << "..." << std::flush;

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
                CHECK_CUDA(cudaMemcpy(&p_full_t[k], d_out_prob, sizeof(double), cudaMemcpyDeviceToHost));
            }

            // B. Transverse Field Alone: H_X = -g sum(X)
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
                CHECK_CUDA(cudaMemcpy(&p_transv_t[k], d_out_prob, sizeof(double), cudaMemcpyDeviceToHost));
            }

            // Write trajectory to CSV
            for (size_t k = 0; k <= k_steps; ++k) {
                double t = k * cfg.dt;
                csv << cfg.n << "," << s_idx << "," << seed_rec.energy << ","
                    << g_val << "," << std::fixed << std::setprecision(4) << t << ","
                    << std::setprecision(8) << p_full_t[k] << "," << p_transv_t[k] << "," << p_blind << "\n";
            }
            std::cout << " Done. (Peak P_full=" << std::setprecision(4) 
                      << *std::max_element(p_full_t.begin(), p_full_t.end()) << ")\n";
        }
    }

    CHECK_CUSTATEVEC(custatevecDestroy(handle));
    CHECK_CUDA(cudaFree(d_state));
    CHECK_CUDA(cudaFree(d_energies));
    CHECK_CUDA(cudaFree(d_out_prob));
    CHECK_CUDA(cudaFree(d_out_count));

    std::cout << "\n[Dynamics] All trajectories recorded to: " << cfg.csv_output << std::endl;
    return 0;
}
