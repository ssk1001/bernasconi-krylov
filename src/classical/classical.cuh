#pragma once

#include <vector>
#include <unordered_map>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <chrono>
#include <array>
#include <string>
#include <filesystem>
#include <cctype>

#if defined(__CUDACC__) && __has_include(<cuda_runtime.h>)
    #define HAS_CUDA_CLASSICAL 1
    #include <cuda_runtime.h>
#else
    #define HAS_CUDA_CLASSICAL 0
#endif

#include "../labs_energy/labs_energy.hpp"

#if HAS_CUDA_CLASSICAL
// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
    } \
}
#endif
#endif

// ============================================================================
// Data Structures
// ============================================================================

/// Result of a single steepest descent trajectory
struct DescentResult {
    SpinState start_state;
    int64_t start_energy = 0;
    SpinState final_state;
    int64_t final_energy = 0;
    size_t evals_used = 0;
    size_t steps_taken = 0;
    bool reached_local_min = false;
    bool success = false;          // final_energy < start_energy
    double success_rate = 0.0;     // success ? 1.0 : 0.0 (1 trial)
    double per_eval_rate = 0.0;    // (success ? 1.0 : 0.0) / evals_used
};

/// Record for a discovered symmetry-distinct local minimum family
struct MinimumFamily {
    size_t id = 0;
    SpinState canonical_state;     // Canonical representative of the symmetry orbit
    SpinState sample_state;        // A sample converged state belonging to this orbit
    int64_t energy = 0;            // LABS energy E(s)
    size_t times_visited = 0;      // Frequency of descent convergence into this orbit (basin size proxy)
    size_t evals_to_first_find = 0;// Total classical energy evaluations spent until first discovery
};

/// Result of a budget-constrained search (Random Restarts or Random Kick)
struct BudgetedResult {
    SpinState og_state;
    int64_t og_energy = 0;
    SpinState best_state;
    int64_t best_energy = std::numeric_limits<int64_t>::max();
    size_t total_evals_used = 0;   // Number of evaluations taken
    size_t total_trajectories = 0; // Number of trials
    size_t improved_hits = 0;      // Successes (final_energy < og_energy)
    double success_rate = 0.0;     // improved_hits / total_trajectories
    double per_eval_rate = 0.0;    // improved_hits / total_evals_used
};

/// Result of multi-start steepest descent
struct MultiStartResult {
    SpinState og_state;
    int64_t og_energy = 0;
    size_t total_starts = 0;       // Number of trials
    size_t improved_hits = 0;      // Successes (final_energy < og_energy)
    size_t total_evals = 0;        // Number of evaluations taken
    double success_rate = 0.0;     // improved_hits / total_starts
    double per_eval_rate = 0.0;    // improved_hits / total_evals
    SpinState best_state;
    int64_t best_energy = std::numeric_limits<int64_t>::max();
    std::vector<SpinState> final_states;
    std::vector<int64_t> final_energies;
    std::vector<int64_t> eval_counts;
};

/// Batch output from the GPU multi-start kernel
struct BatchDescentResult {
    std::vector<SpinState> final_states;
    std::vector<int64_t> final_energies;
    std::vector<int64_t> eval_counts;
};

// ============================================================================
// Logging and Reconstruction Structures
// ============================================================================

enum class LogLevel {
    Silent,   // No console logs
    Summary,  // Final table and metrics
    Verbose   // Detailed batch-by-batch progress
};

struct ClassicalRunMetadata {
    int n = 0;
    uint64_t seed = 0;
    std::string algorithm_name;
    size_t total_starts = 0;
    size_t total_evals = 0;
    double elapsed_ms = 0.0;
    std::string device_name;
};

class ClassicalLogger {
public:
    /// Prints a rich terminal summary table of discovered minima families
    static void print_minima_summary(
        const std::vector<MinimumFamily>& families,
        const ClassicalRunMetadata& meta)
    {
        std::cout << "\n==================================================================================================\n";
        std::cout << "                         CLASSICAL MINIMA DISCOVERY SUMMARY (N=" << meta.n << ")\n";
        std::cout << "==================================================================================================\n";
        std::cout << "  Algorithm        : " << meta.algorithm_name << " [" << meta.device_name << "]\n";
        std::cout << "  RNG Seed         : " << meta.seed << "\n";
        std::cout << "  Total Starts     : " << meta.total_starts << "\n";
        std::cout << "  Total Evals      : " << meta.total_evals << " (Mean: "
                  << (meta.total_starts > 0 ? meta.total_evals / meta.total_starts : 0) << " evals/start)\n";
        std::cout << "  Distinct Families: " << families.size() << " symmetry orbits\n";
        std::cout << "  Elapsed Time     : " << std::fixed << std::setprecision(2) << meta.elapsed_ms << " ms\n";
        std::cout << "--------------------------------------------------------------------------------------------------\n";
        std::cout << " Rank | Energy | Basin % | Visits | Evals to Find | Canonical Hex       | Bitstring\n";
        std::cout << "------+--------+---------+--------+---------------+--------------------+---------------------------\n";

        for (const auto& fam : families) {
            double basin_pct = (meta.total_starts > 0)
                ? (100.0 * fam.times_visited / meta.total_starts) : 0.0;

            std::stringstream ss_hex;
            ss_hex << "0x" << std::hex << std::uppercase << fam.canonical_state.bits;

            std::string bitstr(meta.n, '0');
            for (int i = 0; i < meta.n; ++i) {
                if (fam.canonical_state.get_bit(i)) bitstr[meta.n - 1 - i] = '1';
            }

            std::cout << " " << std::setw(4) << std::right << fam.id << " | "
                      << std::setw(6) << fam.energy << " | "
                      << std::setw(6) << std::fixed << std::setprecision(2) << basin_pct << "% | "
                      << std::setw(6) << fam.times_visited << " | "
                      << std::setw(13) << fam.evals_to_first_find << " | "
                      << std::setw(18) << std::left << ss_hex.str() << " | "
                      << bitstr << "\n";
        }
        std::cout << "==================================================================================================\n\n";
    }

    /// Exports full run data to CSV for complete run reconstruction
    static bool export_to_csv(
        const std::string& filepath,
        const std::vector<MinimumFamily>& families,
        const ClassicalRunMetadata& meta)
    {
        try {
            std::filesystem::path p(filepath);
            if (p.has_parent_path()) {
                std::filesystem::create_directories(p.parent_path());
            }
        } catch (...) {}

        std::ofstream out(filepath);
        if (!out.is_open()) return false;

        out << "# N=" << meta.n << ",Seed=" << meta.seed
            << ",TotalStarts=" << meta.total_starts
            << ",TotalEvals=" << meta.total_evals
            << ",ElapsedMs=" << meta.elapsed_ms
            << ",Device=" << meta.device_name << "\n";
        out << "id,energy,times_visited,basin_fraction,evals_to_first_find,canonical_hex,sample_hex,canonical_bitstring\n";

        for (const auto& fam : families) {
            double basin_frac = (meta.total_starts > 0)
                ? (static_cast<double>(fam.times_visited) / meta.total_starts) : 0.0;

            std::string bitstr(meta.n, '0');
            for (int i = 0; i < meta.n; ++i) {
                if (fam.canonical_state.get_bit(i)) bitstr[meta.n - 1 - i] = '1';
            }

            out << fam.id << ","
                << fam.energy << ","
                << fam.times_visited << ","
                << std::fixed << std::setprecision(6) << basin_frac << ","
                << fam.evals_to_first_find << ","
                << "0x" << std::hex << std::uppercase << fam.canonical_state.bits << std::dec << ","
                << "0x" << std::hex << std::uppercase << fam.sample_state.bits << std::dec << ","
                << bitstr << "\n";
        }
        return true;
    }

    static inline std::string trim_string(const std::string& s) {
        size_t first = s.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) return "";
        size_t last = s.find_last_not_of(" \t\r\n");
        return s.substr(first, last - first + 1);
    }

    static inline bool is_bitstring(const std::string& s) {
        if (s.empty()) return false;
        for (char c : s) {
            if (c != '0' && c != '1') return false;
        }
        return true;
    }

    static inline bool is_hex_string(const std::string& s) {
        if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) {
            return s.size() > 2;
        }
        if (s.empty() || s.size() > 16) return false;
        bool has_alpha = false;
        for (char c : s) {
            if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
            if (std::isalpha(static_cast<unsigned char>(c))) has_alpha = true;
        }
        return has_alpha;
    }

    static inline SpinState parse_bitstring(const std::string& s) {
        uint64_t bits = 0;
        int len = static_cast<int>(s.size());
        for (int i = 0; i < len; ++i) {
            if (s[i] == '1') {
                bits |= (1ULL << (len - 1 - i));
            }
        }
        return SpinState{bits};
    }

    /// Loads discovered minima from CSV, supporting modern CSV, legacy formats, and raw bitstrings/hex.
    static std::vector<MinimumFamily> load_from_csv(const std::string& filepath, int n = 0) {
        std::vector<MinimumFamily> families;
        std::ifstream in(filepath);
        if (!in.is_open()) return families;

        std::unordered_map<uint64_t, size_t> seen_canonical;
        std::string line;

        while (std::getline(in, line)) {
            line = trim_string(line);
            if (line.empty()) continue;

            // Comment lines: check for metadata
            if (line[0] == '#' || line[0] == ';' || (line.size() >= 2 && line[0] == '/' && line[1] == '/')) {
                if (n == 0) {
                    size_t n_pos = line.find("N=");
                    if (n_pos == std::string::npos) n_pos = line.find("n=");
                    if (n_pos != std::string::npos) {
                        try {
                            n = std::stoi(line.substr(n_pos + 2));
                        } catch (...) {}
                    }
                }
                continue;
            }

            // Tokenize by comma if comma is present, otherwise by whitespace
            std::vector<std::string> cols;
            if (line.find(',') != std::string::npos) {
                std::stringstream ss(line);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    std::string trimmed = trim_string(tok);
                    if (!trimmed.empty() || cols.size() < 2) {
                        cols.push_back(trimmed);
                    }
                }
            } else {
                std::stringstream ss(line);
                std::string tok;
                while (ss >> tok) {
                    cols.push_back(trim_string(tok));
                }
            }

            if (cols.empty()) continue;

            // Header line detection
            bool is_header = false;
            for (const auto& tok : cols) {
                std::string lower_tok = tok;
                std::transform(lower_tok.begin(), lower_tok.end(), lower_tok.begin(),
                               [](unsigned char c) { return std::tolower(c); });
                if (lower_tok == "id" || lower_tok == "energy" || lower_tok == "canonical_hex" ||
                    lower_tok == "sample_hex" || lower_tok == "bitstring" || lower_tok == "canonical_bitstring" ||
                    lower_tok == "basin_fraction" || lower_tok == "times_visited" || lower_tok == "visits" ||
                    lower_tok == "evals_to_first_find" || lower_tok == "evals" || lower_tok == "state") {
                    is_header = true;
                    break;
                }
            }
            if (is_header) continue;

            SpinState raw_state{0};
            SpinState sample_state{0};
            int64_t energy = 0;
            size_t visits = 1;
            size_t evals = 0;
            bool parsed = false;

            // 1. Standard modern CSV (>= 7 columns)
            // schema: id, energy, times_visited, basin_fraction, evals_to_first_find, canonical_hex, sample_hex, [canonical_bitstring]
            if (cols.size() >= 7 && (cols[5].rfind("0x", 0) == 0 || cols[5].rfind("0X", 0) == 0 ||
                                     cols[6].rfind("0x", 0) == 0 || cols[6].rfind("0X", 0) == 0 ||
                                     is_hex_string(cols[5]) || is_bitstring(cols[5]))) {
                try {
                    raw_state = is_bitstring(cols[5]) ? parse_bitstring(cols[5]) : SpinState{std::stoull(cols[5], nullptr, 0)};
                    sample_state = is_bitstring(cols[6]) ? parse_bitstring(cols[6]) : SpinState{std::stoull(cols[6], nullptr, 0)};
                    energy = std::stoll(cols[1]);
                    visits = std::stoul(cols[2]);
                    evals = std::stoul(cols[4]);
                    if (cols.size() >= 8 && is_bitstring(cols[7])) {
                        if (n == 0) n = static_cast<int>(cols[7].size());
                    }
                    parsed = true;
                } catch (...) {}
            }

            // 2. Legacy schema (>= 3 columns): e.g. id, energy, bitstring/hex, [evals, start_index]
            if (!parsed && cols.size() >= 3) {
                if (is_bitstring(cols[2])) {
                    raw_state = parse_bitstring(cols[2]);
                    if (n == 0) n = static_cast<int>(cols[2].size());
                    sample_state = raw_state;
                    try { energy = std::stoll(cols[1]); } catch (...) {}
                    if (cols.size() >= 4) {
                        try { evals = std::stoul(cols[3]); } catch (...) {}
                    }
                    parsed = true;
                } else if (is_hex_string(cols[2]) || cols[2].rfind("0x", 0) == 0 || cols[2].rfind("0X", 0) == 0) {
                    try {
                        raw_state = SpinState{std::stoull(cols[2], nullptr, 0)};
                        sample_state = raw_state;
                        energy = std::stoll(cols[1]);
                        if (cols.size() >= 4) {
                            try { evals = std::stoul(cols[3]); } catch (...) {}
                        }
                        parsed = true;
                    } catch (...) {}
                }
            }

            // 3. 2-column format: (energy, state) or (state, energy) or (id, state)
            if (!parsed && cols.size() == 2) {
                for (int s_idx = 0; s_idx < 2; ++s_idx) {
                    int other_idx = 1 - s_idx;
                    if (is_bitstring(cols[s_idx])) {
                        raw_state = parse_bitstring(cols[s_idx]);
                        if (n == 0) n = static_cast<int>(cols[s_idx].size());
                        sample_state = raw_state;
                        try { energy = std::stoll(cols[other_idx]); } catch (...) {}
                        parsed = true;
                        break;
                    } else if (is_hex_string(cols[s_idx]) || cols[s_idx].rfind("0x", 0) == 0 || cols[s_idx].rfind("0X", 0) == 0) {
                        try {
                            raw_state = SpinState{std::stoull(cols[s_idx], nullptr, 0)};
                            sample_state = raw_state;
                            energy = std::stoll(cols[other_idx]);
                            parsed = true;
                            break;
                        } catch (...) {}
                    }
                }
            }

            // 4. 1-column format: raw bitstring or hex per line
            if (!parsed && cols.size() == 1) {
                if (is_bitstring(cols[0])) {
                    raw_state = parse_bitstring(cols[0]);
                    if (n == 0) n = static_cast<int>(cols[0].size());
                    sample_state = raw_state;
                    parsed = true;
                } else if (is_hex_string(cols[0]) || cols[0].rfind("0x", 0) == 0 || cols[0].rfind("0X", 0) == 0) {
                    try {
                        raw_state = SpinState{std::stoull(cols[0], nullptr, 0)};
                        sample_state = raw_state;
                        parsed = true;
                    } catch (...) {}
                }
            }

            if (!parsed) continue;

            SpinState canon = (n > 0) ? raw_state.canonical(n) : raw_state;
            if (sample_state.bits == 0) sample_state = raw_state;

            auto it = seen_canonical.find(canon.bits);
            if (it != seen_canonical.end()) {
                families[it->second].times_visited += visits;
            } else {
                seen_canonical[canon.bits] = families.size();
                MinimumFamily fam;
                fam.id = families.size() + 1;
                fam.canonical_state = canon;
                fam.sample_state = sample_state;
                fam.energy = energy;
                fam.times_visited = visits;
                fam.evals_to_first_find = evals;
                families.push_back(fam);
            }
        }

        // Compute energies if missing/zero and n is known
        if (n > 0) {
            LabsEvaluator eval(n);
            for (auto& fam : families) {
                if (fam.energy == 0) {
                    fam.energy = eval.calcEnergy(fam.canonical_state);
                }
            }
        }

        // Sort ascending by energy, then descending by times_visited
        std::sort(families.begin(), families.end(), [](const MinimumFamily& a, const MinimumFamily& b) {
            if (a.energy != b.energy) return a.energy < b.energy;
            return a.times_visited > b.times_visited;
        });

        for (size_t i = 0; i < families.size(); ++i) {
            families[i].id = i + 1;
        }

        return families;
    }

    /// Alias for load_from_csv
    static inline std::vector<MinimumFamily> load_from_file(const std::string& filepath, int n = 0) {
        return load_from_csv(filepath, n);
    }
};

// ============================================================================
// CUDA Kernel for Batch Multi-Start Steepest Descent
// ============================================================================

#if HAS_CUDA_CLASSICAL
static __global__ void batch_steepest_descent_kernel(
    int n,
    int num_starts,
    const uint64_t* __restrict__ d_start_states,
    uint64_t* __restrict__ d_final_states,
    int64_t* __restrict__ d_final_energies,
    int64_t* __restrict__ d_eval_counts,
    size_t max_evals_per_start)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_starts) return;

    uint64_t full_mask = (n >= 64) ? ~0ULL : ((1ULL << n) - 1ULL);
    uint64_t my_state = d_start_states[tid] & full_mask;

    // Fast register-resident storage for C_k (fits in on-chip registers)
    int16_t C[64];
    int64_t local_energy = 0;

    // 1. Bit-parallel Popcount initialization of C_k in pure hardware instructions
    for (int k = 1; k < n; ++k) {
        uint64_t mask_k = (1ULL << (n - k)) - 1ULL;
        int diffs = __popcll((my_state ^ (my_state >> k)) & mask_k);
        C[k] = static_cast<int16_t>((n - k) - 2 * diffs);
        local_energy += static_cast<int64_t>(C[k]) * C[k];
    }

    int64_t evals = 1;
    bool improved = true;

    int16_t best_delta_C[64];
    int16_t delta_C[64];

    // 2. Incremental steepest descent loop
    uint64_t eff_max = (max_evals_per_start == 0) ? ~0ULL : max_evals_per_start;
    while (improved) {
        if (evals + n > eff_max) break;

        evals += n;
        improved = false;
        int64_t best_neighbor_energy = local_energy;
        int best_bit = -1;

        // Probe all N single-bit flips incrementally
        for (int p = 0; p < n; ++p) {
            int s_p = ((my_state >> p) & 1ULL) ? -1 : 1;
            int64_t neighbor_e = 0;

            for (int k = 1; k < n; ++k) {
                int delta_k = 0;
                if (p + k < n) {
                    int s_pk = ((my_state >> (p + k)) & 1ULL) ? -1 : 1;
                    delta_k += -2 * (s_p * s_pk);
                }
                if (p - k >= 0) {
                    int s_pmk = ((my_state >> (p - k)) & 1ULL) ? -1 : 1;
                    delta_k += -2 * (s_pmk * s_p);
                }
                delta_C[k] = static_cast<int16_t>(delta_k);
                int64_t new_ck = C[k] + delta_k;
                neighbor_e += new_ck * new_ck;
            }

            if (neighbor_e < best_neighbor_energy) {
                best_neighbor_energy = neighbor_e;
                best_bit = p;
                for (int k = 1; k < n; ++k) best_delta_C[k] = delta_C[k];
                improved = true;
            }
        }

        if (improved) {
            my_state ^= (1ULL << best_bit);
            local_energy = best_neighbor_energy;
            for (int k = 1; k < n; ++k) C[k] += best_delta_C[k];
        }
    }

    d_final_states[tid] = my_state;
    d_final_energies[tid] = local_energy;
    if (d_eval_counts) d_eval_counts[tid] = evals;
}
#endif // HAS_CUDA_CLASSICAL

// ============================================================================
// Classical Optimizer (Host Controller + GPU Batch Dispatcher)
// ============================================================================

class ClassicalOptimizer {
private:
    LabsEvaluator evaluator;
    int n;
    uint64_t full_mask;
    uint64_t base_seed;
    mutable std::mt19937_64 rng;
    std::string device_name;

public:
    explicit ClassicalOptimizer(const LabsEvaluator& eval, uint64_t seed = 42)
        : evaluator(eval),
          n(eval.get_n()),
          full_mask((n >= 64) ? ~0ULL : ((1ULL << n) - 1ULL)),
          base_seed(seed),
          rng(seed),
          device_name("CPU Serial")
    {
#if HAS_CUDA_CLASSICAL
        int dev = 0;
        cudaDeviceProp prop;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            device_name = std::string("CUDA: ") + prop.name;
        }
#endif
    }

    explicit ClassicalOptimizer(int num_spins, uint64_t seed = 42)
        : evaluator(num_spins),
          n(num_spins),
          full_mask((num_spins >= 64) ? ~0ULL : ((1ULL << num_spins) - 1ULL)),
          base_seed(seed),
          rng(seed),
          device_name("CPU Serial")
    {
#if HAS_CUDA_CLASSICAL
        int dev = 0;
        cudaDeviceProp prop;
        if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
            device_name = std::string("CUDA: ") + prop.name;
        }
#endif
    }

    void set_seed(uint64_t seed) {
        base_seed = seed;
        rng.seed(seed);
    }

    uint64_t get_seed() const { return base_seed; }

    // ------------------------------------------------------------------------
    // State Generators
    // ------------------------------------------------------------------------

    SpinState random_state() const {
        return SpinState{rng() & full_mask};
    }

    SpinState random_kick_state(const SpinState& seed_state, int d) const {
        SpinState kicked = seed_state;
        std::array<int, 64> indices;
        for (int i = 0; i < n; ++i) indices[i] = i;

        // Partial Fisher-Yates shuffle for d distinct bit picks
        for (int i = 0; i < d && i < n; ++i) {
            std::uniform_int_distribution<int> dist(i, n - 1);
            int pick = dist(rng);
            std::swap(indices[i], indices[pick]);
            kicked.flip(indices[i]);
        }
        return kicked;
    }

    // ------------------------------------------------------------------------
    // 1. Single-Trajectory Steepest Descent (Host CPU, Zero-Allocation)
    // ------------------------------------------------------------------------

    DescentResult steepest_descent(
        const SpinState& start_state,
        size_t max_evals = std::numeric_limits<size_t>::max(),
        bool verbose = false) const
    {
        DescentResult res;
        res.start_state = start_state;
        res.start_energy = evaluator.calcEnergy(start_state);

        size_t eff_max_evals = (max_evals == 0) ? std::numeric_limits<size_t>::max() : max_evals;

        SpinState current = start_state;

        // Zero-allocation stack buffer for C_k
        std::array<int64_t, 64> C{};
        std::vector<int64_t> initial_C = evaluator.calcCk(current);
        for (int k = 1; k < n; ++k) C[k] = initial_C[k];
        res.evals_used = 1;

        int64_t current_energy = 0;
        for (int k = 1; k < n; ++k) current_energy += C[k] * C[k];

        std::array<int64_t, 64> best_delta_C{};
        std::array<int64_t, 64> delta_C{};

        while (true) {
            if (res.evals_used + n > eff_max_evals) break;

            bool improved = false;
            int64_t best_neighbor_energy = current_energy;
            int best_bit = -1;

            // Probe all N single-bit flips incrementally in registers / L1 cache
            for (int p = 0; p < n; ++p) {
                int s_p = current.get_spin(p);
                int64_t neighbor_e = 0;

                for (int k = 1; k < n; ++k) {
                    int64_t delta_k = 0;
                    if (p + k < n) {
                        delta_k += -2 * (s_p * current.get_spin(p + k));
                    }
                    if (p - k >= 0) {
                        delta_k += -2 * (current.get_spin(p - k) * s_p);
                    }
                    delta_C[k] = delta_k;
                    int64_t new_ck = C[k] + delta_k;
                    neighbor_e += new_ck * new_ck;
                }

                if (neighbor_e < best_neighbor_energy) {
                    best_neighbor_energy = neighbor_e;
                    best_bit = p;
                    best_delta_C = delta_C;
                    improved = true;
                }
            }

            res.evals_used += n;

            if (improved) {
                current.flip(best_bit);
                current_energy = best_neighbor_energy;
                for (int k = 1; k < n; ++k) C[k] += best_delta_C[k];
                res.steps_taken++;
            } else {
                res.reached_local_min = true;
                break;
            }
        }

        res.final_state = current;
        res.final_energy = current_energy;
        res.success = (res.final_energy < res.start_energy);
        res.success_rate = res.success ? 1.0 : 0.0;
        res.per_eval_rate = (res.evals_used > 0 && res.success) ? (1.0 / static_cast<double>(res.evals_used)) : 0.0;

        if (verbose) {
            std::cout << "[Steepest Descent (Fixed Budget)] Successes / Trials: "
                      << (res.success ? 1 : 0) << " / 1 ("
                      << (res.success ? "100.00%" : "0.00%") << ")"
                      << " | E_final = " << res.final_energy << " vs E_og = " << res.start_energy
                      << " | Evals: " << res.evals_used
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << res.per_eval_rate
                      << std::defaultfloat << "\n";
        }

        return res;
    }

    // ------------------------------------------------------------------------
    // 2. GPU Batch Multi-Start Execution
    // ------------------------------------------------------------------------

    BatchDescentResult run_batch_gpu(
        const std::vector<SpinState>& start_states,
        size_t max_evals_per_start = std::numeric_limits<size_t>::max()) const
    {
#if HAS_CUDA_CLASSICAL
        int num_starts = static_cast<int>(start_states.size());
        if (num_starts == 0) return {};

        std::vector<uint64_t> h_starts(num_starts);
        for (int i = 0; i < num_starts; ++i) h_starts[i] = start_states[i].bits;

        uint64_t *d_starts = nullptr, *d_finals = nullptr;
        int64_t *d_energies = nullptr, *d_evals = nullptr;

        CHECK_CUDA(cudaMalloc(&d_starts, num_starts * sizeof(uint64_t)));
        CHECK_CUDA(cudaMalloc(&d_finals, num_starts * sizeof(uint64_t)));
        CHECK_CUDA(cudaMalloc(&d_energies, num_starts * sizeof(int64_t)));
        CHECK_CUDA(cudaMalloc(&d_evals, num_starts * sizeof(int64_t)));

        CHECK_CUDA(cudaMemcpy(d_starts, h_starts.data(), num_starts * sizeof(uint64_t), cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = (num_starts + block_size - 1) / block_size;

        batch_steepest_descent_kernel<<<grid_size, block_size>>>(
            n, num_starts, d_starts, d_finals, d_energies, d_evals, max_evals_per_start);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        BatchDescentResult res;
        res.final_states.resize(num_starts);
        res.final_energies.resize(num_starts);
        res.eval_counts.resize(num_starts);

        std::vector<uint64_t> h_finals(num_starts);
        CHECK_CUDA(cudaMemcpy(h_finals.data(), d_finals, num_starts * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(res.final_energies.data(), d_energies, num_starts * sizeof(int64_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(res.eval_counts.data(), d_evals, num_starts * sizeof(int64_t), cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_starts; ++i) res.final_states[i] = SpinState{h_finals[i]};

        cudaFree(d_starts);
        cudaFree(d_finals);
        cudaFree(d_energies);
        cudaFree(d_evals);

        return res;
#else
        (void)start_states; (void)max_evals_per_start;
        throw std::runtime_error("GPU batch steepest descent requires CUDA compiler and runtime.");
#endif
    }

    // ------------------------------------------------------------------------
    // 2b. Multi-Start Steepest Descent (with success tracking vs. og_state)
    // ------------------------------------------------------------------------
    MultiStartResult multi_start_descent(
        const std::vector<SpinState>& start_states,
        const SpinState& og_state,
        bool use_gpu = true,
        size_t max_evals_per_start = std::numeric_limits<size_t>::max(),
        bool verbose = true) const
    {
        MultiStartResult res;
        res.og_state = og_state;
        res.og_energy = evaluator.calcEnergy(og_state);
        res.total_starts = start_states.size();
        size_t num_starts = start_states.size();

        if (num_starts == 0) return res;

        res.final_states.resize(num_starts);
        res.final_energies.resize(num_starts);
        res.eval_counts.resize(num_starts);

#if HAS_CUDA_CLASSICAL
        if (use_gpu) {
            BatchDescentResult gpu_res = run_batch_gpu(start_states, max_evals_per_start);
            res.final_states = std::move(gpu_res.final_states);
            res.final_energies = std::move(gpu_res.final_energies);
            res.eval_counts = std::move(gpu_res.eval_counts);
        } else
#endif
        {
            (void)use_gpu;
            for (size_t i = 0; i < num_starts; ++i) {
                DescentResult d = steepest_descent(start_states[i], max_evals_per_start, false);
                res.final_states[i] = d.final_state;
                res.final_energies[i] = d.final_energy;
                res.eval_counts[i] = static_cast<int64_t>(d.evals_used);
            }
        }

        for (size_t i = 0; i < num_starts; ++i) {
            res.total_evals += res.eval_counts[i];
            if (res.final_energies[i] < res.best_energy) {
                res.best_energy = res.final_energies[i];
                res.best_state = res.final_states[i];
            }
            if (res.final_energies[i] < res.og_energy) {
                res.improved_hits++;
            }
        }

        if (res.total_starts > 0) {
            res.success_rate = static_cast<double>(res.improved_hits) / static_cast<double>(res.total_starts);
        }
        if (res.total_evals > 0) {
            res.per_eval_rate = static_cast<double>(res.improved_hits) / static_cast<double>(res.total_evals);
        }

        if (verbose) {
            std::cout << "[Multi-Start Steepest Descent] Successes / Trials: "
                      << res.improved_hits << " / " << res.total_starts
                      << " (" << std::fixed << std::setprecision(2) << (res.success_rate * 100.0) << "%)"
                      << " | E < E_og (E_og = " << res.og_energy << ")"
                      << " | Total Evals: " << res.total_evals
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << res.per_eval_rate
                      << std::defaultfloat << "\n";
        }

        return res;
    }

    MultiStartResult multi_start_descent(
        const SpinState& og_state,
        size_t num_starts,
        size_t max_evals_per_start = std::numeric_limits<size_t>::max(),
        bool use_gpu = true,
        bool verbose = true) const
    {
        std::vector<SpinState> start_states(num_starts);
        for (size_t i = 0; i < num_starts; ++i) {
            start_states[i] = random_state();
        }
        return multi_start_descent(start_states, og_state, use_gpu, max_evals_per_start, verbose);
    }

    // ------------------------------------------------------------------------
    // 3. Collect X Distinct Symmetry Families (GPU Accelerated)
    // ------------------------------------------------------------------------

    std::vector<MinimumFamily> collect_distinct_minima_gpu(
        size_t target_families,
        size_t batch_size = 4096,
        size_t max_starts = 100000,
        LogLevel log_level = LogLevel::Summary) const
    {
#if HAS_CUDA_CLASSICAL
        auto t_start = std::chrono::high_resolution_clock::now();
        std::unordered_map<SpinState, MinimumFamily, SpinStateHash> families;
        families.reserve(target_families * 2);

        size_t total_evals = 0;
        size_t total_starts = 0;
        size_t batch_idx = 0;
        size_t stagnant_batches = 0;

        while (families.size() < target_families && total_starts < max_starts) {
            batch_idx++;
            std::vector<SpinState> batch_starts(batch_size);
            for (size_t i = 0; i < batch_size; ++i) batch_starts[i] = random_state();

            BatchDescentResult batch_res = run_batch_gpu(batch_starts);
            total_starts += batch_size;

            size_t new_discoveries = 0;
            for (size_t i = 0; i < batch_size; ++i) {
                total_evals += batch_res.eval_counts[i];
                SpinState canon = batch_res.final_states[i].canonical(n);

                auto it = families.find(canon);
                if (it == families.end()) {
                    MinimumFamily fam;
                    fam.id = families.size() + 1;
                    fam.canonical_state = canon;
                    fam.sample_state = batch_res.final_states[i];
                    fam.energy = batch_res.final_energies[i];
                    fam.times_visited = 1;
                    fam.evals_to_first_find = total_evals;
                    families.emplace(canon, fam);
                    new_discoveries++;

                    if (families.size() >= target_families) break;
                } else {
                    it->second.times_visited++;
                }
            }

            if (new_discoveries == 0) {
                stagnant_batches++;
                if (stagnant_batches >= 4) {
                    if (log_level != LogLevel::Silent) {
                        std::cout << "[GPU Minima Collector] Pool saturated: collected all reachable minima (" 
                                  << families.size() << " families after " << total_starts << " starts).\n";
                    }
                    break;
                }
            } else {
                stagnant_batches = 0;
            }

            if (log_level == LogLevel::Verbose) {
                std::cout << "[GPU Batch #" << batch_idx << "] Found " << new_discoveries 
                          << " new families (Total: " << families.size() << "/" << target_families << ")\n";
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::vector<MinimumFamily> result;
        result.reserve(families.size());
        for (auto& kv : families) result.push_back(std::move(kv.second));

        // Sort by energy (lowest energy / deepest traps first)
        std::sort(result.begin(), result.end(), [](const MinimumFamily& a, const MinimumFamily& b) {
            if (a.energy != b.energy) return a.energy < b.energy;
            return a.times_visited > b.times_visited;
        });

        for (size_t i = 0; i < result.size(); ++i) result[i].id = i + 1;

        if (log_level != LogLevel::Silent) {
            ClassicalRunMetadata meta;
            meta.n = n;
            meta.seed = base_seed;
            meta.algorithm_name = "GPU Multi-Start Minima Collector";
            meta.total_starts = total_starts;
            meta.total_evals = total_evals;
            meta.elapsed_ms = elapsed_ms;
            meta.device_name = device_name;
            ClassicalLogger::print_minima_summary(result, meta);
        }

        return result;
#else
        (void)target_families; (void)batch_size; (void)max_starts; (void)log_level;
        throw std::runtime_error("GPU minima collector requires CUDA compiler and runtime.");
#endif
    }

    // ------------------------------------------------------------------------
    // 3b. Collect X Distinct Symmetry Families (Host CPU Serial)
    // ------------------------------------------------------------------------

    std::vector<MinimumFamily> collect_distinct_minima_cpu(
        size_t target_families,
        size_t max_starts = 100000,
        LogLevel log_level = LogLevel::Summary) const
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        std::unordered_map<SpinState, MinimumFamily, SpinStateHash> families;
        families.reserve(target_families * 2);

        size_t total_evals = 0;
        size_t total_starts = 0;
        size_t stagnant_starts = 0;

        for (size_t s = 0; s < max_starts && families.size() < target_families; ++s) {
            SpinState start = random_state();
            DescentResult d = steepest_descent(start);
            total_evals += d.evals_used;
            total_starts++;

            SpinState canon = d.final_state.canonical(n);

            auto it = families.find(canon);
            if (it == families.end()) {
                MinimumFamily fam;
                fam.id = families.size() + 1;
                fam.canonical_state = canon;
                fam.sample_state = d.final_state;
                fam.energy = d.final_energy;
                fam.times_visited = 1;
                fam.evals_to_first_find = total_evals;
                families.emplace(canon, fam);
                stagnant_starts = 0;
            } else {
                it->second.times_visited++;
                stagnant_starts++;
                if (stagnant_starts >= 5000 && families.size() >= 10) {
                    if (log_level != LogLevel::Silent) {
                        std::cout << "[CPU Minima Collector] Pool saturated: collected all reachable minima (" 
                                  << families.size() << " families after " << total_starts << " starts).\n";
                    }
                    break;
                }
            }
        }

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

        std::vector<MinimumFamily> result;
        result.reserve(families.size());
        for (auto& kv : families) result.push_back(std::move(kv.second));

        std::sort(result.begin(), result.end(), [](const MinimumFamily& a, const MinimumFamily& b) {
            if (a.energy != b.energy) return a.energy < b.energy;
            return a.times_visited > b.times_visited;
        });

        for (size_t i = 0; i < result.size(); ++i) result[i].id = i + 1;

        if (log_level != LogLevel::Silent) {
            ClassicalRunMetadata meta;
            meta.n = n;
            meta.seed = base_seed;
            meta.algorithm_name = "CPU Serial Minima Collector";
            meta.total_starts = total_starts;
            meta.total_evals = total_evals;
            meta.elapsed_ms = elapsed_ms;
            meta.device_name = "CPU Serial";
            ClassicalLogger::print_minima_summary(result, meta);
        }

        return result;
    }

    // ------------------------------------------------------------------------
    // 4. Budget-Enforced Random Restarts
    // ------------------------------------------------------------------------

    BudgetedResult random_restarts(
        const SpinState& og_state,
        size_t evaluation_budget,
        bool verbose = true) const
    {
        return random_restarts(evaluation_budget, evaluator.calcEnergy(og_state), verbose, og_state);
    }

    BudgetedResult random_restarts(
        size_t evaluation_budget,
        const SpinState& og_state,
        bool verbose = true) const
    {
        return random_restarts(evaluation_budget, evaluator.calcEnergy(og_state), verbose, og_state);
    }

    BudgetedResult random_restarts(
        size_t evaluation_budget,
        int64_t target_energy = std::numeric_limits<int64_t>::max(),
        bool verbose = true,
        const SpinState& og_state = SpinState{0ULL}) const
    {
        BudgetedResult b_res;
        b_res.og_state = og_state;
        b_res.og_energy = (target_energy == std::numeric_limits<int64_t>::max()) ? 0 : target_energy;
        b_res.best_energy = std::numeric_limits<int64_t>::max();

        while (b_res.total_evals_used < evaluation_budget) {
            size_t remaining = evaluation_budget - b_res.total_evals_used;
            SpinState start = random_state();
            DescentResult d = steepest_descent(start, remaining, false);

            b_res.total_evals_used += d.evals_used;
            b_res.total_trajectories++;

            if (d.final_energy < b_res.best_energy) {
                b_res.best_energy = d.final_energy;
                b_res.best_state = d.final_state;
            }
            if (d.final_energy < target_energy) {
                b_res.improved_hits++;
            }
        }

        if (b_res.total_trajectories > 0) {
            b_res.success_rate = static_cast<double>(b_res.improved_hits) / b_res.total_trajectories;
        }
        if (b_res.total_evals_used > 0) {
            b_res.per_eval_rate = static_cast<double>(b_res.improved_hits) / b_res.total_evals_used;
        }

        if (verbose) {
            std::cout << "[Random Restarts] Successes / Trials: "
                      << b_res.improved_hits << " / " << b_res.total_trajectories
                      << " (" << std::fixed << std::setprecision(2) << (b_res.success_rate * 100.0) << "%)"
                      << " | E < E_og (E_og = " << b_res.og_energy << ")"
                      << " | Total Evals: " << b_res.total_evals_used
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << b_res.per_eval_rate
                      << std::defaultfloat << "\n";
        }

        return b_res;
    }

    // ------------------------------------------------------------------------
    // 5. Budget-Enforced Random Kick + SD (Method 3)
    // ------------------------------------------------------------------------

    BudgetedResult random_kick(
        const SpinState& seed_state,
        int d,
        size_t evaluation_budget,
        bool verbose = true) const
    {
        BudgetedResult b_res;
        b_res.og_state = seed_state;
        int64_t seed_energy = evaluator.calcEnergy(seed_state);
        b_res.og_energy = seed_energy;
        b_res.best_energy = seed_energy;
        b_res.best_state = seed_state;

        while (b_res.total_evals_used < evaluation_budget) {
            size_t remaining = evaluation_budget - b_res.total_evals_used;
            SpinState kicked = random_kick_state(seed_state, d);
            DescentResult dr = steepest_descent(kicked, remaining, false);

            b_res.total_evals_used += dr.evals_used;
            b_res.total_trajectories++;

            if (dr.final_energy < b_res.best_energy) {
                b_res.best_energy = dr.final_energy;
                b_res.best_state = dr.final_state;
            }
            if (dr.final_energy < seed_energy) {
                b_res.improved_hits++;
            }
        }

        if (b_res.total_trajectories > 0) {
            b_res.success_rate = static_cast<double>(b_res.improved_hits) / b_res.total_trajectories;
        }
        if (b_res.total_evals_used > 0) {
            b_res.per_eval_rate = static_cast<double>(b_res.improved_hits) / b_res.total_evals_used;
        }

        if (verbose) {
            std::cout << "[Random Kick (d=" << d << ")] Successes / Trials: "
                      << b_res.improved_hits << " / " << b_res.total_trajectories
                      << " (" << std::fixed << std::setprecision(2) << (b_res.success_rate * 100.0) << "%)"
                      << " | E < E_og (E_og = " << seed_energy << ")"
                      << " | Total Evals: " << b_res.total_evals_used
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << b_res.per_eval_rate
                      << std::defaultfloat << "\n";
        }

        return b_res;
    }

    // ------------------------------------------------------------------------
    // 6. Fixed-Trial Random Kick + SD (10^4 - 10^5 Trials per d)
    // ------------------------------------------------------------------------
    MultiStartResult run_kick_trials(
        const SpinState& og_state,
        int d,
        size_t total_trials,
        size_t batch_size = 10000,
        bool use_gpu = true,
        size_t max_evals_per_start = 0,
        bool verbose = true) const
    {
        MultiStartResult agg;
        agg.og_state = og_state;
        agg.og_energy = evaluator.calcEnergy(og_state);
        agg.best_energy = agg.og_energy;
        agg.best_state = og_state;
        agg.total_starts = total_trials;

        size_t trials_completed = 0;
        while (trials_completed < total_trials) {
            size_t chunk = std::min(batch_size, total_trials - trials_completed);
            std::vector<SpinState> batch_starts(chunk);
            for (size_t i = 0; i < chunk; ++i) {
                batch_starts[i] = random_kick_state(og_state, d);
            }

            MultiStartResult chunk_res = multi_start_descent(
                batch_starts, og_state, use_gpu, max_evals_per_start, false
            );

            agg.improved_hits += chunk_res.improved_hits;
            agg.total_evals += chunk_res.total_evals;
            if (chunk_res.best_energy < agg.best_energy) {
                agg.best_energy = chunk_res.best_energy;
                agg.best_state = chunk_res.best_state;
            }
            trials_completed += chunk;
        }

        if (agg.total_starts > 0) {
            agg.success_rate = static_cast<double>(agg.improved_hits) / static_cast<double>(agg.total_starts);
        }
        if (agg.total_evals > 0) {
            agg.per_eval_rate = static_cast<double>(agg.improved_hits) / static_cast<double>(agg.total_evals);
        }

        if (verbose) {
            std::cout << "[Random Kick Trials (d=" << d << ")] Successes / Trials: "
                      << agg.improved_hits << " / " << agg.total_starts
                      << " (" << std::fixed << std::setprecision(2) << (agg.success_rate * 100.0) << "%)"
                      << " | E < E_og (E_og = " << agg.og_energy << ")"
                      << " | Total Evals: " << agg.total_evals
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << agg.per_eval_rate
                      << std::defaultfloat << "\n";
        }

        return agg;
    }

    // ------------------------------------------------------------------------
    // 7. Fixed-Trial Random Restarts (10^4 - 10^5 Trials)
    // ------------------------------------------------------------------------
    MultiStartResult run_restart_trials(
        const SpinState& og_state,
        size_t total_trials,
        size_t batch_size = 10000,
        bool use_gpu = true,
        size_t max_evals_per_start = 0,
        bool verbose = true) const
    {
        MultiStartResult agg;
        agg.og_state = og_state;
        agg.og_energy = evaluator.calcEnergy(og_state);
        agg.best_energy = agg.og_energy;
        agg.best_state = og_state;
        agg.total_starts = total_trials;

        size_t trials_completed = 0;
        while (trials_completed < total_trials) {
            size_t chunk = std::min(batch_size, total_trials - trials_completed);
            std::vector<SpinState> batch_starts(chunk);
            for (size_t i = 0; i < chunk; ++i) {
                batch_starts[i] = random_state();
            }

            MultiStartResult chunk_res = multi_start_descent(
                batch_starts, og_state, use_gpu, max_evals_per_start, false
            );

            agg.improved_hits += chunk_res.improved_hits;
            agg.total_evals += chunk_res.total_evals;
            if (chunk_res.best_energy < agg.best_energy) {
                agg.best_energy = chunk_res.best_energy;
                agg.best_state = chunk_res.best_state;
            }
            trials_completed += chunk;
        }

        if (agg.total_starts > 0) {
            agg.success_rate = static_cast<double>(agg.improved_hits) / static_cast<double>(agg.total_starts);
        }
        if (agg.total_evals > 0) {
            agg.per_eval_rate = static_cast<double>(agg.improved_hits) / static_cast<double>(agg.total_evals);
        }

        if (verbose) {
            std::cout << "[Random Restarts Trials] Successes / Trials: "
                      << agg.improved_hits << " / " << agg.total_starts
                      << " (" << std::fixed << std::setprecision(2) << (agg.success_rate * 100.0) << "%)"
                      << " | E < E_og (E_og = " << agg.og_energy << ")"
                      << " | Total Evals: " << agg.total_evals
                      << " | Per-Eval Rate: " << std::scientific << std::setprecision(4) << agg.per_eval_rate
                      << std::defaultfloat << "\n";
        }

        return agg;
    }
};
