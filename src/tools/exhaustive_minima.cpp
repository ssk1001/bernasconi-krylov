// ============================================================================
// Exhaustive Local Minima Explorer for Bernasconi / LABS Spin Model
//
// Explores the entire state space (pruned by global spin negation symmetry s -> ~s)
// to identify all 1-flip local minima, canonicalizes them under the order-8 symmetry
// group (D4 x Z2), identifies E* and the first two excited trap tiers (Level 1 and Level 2),
// and computes exact barrier distances d_min.
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
#include <map>
#include <set>
#include <cstdint>
#include <limits>

#include "labs_energy/labs_energy.hpp"
#include "classical/classical.cuh"

namespace fs = std::filesystem;

struct OrbitInfo {
    SpinState canonical_state;
    std::string canonical_id;
    std::string bitstring;
    int64_t energy = 0;
    int d_min = 0;
    double sd_hit_fraction = 0.0;
};

struct LevelInfo {
    int64_t energy = 0;
    std::vector<OrbitInfo> orbits;
};

struct ExhaustiveResult {
    int n = 0;
    int64_t e_star = 0;
    int64_t total_minima_found = 0;
    int64_t total_orbits_found = 0;
    std::vector<LevelInfo> levels;
    double search_time_sec = 0.0;
};

void run_sd_benchmarks(ExhaustiveResult& res, size_t num_sd_starts);

// Converts canonical bits to hex string "0x..."
std::string to_hex_id(uint64_t bits) {
    std::stringstream ss;
    ss << "0x" << std::hex << std::uppercase << bits;
    return ss.str();
}

// Converts canonical bits to N-bit binary string (MSB at index 0)
std::string to_bitstring(uint64_t bits, int n) {
    std::string s(n, '0');
    for (int i = 0; i < n; ++i) {
        if ((bits >> (n - 1 - i)) & 1ULL) {
            s[i] = '1';
        }
    }
    return s;
}

// Fast check whether configuration s is a 1-flip local minimum
// Returns true iff for all p in [0, n), E(s ^ 2^p) >= E(s)
inline bool is_one_flip_local_minimum(uint64_t s, int n, const uint64_t* masks, int16_t* C) {
    // 1. Compute autocorrelation C_k
    for (int k = 1; k < n; ++k) {
        int diffs = __builtin_popcountll((s ^ (s >> k)) & masks[k]);
        C[k] = static_cast<int16_t>((n - k) - 2 * diffs);
    }

    // 2. Early-exit neighbor checks
    for (int p = 0; p < n; ++p) {
        int s_p = ((s >> p) & 1ULL) ? -1 : 1;
        int64_t delta_e = 0;

        for (int k = 1; k < n; ++k) {
            int d_k = 0;
            if (p + k < n) {
                int s_pk = ((s >> (p + k)) & 1ULL) ? -1 : 1;
                d_k -= 2 * (s_p * s_pk);
            }
            if (p - k >= 0) {
                int s_pmk = ((s >> (p - k)) & 1ULL) ? -1 : 1;
                d_k -= 2 * (s_pmk * s_p);
            }
            if (d_k != 0) {
                delta_e += static_cast<int64_t>(d_k) * (2 * C[k] + d_k);
            }
        }

        if (delta_e < 0) {
            return false; // Neighbor has lower energy -> not a local minimum
        }
    }

    return true;
}

ExhaustiveResult run_exploration(int n, int num_threads = 0, bool run_sd = false, size_t num_sd_starts = 100000) {
    ExhaustiveResult result;
    result.n = n;

    std::cout << "\n======================================================================\n";
    std::cout << " Exhaustive State Space Explorer for N = " << n << " (Single Processor)\n";
    std::cout << "======================================================================\n";

    // Precompute masks: mask_k = (1 << (n - k)) - 1
    uint64_t masks[64];
    for (int k = 1; k < n; ++k) {
        masks[k] = (k >= 64) ? 0ULL : ((1ULL << (n - k)) - 1ULL);
    }

    // Search space: [0, 2^(n-1)) via global spin negation symmetry s -> ~s
    uint64_t search_limit = 1ULL << (n - 1);
    std::cout << "Scanning " << search_limit << " configurations (exploiting s -> ~s symmetry)...\n";

    auto t_start = std::chrono::high_resolution_clock::now();

    std::map<int64_t, std::set<uint64_t>> energy_to_orbits;
    int64_t total_minima = 0;
    int16_t local_C[64];

    for (uint64_t s = 0; s < search_limit; ++s) {
        if (is_one_flip_local_minimum(s, n, masks, local_C)) {
            int64_t energy = 0;
            for (int k = 1; k < n; ++k) {
                energy += static_cast<int64_t>(local_C[k]) * local_C[k];
            }
            SpinState canon = SpinState{s}.canonical(n);
            energy_to_orbits[energy].insert(canon.bits);
            total_minima++;
        }
    }

    auto t_search_end = std::chrono::high_resolution_clock::now();
    result.search_time_sec = std::chrono::duration<double>(t_search_end - t_start).count();
    std::cout << "Search completed in " << std::fixed << std::setprecision(3) 
              << result.search_time_sec << " seconds.\n";

    result.total_minima_found = total_minima * 2; // account for ~s complement
    int64_t total_orbits = 0;
    for (const auto& kv : energy_to_orbits) {
        total_orbits += kv.second.size();
    }
    result.total_orbits_found = total_orbits;

    std::cout << "Found " << total_minima << " minima in half-space (" 
              << total_orbits << " unique canonical orbits across " 
              << energy_to_orbits.size() << " energy levels).\n";

    if (energy_to_orbits.empty()) {
        std::cerr << "Error: No local minima found!\n";
        return result;
    }

    // Determine E* (global optimum) and trap tiers
    auto it = energy_to_orbits.begin();
    result.e_star = it->first;
    int64_t expected_e_star = LabsEvaluator::known_optimal_energy(n);

    std::cout << "Global optimum E* = " << result.e_star;
    if (expected_e_star >= 0) {
        std::cout << " (known optimal = " << expected_e_star << ")";
        if (result.e_star != expected_e_star) {
            std::cout << " [WARNING: Mismatch with literature E*!]";
        } else {
            std::cout << " [VERIFIED]";
        }
    }
    std::cout << "\n";

    // Advance to Level 1 (lowest energy strictly above E*)
    ++it;
    LabsEvaluator evaluator(n);

    for (int level_idx = 1; level_idx <= 2 && it != energy_to_orbits.end(); ++level_idx, ++it) {
        LevelInfo level;
        level.energy = it->first;
        std::cout << " Level " << level_idx << ": E0 = " << level.energy 
                  << " (" << it->second.size() << " orbits)\n";

        for (uint64_t canon_bits : it->second) {
            OrbitInfo orb;
            orb.canonical_state = SpinState{canon_bits};
            orb.canonical_id = to_hex_id(canon_bits);
            orb.bitstring = to_bitstring(canon_bits, n);
            orb.energy = level.energy;

            // Compute exact d_min
            std::cout << "   Computing d_min for orbit " << orb.canonical_id << "... " << std::flush;
            orb.d_min = evaluator.compute_d_min(orb.canonical_state, level.energy);
            std::cout << "d_min = " << orb.d_min << "\n";

            level.orbits.push_back(orb);
        }
        result.levels.push_back(level);
    }

    // Optional SD hit fraction computation
    if (run_sd && !result.levels.empty()) {
        run_sd_benchmarks(result, num_sd_starts);
    }

    return result;
}

void run_sd_benchmarks(ExhaustiveResult& res, size_t num_sd_starts) {
    if (res.levels.empty()) return;
    int n = res.n;
    std::cout << "\nRunning " << num_sd_starts << " random-start steepest descents for N=" << n << "...\n";
    LabsEvaluator evaluator(n);

    // Map canonical bits to hit counter
    std::map<uint64_t, size_t> target_orbit_hits;
    for (const auto& lvl : res.levels) {
        for (const auto& orb : lvl.orbits) {
            target_orbit_hits[orb.canonical_state.bits] = 0;
        }
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    LabsEvaluator opt_eval(n);
    ClassicalOptimizer opt(opt_eval, 42);

    for (size_t s = 0; s < num_sd_starts; ++s) {
        SpinState start = opt.random_state();
        DescentResult d = opt.steepest_descent(start);
        SpinState canon = d.final_state.canonical(n);

        auto hit_it = target_orbit_hits.find(canon.bits);
        if (hit_it != target_orbit_hits.end()) {
            hit_it->second++;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();
    std::cout << "Descents completed in " << std::fixed << std::setprecision(2) << elapsed << "s.\n";

    for (size_t l_idx = 0; l_idx < res.levels.size(); ++l_idx) {
        auto& lvl = res.levels[l_idx];
        size_t level_total_hits = 0;
        std::cout << " Level " << (l_idx + 1) << " (E0=" << lvl.energy << "):\n";
        for (auto& orb : lvl.orbits) {
            size_t hits = target_orbit_hits[orb.canonical_state.bits];
            level_total_hits += hits;
            orb.sd_hit_fraction = static_cast<double>(hits) / static_cast<double>(num_sd_starts);
            std::cout << "   Orbit " << orb.canonical_id 
                      << ": " << hits << " / " << num_sd_starts 
                      << " hits (sd_hit_fraction = " << std::fixed << std::setprecision(6) 
                      << orb.sd_hit_fraction << " | " << std::setprecision(4) 
                      << (orb.sd_hit_fraction * 100.0) << "%)\n";
        }
        double lvl_frac = static_cast<double>(level_total_hits) / static_cast<double>(num_sd_starts);
        std::cout << "   --> Total Level " << (l_idx + 1) << " hit fraction: " 
                  << std::fixed << std::setprecision(6) << lvl_frac 
                  << " (" << std::setprecision(4) << (lvl_frac * 100.0) << "%)\n";
    }
}

bool load_from_json(const std::string& filepath, ExhaustiveResult& res) {
    std::ifstream in(filepath);
    if (!in.is_open()) return false;
    std::string line;
    LevelInfo cur_lvl;
    bool in_level = false;
    OrbitInfo cur_orb;
    bool in_orbit = false;

    res.levels.clear();

    while (std::getline(in, line)) {
        if (line.find("\"N\":") != std::string::npos) {
            auto pos = line.find(":");
            res.n = std::stoi(line.substr(pos + 1));
        } else if (line.find("\"E_star\":") != std::string::npos) {
            auto pos = line.find(":");
            res.e_star = std::stoll(line.substr(pos + 1));
        } else if (line.find("\"E0\":") != std::string::npos) {
            if (in_level) {
                res.levels.push_back(cur_lvl);
                cur_lvl = LevelInfo{};
            }
            in_level = true;
            auto pos = line.find(":");
            cur_lvl.energy = std::stoll(line.substr(pos + 1));
        } else if (line.find("\"canonical_id\":") != std::string::npos) {
            in_orbit = true;
            auto first = line.find("\"0x");
            if (first == std::string::npos) first = line.find("\"0X");
            auto last = line.find("\"", first + 1);
            cur_orb.canonical_id = line.substr(first + 1, last - first - 1);
            cur_orb.canonical_state = SpinState{std::stoull(cur_orb.canonical_id, nullptr, 16)};
        } else if (line.find("\"bitstring\":") != std::string::npos) {
            auto colon = line.find(":");
            auto first = line.find_first_of("01", colon);
            auto last = line.find_last_of("01");
            if (first != std::string::npos && last != std::string::npos) {
                cur_orb.bitstring = line.substr(first, last - first + 1);
            }
        } else if (line.find("\"d_min\":") != std::string::npos) {
            auto pos = line.find(":");
            cur_orb.d_min = std::stoi(line.substr(pos + 1));
        } else if (line.find("\"sd_hit_fraction\":") != std::string::npos) {
            auto pos = line.find(":");
            cur_orb.sd_hit_fraction = std::stod(line.substr(pos + 1));
            cur_orb.energy = cur_lvl.energy;
            cur_lvl.orbits.push_back(cur_orb);
            cur_orb = OrbitInfo{};
            in_orbit = false;
        }
    }
    if (in_level && !cur_lvl.orbits.empty()) {
        res.levels.push_back(cur_lvl);
    }
    return !res.levels.empty();
}

bool save_to_json(const ExhaustiveResult& res, const std::string& filepath) {
    try {
        fs::path p(filepath);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
    } catch (...) {}

    std::ofstream out(filepath);
    if (!out.is_open()) {
        std::cerr << "Failed to open output file: " << filepath << "\n";
        return false;
    }

    out << "{\n";
    out << "  \"N\": " << res.n << ",\n";
    out << "  \"E_star\": " << res.e_star << ",\n";
    out << "  \"levels\": [\n";

    for (size_t i = 0; i < res.levels.size(); ++i) {
        const auto& lvl = res.levels[i];
        out << "    {\n";
        out << "      \"E0\": " << lvl.energy << ",\n";
        out << "      \"orbits\": [\n";

        for (size_t j = 0; j < lvl.orbits.size(); ++j) {
            const auto& orb = lvl.orbits[j];
            out << "        {\n";
            out << "          \"canonical_id\": \"" << orb.canonical_id << "\",\n";
            out << "          \"bitstring\": \"" << orb.bitstring << "\",\n";
            out << "          \"d_min\": " << orb.d_min << ",\n";
            out << "          \"sd_hit_fraction\": " << std::fixed << std::setprecision(6) << orb.sd_hit_fraction << "\n";
            out << "        }" << (j + 1 < lvl.orbits.size() ? "," : "") << "\n";
        }

        out << "      ]\n";
        out << "    }" << (i + 1 < res.levels.size() ? "," : "") << "\n";
    }

    out << "  ]\n";
    out << "}\n";

    std::cout << "Successfully saved to: " << filepath << "\n";
    return true;
}

int main(int argc, char* argv[]) {
    int n = 20;
    int threads = 0;
    bool all_n = false;
    bool run_sd = false;
    bool sd_only = false;
    size_t num_sd = 100000;
    std::string out_path = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n" && i + 1 < argc) {
            n = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            out_path = argv[++i];
        } else if (arg == "--all-n") {
            all_n = true;
        } else if (arg == "--with-sd") {
            run_sd = true;
        } else if (arg == "--sd-only") {
            sd_only = true;
        } else if (arg == "--num-sd" && i + 1 < argc) {
            num_sd = std::stoull(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --n <int>        System size N (default: 20)\n"
                      << "  --all-n          Run across N in {20, 22, 24, 26, 28, 30}\n"
                      << "  --threads <int>  Number of OpenMP threads (default: auto)\n"
                      << "  --output <path>  Output JSON file path\n"
                      << "  --with-sd        Compute sd_hit_fraction during full exploration\n"
                      << "  --sd-only        Load existing exhaustive JSON and only run SD hit sampling\n"
                      << "  --num-sd <int>   Number of random-start descents (default: 100000)\n"
                      << "  -h, --help       Show this help message\n";
            return 0;
        }
    }

    std::vector<int> target_n;
    if (all_n) {
        target_n = {20, 22, 24, 26, 28, 30};
    } else {
        target_n = {n};
    }

    for (int cur_n : target_n) {
        std::string save_file = out_path;
        if (save_file.empty()) {
            save_file = "data_processed/exhaustive_minima_n" + std::to_string(cur_n) + ".json";
        }

        ExhaustiveResult res;
        if (sd_only) {
            if (!load_from_json(save_file, res)) {
                std::cerr << "Error: Could not load existing JSON for N=" << cur_n << " from " << save_file << "\n";
                continue;
            }
            run_sd_benchmarks(res, num_sd);
            save_to_json(res, save_file);
        } else {
            res = run_exploration(cur_n, threads, run_sd, num_sd);
            save_to_json(res, save_file);
        }
    }

    return 0;
}
