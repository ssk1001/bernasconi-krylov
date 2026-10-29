#pragma once

// ============================================================================
// DEPRECATED / REDUNDANT
// This file is superseded by src/classical/classical.cuh (ClassicalOptimizer & ClassicalLogger).
// Do not use in new implementations.
// ============================================================================

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <cctype>

#include "symmetry.hpp"
#ifdef __CUDACC__
#include "sd_collector.cuh"
#endif
#include "../labs_energy/labs_energy.hpp"

namespace SeedManager {

    inline std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) return "";
        size_t end = s.find_last_not_of(" \t\r\n");
        return s.substr(start, end - start + 1);
    }

    /**
     * Resolves the seeds directory path.
     * Checks current working directory (e.g. ./seeds), parent directory (e.g. ../seeds when run from build/),
     * and creates the directory if writing is needed.
     */
    inline std::string resolve_seeds_directory(const std::string& configured_dir = "seeds") {
        namespace fs = std::filesystem;
        try {
            if (fs::exists(configured_dir) && fs::is_directory(configured_dir)) {
                return configured_dir;
            }
            std::string parent_rel = "../" + configured_dir;
            if (fs::exists(parent_rel) && fs::is_directory(parent_rel)) {
                return parent_rel;
            }
        } catch (...) {
            // Fall back to configured_dir on filesystem inspection failure
        }
        return configured_dir;
    }

    /**
     * Searches for an existing seed file for system size n_spins.
     * Looks for candidate naming conventions:
     *   seeds_N<N>.csv, seeds_N<N>.txt, N<N>.csv, N<N>.txt, <N>.csv, <N>.txt (and lowercase variants).
     */
    inline std::string find_seed_file_for_n(const std::string& seeds_dir, size_t n_spins) {
        namespace fs = std::filesystem;
        std::string n_str = std::to_string(n_spins);
        std::vector<std::string> candidates = {
            seeds_dir + "/seeds_N" + n_str + ".csv",
            seeds_dir + "/seeds_N" + n_str + ".txt",
            seeds_dir + "/seeds_n" + n_str + ".csv",
            seeds_dir + "/seeds_n" + n_str + ".txt",
            seeds_dir + "/N" + n_str + ".csv",
            seeds_dir + "/N" + n_str + ".txt",
            seeds_dir + "/n" + n_str + ".csv",
            seeds_dir + "/n" + n_str + ".txt",
            seeds_dir + "/" + n_str + ".csv",
            seeds_dir + "/" + n_str + ".txt"
        };

        for (const auto& path_str : candidates) {
            try {
                fs::path p(path_str);
                if (fs::exists(p) && fs::is_regular_file(p) && fs::file_size(p) > 0) {
                    return path_str;
                }
            } catch (...) {
                continue;
            }
        }
        return "";
    }

    /**
     * Default filepath used when dumping newly discovered minima for system size n_spins.
     */
    inline std::string get_default_seed_filepath(const std::string& seeds_dir, size_t n_spins) {
        return seeds_dir + "/seeds_N" + std::to_string(n_spins) + ".csv";
    }

    /**
     * Saves a list of SDMinimumRecord to CSV.
     */
    inline bool save_seeds_to_file(const std::string& filepath, const std::vector<SDMinimumRecord>& minima) {
        namespace fs = std::filesystem;
        try {
            fs::path p(filepath);
            if (p.has_parent_path()) {
                fs::create_directories(p.parent_path());
            }
        } catch (const std::exception& e) {
            std::cerr << "[SEEDS ERROR] Could not create directory for " << filepath << ": " << e.what() << "\n";
            return false;
        }

        std::ofstream out(filepath);
        if (!out.is_open()) {
            std::cerr << "[SEEDS ERROR] Could not open file for writing: " << filepath << "\n";
            return false;
        }

        out << "id,energy,bitstring,evals_to_find,start_index\n";
        for (const auto& m : minima) {
            out << m.id << ","
                << m.energy << ","
                << SpinSymmetry::to_bitstring(m.state) << ","
                << m.evals_to_find << ","
                << m.start_index << "\n";
        }
        out.close();
        return true;
    }

    /**
     * Loads minima from a seed file.
     * Supports:
     *   - CSV format (id, energy, bitstring, evals, start_index)
     *   - CSV format (id, energy, bitstring)
     *   - Space/tab separated formats
     *   - Raw bitstrings (one per line)
     * Computes canonical state and validates energy.
     */
    inline std::vector<SDMinimumRecord> load_seeds_from_file(
        const std::string& filepath, 
        size_t n_spins, 
        SpinCache& cache) 
    {
        std::vector<SDMinimumRecord> records;
        std::ifstream in(filepath);
        if (!in.is_open()) {
            std::cerr << "[SEEDS ERROR] Unable to open seed file: " << filepath << "\n";
            return records;
        }

        std::string line;
        size_t line_no = 0;

        while (std::getline(in, line)) {
            line_no++;
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;

            // Split line by comma or whitespace
            std::vector<std::string> tokens;
            if (line.find(',') != std::string::npos) {
                std::stringstream ss(line);
                std::string item;
                while (std::getline(ss, item, ',')) {
                    item = trim(item);
                    if (!item.empty()) tokens.push_back(item);
                }
            } else {
                std::stringstream ss(line);
                std::string item;
                while (ss >> item) {
                    tokens.push_back(trim(item));
                }
            }

            if (tokens.empty()) continue;

            // Header line detection: check if any token contains non-binary letters like 'id', 'energy', 'bitstring'
            bool is_header = false;
            for (const auto& tok : tokens) {
                std::string lower_tok = tok;
                std::transform(lower_tok.begin(), lower_tok.end(), lower_tok.begin(), ::tolower);
                if (lower_tok == "id" || lower_tok == "energy" || lower_tok == "bitstring" || 
                    lower_tok == "state" || lower_tok == "evals" || lower_tok == "start_index") {
                    is_header = true;
                    break;
                }
            }
            if (is_header) continue;

            // Find the bitstring token: exactly n_spins long, containing only '0' and '1'
            int bitstring_idx = -1;
            for (int i = 0; i < static_cast<int>(tokens.size()); ++i) {
                if (tokens[i].size() == n_spins) {
                    bool all_bits = true;
                    for (char c : tokens[i]) {
                        if (c != '0' && c != '1') {
                            all_bits = false;
                            break;
                        }
                    }
                    if (all_bits) {
                        bitstring_idx = i;
                        break;
                    }
                }
            }

            if (bitstring_idx < 0) {
                // If line contains a bitstring with different length, warn the user
                std::cerr << "[SEEDS WARNING] Line " << line_no << " in " << filepath 
                          << " does not contain a valid bitstring of length N=" << n_spins << ". Skipping.\n";
                continue;
            }

            std::string bitstr = tokens[bitstring_idx];
            SpinState state = SpinSymmetry::from_bitstring(bitstr);
            SpinState canon = SpinSymmetry::get_canonical_state(state);
            long long energy = SpinLib::get_energy(cache, state);

            SDMinimumRecord rec;
            rec.id = records.size();
            rec.state = state;
            rec.canonical_state = canon;
            rec.energy = energy;
            rec.evals_to_find = 0;
            rec.start_index = records.size() + 1;

            // Extract additional fields if standard schema (id, energy, bitstring, evals, start_index)
            if (tokens.size() >= 5 && bitstring_idx == 2) {
                try {
                    rec.id = std::stoull(tokens[0]);
                    long long file_e = std::stoll(tokens[1]);
                    if (file_e != energy) {
                        std::cerr << "[SEEDS WARNING] Line " << line_no << " energy mismatch (file: " 
                                  << file_e << ", computed: " << energy << "). Using computed.\n";
                    }
                    rec.evals_to_find = std::stoll(tokens[3]);
                    rec.start_index = std::stoull(tokens[4]);
                } catch (...) {}
            } else if (tokens.size() >= 3 && bitstring_idx == 2) {
                try {
                    rec.id = std::stoull(tokens[0]);
                    long long file_e = std::stoll(tokens[1]);
                    if (file_e != energy) {
                        std::cerr << "[SEEDS WARNING] Line " << line_no << " energy mismatch (file: " 
                                  << file_e << ", computed: " << energy << "). Using computed.\n";
                    }
                } catch (...) {}
            }

            records.push_back(rec);
        }

        in.close();
        return records;
    }

    /**
     * Main entry point: checks for cached seeds, loads them if present (skipping SD),
     * otherwise runs GPU multi-start steepest descent, saves results, and returns minima.
     */
    inline std::vector<SDMinimumRecord> get_or_collect_minima(
        size_t n_spins,
        size_t pool_count,
        const std::string& configured_seeds_dir,
        bool force_regenerate,
        SpinCache& cache,
        bool& was_cached)
    {
        std::string seeds_dir = resolve_seeds_directory(configured_seeds_dir);
        std::string seed_file = force_regenerate ? "" : find_seed_file_for_n(seeds_dir, n_spins);

        if (!seed_file.empty()) {
            std::cout << "\n======================================================================\n";
            std::cout << " [SEEDS] Found Cached Seed File for N=" << n_spins << ": " << seed_file << "\n";
            std::cout << " [SEEDS] Loading seeds and SKIPPING Steepest Descent discovery...\n";
            std::cout << "======================================================================\n" << std::endl;

            auto minima = load_seeds_from_file(seed_file, n_spins, cache);
            if (!minima.empty()) {
                was_cached = true;
                std::cout << "[SEEDS] Successfully loaded " << minima.size() 
                          << " symmetry-distinct minima from " << seed_file << "\n" << std::endl;
                return minima;
            } else {
                std::cerr << "[SEEDS WARNING] Seed file " << seed_file 
                          << " contained no valid states. Falling back to Steepest Descent.\n";
            }
        }

        was_cached = false;
        if (force_regenerate) {
            std::cout << "[SEEDS] Force regeneration requested. Running Steepest Descent for N=" 
                      << n_spins << "...\n";
        } else {
            std::cout << "[SEEDS] No cached seed file found for N=" << n_spins 
                      << " in '" << seeds_dir << "'. Running Steepest Descent discovery...\n";
        }

#ifdef __CUDACC__
        auto minima = collect_distinct_sd_minima(n_spins, pool_count, cache);
#else
        std::vector<SDMinimumRecord> minima;
        std::cerr << "[SEEDS ERROR] Steepest Descent collection requires CUDA compiler.\n";
#endif

        std::string default_save_path = get_default_seed_filepath(seeds_dir, n_spins);
        if (save_seeds_to_file(default_save_path, minima)) {
            std::cout << "[SEEDS] Successfully saved " << minima.size() 
                      << " symmetry-distinct minima to '" << default_save_path 
                      << "' for future runs.\n";
        } else {
            std::cerr << "[SEEDS ERROR] Failed to save seeds to '" << default_save_path << "'.\n";
        }

        return minima;
    }

} // namespace SeedManager
