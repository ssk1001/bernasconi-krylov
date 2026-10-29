// ============================================================================
// DEPRECATED / REDUNDANT
// Tests legacy seed_manager.hpp. 
// Superseded by tests/test_steepest_descent.cu (CSV reconstruction tests).
// ============================================================================

#include <iostream>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <vector>
#include "../src/labs_energy/labs_energy.hpp"
#include "../src/classical/symmetry.hpp"
#include "../src/classical/seed_manager.hpp"

namespace fs = std::filesystem;

int main() {
    std::cout << "=========================================\n";
    std::cout << "         Testing SeedManager             \n";
    std::cout << "=========================================\n";

    std::string test_dir = "test_seeds_temp";
    fs::create_directories(test_dir);

    SpinCache cache;
    size_t n = 16;

    // Create 3 mock minima
    std::vector<SDMinimumRecord> mock_minima;
    for (size_t i = 0; i < 3; ++i) {
        SpinState s;
        s.num_bits = n;
        s.bits = {0x1234ULL + i * 0x111ULL};
        long long energy = SpinLib::get_energy(cache, s);

        SDMinimumRecord rec;
        rec.id = i;
        rec.state = s;
        rec.canonical_state = SpinSymmetry::get_canonical_state(s);
        rec.energy = energy;
        rec.evals_to_find = 100 + i * 50;
        rec.start_index = 10 + i * 5;
        mock_minima.push_back(rec);
    }

    // 1. Test saving seeds to file
    std::string save_path = test_dir + "/seeds_N16.csv";
    bool saved = SeedManager::save_seeds_to_file(save_path, mock_minima);
    assert(saved && "save_seeds_to_file must succeed");
    assert(fs::exists(save_path) && "Saved file must exist on disk");
    std::cout << "  --> PASSED: save_seeds_to_file\n";

    // 2. Test find_seed_file_for_n
    std::string found_file = SeedManager::find_seed_file_for_n(test_dir, 16);
    assert(!found_file.empty() && "find_seed_file_for_n must locate seeds_N16.csv");
    assert(found_file == save_path);
    std::cout << "  --> PASSED: find_seed_file_for_n (seeds_N16.csv)\n";

    // 3. Test loading saved seeds
    auto loaded = SeedManager::load_seeds_from_file(save_path, 16, cache);
    assert(loaded.size() == 3 && "Must load all 3 records");
    for (size_t i = 0; i < 3; ++i) {
        assert(loaded[i].id == mock_minima[i].id);
        assert(loaded[i].energy == mock_minima[i].energy);
        assert(loaded[i].state == mock_minima[i].state);
        assert(loaded[i].canonical_state == mock_minima[i].canonical_state);
        assert(loaded[i].evals_to_find == mock_minima[i].evals_to_find);
        assert(loaded[i].start_index == mock_minima[i].start_index);
    }
    std::cout << "  --> PASSED: load_seeds_from_file (standard 5-column CSV)\n";

    // 4. Test loading alternative formats
    // Format A: raw bitstrings only
    {
        std::string raw_path = test_dir + "/raw_seeds.txt";
        std::ofstream raw_out(raw_path);
        raw_out << "# Comment line to skip\n";
        for (const auto& m : mock_minima) {
            raw_out << SpinSymmetry::to_bitstring(m.state) << "\n";
        }
        raw_out.close();

        auto loaded_raw = SeedManager::load_seeds_from_file(raw_path, 16, cache);
        assert(loaded_raw.size() == 3);
        for (size_t i = 0; i < 3; ++i) {
            assert(loaded_raw[i].state == mock_minima[i].state);
            assert(loaded_raw[i].energy == mock_minima[i].energy);
            assert(loaded_raw[i].canonical_state == mock_minima[i].canonical_state);
        }
        std::cout << "  --> PASSED: load_seeds_from_file (raw bitstrings with comment)\n";
    }

    // Format B: 3 columns id,energy,bitstring with header
    {
        std::string three_col_path = test_dir + "/N16.csv";
        std::ofstream col3_out(three_col_path);
        col3_out << "id,energy,bitstring\n";
        for (const auto& m : mock_minima) {
            col3_out << m.id << "," << m.energy << "," << SpinSymmetry::to_bitstring(m.state) << "\n";
        }
        col3_out.close();

        auto loaded_3col = SeedManager::load_seeds_from_file(three_col_path, 16, cache);
        assert(loaded_3col.size() == 3);
        for (size_t i = 0; i < 3; ++i) {
            assert(loaded_3col[i].state == mock_minima[i].state);
            assert(loaded_3col[i].energy == mock_minima[i].energy);
        }
        std::cout << "  --> PASSED: load_seeds_from_file (3-column CSV with header)\n";
    }

    // 5. Test candidate resolution: N16.csv candidate
    fs::remove(save_path); // Remove seeds_N16.csv so N16.csv is tested
    std::string found_n16 = SeedManager::find_seed_file_for_n(test_dir, 16);
    assert(found_n16 == test_dir + "/N16.csv" && "find_seed_file_for_n must fall back to N16.csv");
    std::cout << "  --> PASSED: candidate resolution fallback to N16.csv\n";

    // Clean up temporary test files
    fs::remove_all(test_dir);

    std::cout << "\nALL SEED MANAGER TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
