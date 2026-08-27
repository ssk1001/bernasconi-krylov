#include "lambda_lanczos/lambda_lanczos.hpp"

#include <bits/stdc++.h>
#include <omp.h>

using namespace std;
using lambda_lanczos::LambdaLanczos;


// ============================================================
// Global parameters
// ============================================================

int n;
double g;

size_t CACHE_SIZE;
size_t MAX_BUCKET_SIZE;


// ============================================================
// Bounded concurrent cache
// ============================================================

vector<deque<pair<u_int64_t, vector<int>>>> cache;
vector<unique_ptr<mutex>> locks;


// ============================================================
// Cache initialization
//
// Total maximum entries is fixed:
//
// CACHE_SIZE * MAX_BUCKET_SIZE == max_cache_entries
//
// Number of buckets is chosen based on the number of threads.
// ============================================================

void initializeCache(int num_threads, size_t max_cache_entries) {

    constexpr size_t BUCKETS_PER_THREAD = 64;

    if (num_threads <= 0) {
        num_threads = 1;
    }

    if (max_cache_entries == 0) {
        max_cache_entries = 1;
    }

    // Desired number of independently lockable buckets.
    size_t target_buckets = min(
        max_cache_entries,
        static_cast<size_t>(num_threads) * BUCKETS_PER_THREAD
    );

    // Find the largest bucket count <= target_buckets that
    // divides max_cache_entries exactly.
    CACHE_SIZE = 1;

    for (
        size_t candidate = target_buckets;
        candidate > 0;
        --candidate
    ) {
        if (max_cache_entries % candidate == 0) {
            CACHE_SIZE = candidate;
            break;
        }
    }

    MAX_BUCKET_SIZE =
        max_cache_entries / CACHE_SIZE;

    cache.clear();
    cache.resize(CACHE_SIZE);

    locks.clear();
    locks.reserve(CACHE_SIZE);

    for (size_t i = 0; i < CACHE_SIZE; i++) {
        locks.push_back(make_unique<mutex>());
    }

    cout << "================ Cache Configuration ================\n";
    cout << "Threads                : " << num_threads << '\n';
    cout << "Maximum cache entries  : "
         << max_cache_entries << '\n';
    cout << "Cache buckets          : "
         << CACHE_SIZE << '\n';
    cout << "Maximum bucket size    : "
         << MAX_BUCKET_SIZE << '\n';
    cout << "Total cache capacity   : "
         << CACHE_SIZE * MAX_BUCKET_SIZE << '\n';
    cout << "=====================================================\n\n";
}


// ============================================================
// Cache lookup
//
// Searches:
//   1. The requested mask
//   2. Any Hamming-distance-1 neighbor
//
// Uses try_lock():
// If a bucket is currently busy, treat it as a cache miss and
// continue without waiting.
// ============================================================

pair<u_int64_t, vector<int>> cacheSearch(u_int64_t mask) {

    // Empty vector means cache miss.
    pair<u_int64_t, vector<int>> ans = {0, {}};

    // --------------------------------------------------------
    // Search exact mask
    // --------------------------------------------------------

    size_t bucket =
        static_cast<size_t>(mask % CACHE_SIZE);

    {
        unique_lock<mutex> lock(
            *locks[bucket],
            try_to_lock
        );

        if (lock.owns_lock()) {
            for (const auto& entry : cache[bucket]) {
                if (entry.first == mask) {
                    return entry;
                }
            }
        }
    }

    // --------------------------------------------------------
    // Search Hamming-distance-1 neighbors
    // --------------------------------------------------------

    for (int i = 0; i < n; i++) {

        u_int64_t neighbor =
            mask ^ (1ULL << i);

        bucket =
            static_cast<size_t>(neighbor % CACHE_SIZE);

        unique_lock<mutex> lock(
            *locks[bucket],
            try_to_lock
        );

        if (!lock.owns_lock()) {
            continue;
        }

        for (const auto& entry : cache[bucket]) {
            if (entry.first == neighbor) {
                return entry;
            }
        }
    }

    return ans;
}


// ============================================================
// Cache insertion
//
// Blocking lock is intentional here.
//
// The critical section is short and insertion must be atomic
// with respect to other cache operations on this bucket.
// ============================================================

void cacheInsert(
    u_int64_t mask,
    vector<int>&& Ck
) {

    size_t bucket =
        static_cast<size_t>(mask % CACHE_SIZE);

    lock_guard<mutex> lock(*locks[bucket]);

    cache[bucket].emplace_back(
        mask,
        move(Ck)
    );

    if (cache[bucket].size() > MAX_BUCKET_SIZE) {
        cache[bucket].pop_front();
    }
}


// ============================================================
// Cache clearing
//
// Called only after a complete Lanczos run.
// ============================================================

void clearCache() {

    for (size_t i = 0; i < CACHE_SIZE; i++) {
        lock_guard<mutex> lock(*locks[i]);
        cache[i].clear();
    }
}


// ============================================================
// Energy calculation
// ============================================================

int calculateEnergy(u_int64_t mask) {

    auto cacheFind = cacheSearch(mask);

    int energy = 0;

    // ========================================================
    // Complete cache miss
    // ========================================================

    if (cacheFind.second.empty()) {

        vector<int> Ck(n - 1, 0);

        // Calculate all autocorrelations from scratch.
        for (int i = 0; i < n; i++) {

            bool spin_i =
                mask & (1ULL << i);

            for (int j = 0; j < i; j++) {

                bool spin_j =
                    mask & (1ULL << j);

                if (spin_i == spin_j) {
                    Ck[i - j - 1]++;
                } else {
                    Ck[i - j - 1]--;
                }
            }
        }

        // Calculate energy.
        for (int i = 0; i < n - 1; i++) {
            energy += Ck[i] * Ck[i];
        }

        cacheInsert(mask, move(Ck));

        return energy;
    }


    // ========================================================
    // Exact cache hit
    // ========================================================

    u_int64_t prevmask = cacheFind.first;

    if (prevmask == mask) {

        for (int i = 0; i < n - 1; i++) {
            energy +=
                cacheFind.second[i]
                * cacheFind.second[i];
        }

        return energy;
    }


    // ========================================================
    // Hamming-distance-1 cache hit
    // ========================================================

    vector<int> Ck =
        move(cacheFind.second);

    // Find the flipped spin.
    int ind = -1;

    for (int i = 0; i < n; i++) {

        if (
            (mask & (1ULL << i))
            !=
            (prevmask & (1ULL << i))
        ) {
            ind = i;
            break;
        }
    }

    assert(ind != -1);

    // Update correlations affected by the flipped spin.
    bool flipped_spin =
        mask & (1ULL << ind);

    for (int i = 0; i < n; i++) {

        if (i == ind) {
            continue;
        }

        bool spin_i =
            mask & (1ULL << i);

        int k =
            abs(i - ind) - 1;

        if (spin_i == flipped_spin) {
            Ck[k] += 2;
        } else {
            Ck[k] -= 2;
        }
    }

    // Calculate energy from updated correlations.
    for (int i = 0; i < n - 1; i++) {
        energy += Ck[i] * Ck[i];
    }

    cacheInsert(mask, move(Ck));

    return energy;
}


// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {

    // Defaults.
    constexpr size_t DEFAULT_MAX_CACHE_ENTRIES = 800000;

    int num_threads =
        omp_get_max_threads();

    size_t max_cache_entries =
        DEFAULT_MAX_CACHE_ENTRIES;


    // --------------------------------------------------------
    // Input
    //
    // Usage:
    //
    // ./lanczos n
    // ./lanczos n num_threads
    // ./lanczos n num_threads max_cache_entries
    //
    // Examples:
    //
    // ./lanczos 10
    // ./lanczos 10 4
    // ./lanczos 10 4 800000
    // --------------------------------------------------------

    if (argc < 2 || argc > 4) {

        cerr
            << "Usage: " << argv[0]
            << " <n> [num_threads] [max_cache_entries]\n";

        cerr
            << "Example: "
            << argv[0]
            << " 10 4 800000\n";

        return 1;
    }

    try {

        n = stoi(argv[1]);

        if (argc >= 3) {
            num_threads = stoi(argv[2]);
        }

        if (argc >= 4) {
            max_cache_entries =
                stoull(argv[3]);
        }

    } catch (const exception& e) {

        cerr
            << "Error parsing arguments: "
            << e.what()
            << '\n';

        return 1;
    }


    // --------------------------------------------------------
    // Validate input
    // --------------------------------------------------------

    if (n < 2 || n >= 63) {

        cerr
            << "Error: n must satisfy 2 <= n < 63\n";

        return 1;
    }

    if (num_threads <= 0) {

        cerr
            << "Error: number of threads must be positive\n";

        return 1;
    }

    if (max_cache_entries == 0) {

        cerr
            << "Error: maximum cache entries must be positive\n";

        return 1;
    }


    // --------------------------------------------------------
    // OpenMP configuration
    // --------------------------------------------------------

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);


    // --------------------------------------------------------
    // Cache configuration
    // --------------------------------------------------------

    initializeCache(
        num_threads,
        max_cache_entries
    );


    // --------------------------------------------------------
    // Hilbert-space dimension
    // --------------------------------------------------------

    const u_int64_t dimension =
        1ULL << n;


    // ========================================================
    // Hamiltonian matrix-vector multiplication
    //
    // H|v>
    //
    // Each iteration computes exactly one out[i], so the main
    // loop is safe to parallelize.
    // ========================================================

    auto mv_mul =
        [&](const vector<double>& in,
            vector<double>& out) {

            fill(
                out.begin(),
                out.end(),
                0.0
            );

            #pragma omp parallel for schedule(static)
            for (
                int64_t ii = 0;
                ii < static_cast<int64_t>(dimension);
                ii++
            ) {

                u_int64_t i =
                    static_cast<u_int64_t>(ii);

                // Diagonal Bernasconi energy term.
                double value =
                    static_cast<double>(
                        calculateEnergy(i)
                    ) * in[i];

                // Transverse-field term.
                for (int j = 0; j < n; j++) {

                    u_int64_t nxt =
                        i ^ (1ULL << j);

                    value -= g * in[nxt];
                }

                // Only this iteration writes out[i].
                out[i] = value;
            }
        };


    // --------------------------------------------------------
    // Output file
    // --------------------------------------------------------

    string filename =
        "data_raw/spectral_gap/gap_N"
        + to_string(n)
        + ".csv";

    ofstream file(filename);

    if (!file.is_open()) {

        cerr
            << "Failed to open output file: "
            << filename
            << '\n';

        return 1;
    }

    file << fixed << setprecision(16);
    file << "g,E0,E1,Delta\n";


    // ========================================================
    // Sweep g from 0.0 to 20.0 in steps of 0.1
    // ========================================================

    for (int step = 0; step <= 200; step++) {

        g =
            static_cast<double>(step) / 10.0;


        // Request the two lowest eigenvalues.
        LambdaLanczos<double> engine(
            mv_mul,
            dimension,
            false,
            2
        );


        vector<double> eigenvalues;
        vector<vector<double>> eigenvectors;


        // Run Lanczos.
        engine.run(
            eigenvalues,
            eigenvectors
        );


        sort(
            eigenvalues.begin(),
            eigenvalues.end()
        );


        if (eigenvalues.size() < 2) {

            cerr
                << "Failed to get two eigenvalues "
                << "for g = "
                << g
                << '\n';

            clearCache();

            continue;
        }


        double E0 =
            eigenvalues[0];

        double E1 =
            eigenvalues[1];

        double delta =
            E1 - E0;


        // ----------------------------------------------------
        // Write results
        // ----------------------------------------------------

        file
            << g << ","
            << E0 << ","
            << E1 << ","
            << delta
            << '\n';


        cout
            << "g = " << g
            << " | E0 = " << E0
            << " | E1 = " << E1
            << " | Delta = " << delta
            << '\n';


        // engine.run() has returned, so no thread should still
        // be accessing the cache here.
        clearCache();
    }


    cout
        << "\nResults saved to "
        << filename
        << '\n';

    return 0;
}