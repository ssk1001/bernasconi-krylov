#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <limits>
#include "../labs_energy/labs_energy.hpp"

#define MAX_LOG_EVENTS 1000
#define MAX_BITS_SUPPORT 512 // Maximum supported bits for local C_k register storage

struct LogEvent {
    long long timestamp_cycles;
    long long energy;
    int thread_id;
};

struct DescentResult {
    SpinState best_state;
    long long best_energy; 
    std::vector<SpinState> all_states;
    std::vector<long long> all_energies;
};

struct DeviceLogBuffer {
    int count;
    LogEvent events[MAX_LOG_EVENTS];
};

__global__ void multi_start_descent_kernel(
    size_t num_bits, 
    size_t num_words,
    int num_starts, 
    unsigned long long* d_states, 
    unsigned long long* d_best_neighbors,
    DeviceLogBuffer* d_log_buffer, 
    long long* d_global_best_energy,
    unsigned long long* d_global_best_state,
    long long* d_final_energies) 
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_starts) return;

    unsigned long long* my_state = d_states + (tid * num_words);

    // Local array to keep C_k values for this thread (O(N) space per thread)
    long long C[MAX_BITS_SUPPORT];

    // 1. Initial full O(N^2) energy and C_k computation at the start
    long long local_energy = 0;
    for (size_t k = 1; k < num_bits; ++k) {
        int differences = 0;
        for (size_t i = 0; i < num_bits - k; ++i) {
            size_t word_idx1 = i / 64;
            size_t bit_idx1 = i % 64;
            size_t word_idx2 = (i + k) / 64;
            size_t bit_idx2 = (i + k) % 64;

            unsigned long long b1 = (my_state[word_idx1] >> bit_idx1) & 1ULL;
            unsigned long long b2 = (my_state[word_idx2] >> bit_idx2) & 1ULL;
            
            if (b1 != b2) differences++;
        }
        
        C[k] = (long long)(num_bits - k) - 2LL * differences;
        local_energy += C[k] * C[k];
    }

    bool improved = true;
    while (improved) {
        improved = false;
        long long best_neighbor_energy = local_energy;
        size_t best_bit_to_flip = 0;

        // 2. Evaluate all single-bit flips in O(N^2) total per descent step 
        // (O(N) per bit flip using incremental C_k updates instead of full recomputation)
        for (size_t p = 0; p < num_bits; ++p) {
            size_t word_idx_p = p / 64;
            size_t bit_idx_p = p % 64;
            unsigned long long bp_old = (my_state[word_idx_p] >> bit_idx_p) & 1ULL;

            long long neighbor_energy = 0;

            // Compute neighbor energy in O(N) by incrementally adjusting C_k values
            for (size_t k = 1; k < num_bits; ++k) {
                long long delta_k = 0;

                // Check pair (p, p + k)
                if (p + k < num_bits) {
                    size_t w2 = (p + k) / 64;
                    size_t b2 = (p + k) % 64;
                    unsigned long long bp_k = (my_state[w2] >> b2) & 1ULL;
                    int prod = (bp_old == bp_k) ? 1 : -1;
                    delta_k += -2 * prod;
                }

                // Check pair (p - k, p)
                if (p >= k) {
                    size_t w1 = (p - k) / 64;
                    size_t b1 = (p - k) % 64;
                    unsigned long long b_pmk = (my_state[w1] >> b1) & 1ULL;
                    int prod = (b_pmk == bp_old) ? 1 : -1;
                    delta_k += -2 * prod;
                }

                long long new_Ck = C[k] + delta_k;
                neighbor_energy += new_Ck * new_Ck;
            }

            if (neighbor_energy < best_neighbor_energy) {
                best_neighbor_energy = neighbor_energy;
                improved = true;
                best_bit_to_flip = p;
            }
        }

        if (improved) {
            // Permanently apply the best bit flip and update C[k] values
            size_t word_idx = best_bit_to_flip / 64;
            size_t bit_idx = best_bit_to_flip % 64;
            unsigned long long bp_old = (my_state[word_idx] >> bit_idx) & 1ULL;

            for (size_t k = 1; k < num_bits; ++k) {
                long long delta_k = 0;
                if (best_bit_to_flip + k < num_bits) {
                    size_t w2 = (best_bit_to_flip + k) / 64;
                    size_t b2 = (best_bit_to_flip + k) % 64;
                    unsigned long long bp_k = (my_state[w2] >> b2) & 1ULL;
                    int prod = (bp_old == bp_k) ? 1 : -1;
                    delta_k += -2 * prod;
                }
                if (best_bit_to_flip >= k) {
                    size_t w1 = (best_bit_to_flip - k) / 64;
                    size_t b1 = (best_bit_to_flip - k) % 64;
                    unsigned long long b_pmk = (my_state[w1] >> b1) & 1ULL;
                    int prod = (b_pmk == bp_old) ? 1 : -1;
                    delta_k += -2 * prod;
                }
                C[k] += delta_k;
            }

            my_state[word_idx] ^= (1ULL << bit_idx);
            local_energy = best_neighbor_energy;

            long long current_global = *d_global_best_energy;

            if (local_energy < current_global) {
                long long old_global = atomicMin((unsigned long long*)d_global_best_energy, 
                                                 (unsigned long long)local_energy);
                
                if (local_energy < old_global) {
                    for (size_t w = 0; w < num_words; ++w) {
                        d_global_best_state[w] = my_state[w];
                    }

                    int log_idx = atomicAdd(&d_log_buffer->count, 1);
                    if (log_idx < MAX_LOG_EVENTS) {
                        d_log_buffer->events[log_idx].timestamp_cycles = clock64();
                        d_log_buffer->events[log_idx].energy = local_energy;
                        d_log_buffer->events[log_idx].thread_id = tid;
                    }
                }
            }
        }
    }
    
    d_final_energies[tid] = local_energy;
}

DescentResult multi_start_steepest_descent(size_t n, size_t num_starts, SpinCache& cache) {
    std::mt19937_64 rng(std::random_device{}());
    size_t num_words = (n + 63) / 64;
    
    std::vector<unsigned long long> h_start_states(num_starts * num_words, 0);
    for (size_t start = 0; start < num_starts; ++start) {
        size_t offset = start * num_words;
        for (size_t w = 0; w < num_words; ++w) {
            h_start_states[offset + w] = rng();
        }
        if (n % 64 != 0) {
            h_start_states[offset + num_words - 1] &= (1ULL << (n % 64)) - 1;
        }
    }

    unsigned long long* d_states;
    unsigned long long* d_best_neighbors;
    DeviceLogBuffer* d_log_buffer;
    long long* d_global_best_energy;
    unsigned long long* d_global_best_state;
    long long* d_final_energies;

    cudaMalloc(&d_states, num_starts * num_words * sizeof(unsigned long long));
    cudaMalloc(&d_best_neighbors, num_starts * num_words * sizeof(unsigned long long));
    cudaMalloc(&d_log_buffer, sizeof(DeviceLogBuffer));
    cudaMalloc(&d_global_best_energy, sizeof(long long));
    cudaMalloc(&d_global_best_state, num_words * sizeof(unsigned long long));
    cudaMalloc(&d_final_energies, num_starts * sizeof(long long));

    cudaMemcpy(d_states, h_start_states.data(), h_start_states.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    
    cudaMemset(d_log_buffer, 0, sizeof(DeviceLogBuffer));
    long long init_max = std::numeric_limits<long long>::max();
    cudaMemcpy(d_global_best_energy, &init_max, sizeof(long long), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_starts + threadsPerBlock - 1) / threadsPerBlock;
    
    multi_start_descent_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        n, num_words, num_starts, d_states, d_best_neighbors, 
        d_log_buffer, d_global_best_energy, d_global_best_state, d_final_energies
    );
    
    cudaDeviceSynchronize();

    long long h_final_energy;
    std::vector<uint64_t> h_final_state(num_words);
    std::vector<uint64_t> h_all_final_states(num_starts * num_words);
    std::vector<long long> h_all_final_energies(num_starts);
    
    cudaMemcpy(&h_final_energy, d_global_best_energy, sizeof(long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_state.data(), d_global_best_state, num_words * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_final_states.data(), d_states, num_starts * num_words * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_all_final_energies.data(), d_final_energies, num_starts * sizeof(long long), cudaMemcpyDeviceToHost);

    cudaFree(d_states);
    cudaFree(d_best_neighbors);
    cudaFree(d_log_buffer);
    cudaFree(d_global_best_energy);
    cudaFree(d_global_best_state);
    cudaFree(d_final_energies);

    DescentResult final_result;
    final_result.best_energy = h_final_energy;
    final_result.best_state.num_bits = n;
    final_result.best_state.bits = h_final_state;
    
    for (int i = 0; i < num_starts; ++i) {
        SpinState s;
        s.num_bits = n;
        s.bits = std::vector<uint64_t>(
            h_all_final_states.begin() + i * num_words, 
            h_all_final_states.begin() + (i + 1) * num_words
        );
        final_result.all_states.push_back(s);
        final_result.all_energies.push_back(h_all_final_energies[i]);
    }

    return final_result;
}