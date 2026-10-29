#pragma once

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cmath>
#include <limits>
#include <random>
#include <chrono>

#include <Eigen/Dense>
#include "../labs_energy/labs_energy.hpp"

#if defined(__CUDACC__) && __has_include(<cuda_runtime.h>) && __has_include(<cublas_v2.h>) && __has_include(<cusolverDn.h>)
    #define HAS_CUDA_LANCZOS 1
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cusolverDn.h>
#else
    #define HAS_CUDA_LANCZOS 0
#endif

#if HAS_CUDA_LANCZOS

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
    } \
}
#endif

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS error ") + std::to_string(status) + " in " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
}
#endif

#ifndef CHECK_CUSOLVER
#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuSOLVER error ") + std::to_string(status) + " in " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
}
#endif

// ============================================================================
// CUDA Kernels for Matrix-Free Matrix-Vector Multiplication
// ============================================================================

static __global__ void fullspace_matvec_kernel_d(
    int64_t dim,
    int num_bits,
    double g,
    const double* __restrict__ d_diag,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        double yi = d_diag[i] * x[i];
        for (int j = 0; j < num_bits; ++j) {
            int64_t neighbor = i ^ (1LL << j);
            yi -= g * x[neighbor];
        }
        y[i] = yi;
    }
}

static __global__ void subspace_matvec_kernel_d(
    int dim,
    int num_bits,
    double g,
    const uint64_t* __restrict__ d_basis_bits,
    const double* __restrict__ d_diag,
    const double* __restrict__ x,
    double* __restrict__ y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        double yi = d_diag[i] * x[i];
        uint64_t my_bits = d_basis_bits[i];

        for (int j = 0; j < num_bits; ++j) {
            uint64_t target_bits = my_bits ^ (1ULL << j);

            int left = 0;
            int right = dim - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                uint64_t mid_bits = d_basis_bits[mid];

                if (mid_bits == target_bits) {
                    yi -= g * x[mid];
                    break;
                } else if (mid_bits < target_bits) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        y[i] = yi;
    }
}

#endif // HAS_CUDA_LANCZOS

// ============================================================================
// Configuration & Solution Types
// ============================================================================

enum class DiagMethod {
    Lanczos,    // Matrix-free iterative Krylov solver (GPU-accelerated)
    ExactDense  // Explicit matrix construction + dense eigensolver (Eigen)
};

struct DiagConfig {
    DiagMethod method = DiagMethod::Lanczos;
    int num_eigenpairs = 1;     // Number of lowest eigenpairs to retrieve
    int max_krylov_dim = 100;   // Maximum Krylov subspace dimension m
    double tolerance = 1e-12;   // Breakdown threshold for beta
    bool normalize_by_n = true; // Use E / N on diagonal (standard in spin glasses)
};

struct Eigenpair {
    double energy = 0.0;
    std::vector<double> state_vector;
};

struct Eigensolution {
    std::vector<Eigenpair> pairs;
    double ground_energy() const { return pairs.empty() ? 0.0 : pairs[0].energy; }
    const std::vector<double>& ground_state() const { return pairs[0].state_vector; }
};

// ============================================================================
// Unified Diagonalizer Class
// ============================================================================

class Diagonalizer {
public:
    Diagonalizer() = default;

    // ------------------------------------------------------------------------
    // 1. Unified Subspace Interface (std::vector<SpinState>)
    // ------------------------------------------------------------------------
    Eigensolution solve(
        const std::vector<SpinState>& basis,
        double g,
        int n,
        DiagConfig config = {}) const
    {
        if (config.method == DiagMethod::Lanczos) {
            return solve_subspace_lanczos(basis, g, n, config);
        } else {
            return solve_subspace_dense(basis, g, n, config);
        }
    }

    // ------------------------------------------------------------------------
    // 2. Unified Full Space Interface (2^n)
    // ------------------------------------------------------------------------
    Eigensolution solve(
        int n,
        double g,
        DiagConfig config = {}) const
    {
        if (config.method == DiagMethod::Lanczos) {
            return solve_fullspace_lanczos(n, g, config);
        } else {
            return solve_fullspace_dense(n, g, config);
        }
    }

    // ------------------------------------------------------------------------
    // Method 1: Subspace ED (Dense assembly + Eigen SelfAdjointEigenSolver)
    // ------------------------------------------------------------------------
    Eigensolution solve_subspace_dense(
        const std::vector<SpinState>& basis,
        double g,
        int n,
        const DiagConfig& config = {}) const
    {
        int dim = static_cast<int>(basis.size());
        if (dim == 0) return {};

        LabsEvaluator eval(n);
        double scale = config.normalize_by_n ? (1.0 / static_cast<double>(n)) : 1.0;

        std::vector<std::pair<SpinState, int>> sorted_basis(dim);
        for (int i = 0; i < dim; ++i) {
            sorted_basis[i] = {basis[i], i};
        }
        std::sort(sorted_basis.begin(), sorted_basis.end());

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

        for (int a = 0; a < dim; ++a) {
            H(a, a) = static_cast<double>(eval.calcEnergy(basis[a])) * scale;

            for (int bit = 0; bit < n; ++bit) {
                SpinState flipped = basis[a].flipped(bit);
                auto it = std::lower_bound(
                    sorted_basis.begin(), sorted_basis.end(),
                    std::make_pair(flipped, 0),
                    [](const std::pair<SpinState, int>& x, const std::pair<SpinState, int>& val) {
                        return x.first < val.first;
                    }
                );

                if (it != sorted_basis.end() && it->first == flipped) {
                    int b = it->second;
                    H(a, b) = -g;
                }
            }
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen::SelfAdjointEigenSolver failed to converge.");
        }

        int k_pairs = std::min(config.num_eigenpairs, dim);
        Eigensolution sol;
        sol.pairs.resize(k_pairs);

        for (int k = 0; k < k_pairs; ++k) {
            sol.pairs[k].energy = solver.eigenvalues()(k);
            sol.pairs[k].state_vector.resize(dim);
            for (int i = 0; i < dim; ++i) {
                sol.pairs[k].state_vector[i] = solver.eigenvectors()(i, k);
            }
        }
        return sol;
    }

    // ------------------------------------------------------------------------
    // Method 2: Full Space ED (Dense assembly + Eigen SelfAdjointEigenSolver)
    // ------------------------------------------------------------------------
    Eigensolution solve_fullspace_dense(
        int n,
        double g,
        const DiagConfig& config = {}) const
    {
        if (n > 14) {
            throw std::runtime_error("Full space dense ED exceeds feasible memory for N > 14. Use Lanczos.");
        }

        int64_t dim = 1LL << n;
        LabsEvaluator eval(n);
        double scale = config.normalize_by_n ? (1.0 / static_cast<double>(n)) : 1.0;

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);

        for (int64_t a = 0; a < dim; ++a) {
            H(a, a) = static_cast<double>(eval.calcEnergy(SpinState{static_cast<uint64_t>(a)})) * scale;

            for (int bit = 0; bit < n; ++bit) {
                int64_t b = a ^ (1LL << bit);
                if (b > a) {
                    H(a, b) = -g;
                    H(b, a) = -g;
                }
            }
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigen::SelfAdjointEigenSolver failed on full space.");
        }

        int k_pairs = std::min(static_cast<int64_t>(config.num_eigenpairs), dim);
        Eigensolution sol;
        sol.pairs.resize(k_pairs);

        for (int k = 0; k < k_pairs; ++k) {
            sol.pairs[k].energy = solver.eigenvalues()(k);
            sol.pairs[k].state_vector.resize(dim);
            for (int64_t i = 0; i < dim; ++i) {
                sol.pairs[k].state_vector[i] = solver.eigenvectors()(i, k);
            }
        }
        return sol;
    }

    // ------------------------------------------------------------------------
    // Method 3: Subspace Lanczos (GPU Matrix-Free Krylov)
    // ------------------------------------------------------------------------
    Eigensolution solve_subspace_lanczos(
        const std::vector<SpinState>& basis,
        double g,
        int n,
        const DiagConfig& config = {}) const
    {
#if HAS_CUDA_LANCZOS
        int dim = static_cast<int>(basis.size());
        if (dim == 0) return {};

        LabsEvaluator eval(n);
        double scale = config.normalize_by_n ? (1.0 / static_cast<double>(n)) : 1.0;

        struct BasisSortItem {
            uint64_t bits;
            double energy;
            int orig_index;
            bool operator<(const BasisSortItem& o) const { return bits < o.bits; }
        };

        std::vector<BasisSortItem> sorted(dim);
        for (int i = 0; i < dim; ++i) {
            sorted[i].bits = basis[i].bits;
            sorted[i].energy = static_cast<double>(eval.calcEnergy(basis[i])) * scale;
            sorted[i].orig_index = i;
        }
        std::sort(sorted.begin(), sorted.end());

        std::vector<uint64_t> h_basis_bits(dim);
        std::vector<double> h_diag(dim);
        std::vector<int> h_orig_idx(dim);
        for (int i = 0; i < dim; ++i) {
            h_basis_bits[i] = sorted[i].bits;
            h_diag[i] = sorted[i].energy;
            h_orig_idx[i] = sorted[i].orig_index;
        }

        uint64_t* d_basis_bits = nullptr;
        double* d_diag = nullptr;
        CHECK_CUDA(cudaMalloc(&d_basis_bits, dim * sizeof(uint64_t)));
        CHECK_CUDA(cudaMalloc(&d_diag, dim * sizeof(double)));

        CHECK_CUDA(cudaMemcpy(d_basis_bits, h_basis_bits.data(), dim * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_diag, h_diag.data(), dim * sizeof(double), cudaMemcpyHostToDevice));

        auto matvec = [&](const double* d_x, double* d_y) {
            int threads = 256;
            int blocks = (dim + threads - 1) / threads;
            subspace_matvec_kernel_d<<<blocks, threads>>>(dim, n, g, d_basis_bits, d_diag, d_x, d_y);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        };

        Eigensolution raw_sol = run_gpu_lanczos_internal(dim, matvec, config);

        CHECK_CUDA(cudaFree(d_basis_bits));
        CHECK_CUDA(cudaFree(d_diag));

        Eigensolution final_sol;
        final_sol.pairs.resize(raw_sol.pairs.size());

        for (size_t k = 0; k < raw_sol.pairs.size(); ++k) {
            final_sol.pairs[k].energy = raw_sol.pairs[k].energy;
            final_sol.pairs[k].state_vector.resize(dim);
            for (int i = 0; i < dim; ++i) {
                final_sol.pairs[k].state_vector[h_orig_idx[i]] = raw_sol.pairs[k].state_vector[i];
            }
        }
        return final_sol;
#else
        (void)basis; (void)g; (void)n; (void)config;
        throw std::runtime_error("GPU Lanczos is unavailable: cuBLAS/cuSOLVER or CUDA compiler not detected.");
#endif
    }

    // ------------------------------------------------------------------------
    // Method 4: Full Space Lanczos (GPU Matrix-Free Krylov)
    // ------------------------------------------------------------------------
    Eigensolution solve_fullspace_lanczos(
        int n,
        double g,
        const DiagConfig& config = {}) const
    {
#if HAS_CUDA_LANCZOS
        if (n > 30) {
            throw std::runtime_error("Full space dimension 2^N exceeds memory limit for N > 30.");
        }

        int64_t dim = 1LL << n;
        LabsEvaluator eval(n);
        double scale = config.normalize_by_n ? (1.0 / static_cast<double>(n)) : 1.0;

        std::vector<double> h_diag(dim);
        for (int64_t i = 0; i < dim; ++i) {
            h_diag[i] = static_cast<double>(eval.calcEnergy(SpinState{static_cast<uint64_t>(i)})) * scale;
        }

        double* d_diag = nullptr;
        CHECK_CUDA(cudaMalloc(&d_diag, dim * sizeof(double)));
        CHECK_CUDA(cudaMemcpy(d_diag, h_diag.data(), dim * sizeof(double), cudaMemcpyHostToDevice));

        auto matvec = [&](const double* d_x, double* d_y) {
            int threads = 256;
            int64_t blocks = (dim + threads - 1) / threads;
            fullspace_matvec_kernel_d<<<blocks, threads>>>(dim, n, g, d_diag, d_x, d_y);
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
        };

        Eigensolution sol = run_gpu_lanczos_internal(static_cast<int>(dim), matvec, config);

        CHECK_CUDA(cudaFree(d_diag));
        return sol;
#else
        (void)n; (void)g; (void)config;
        throw std::runtime_error("GPU Lanczos is unavailable: cuBLAS/cuSOLVER or CUDA compiler not detected.");
#endif
    }

#if HAS_CUDA_LANCZOS
private:
    template <typename MatVecFunc>
    static Eigensolution run_gpu_lanczos_internal(
        int N,
        MatVecFunc matvec,
        const DiagConfig& config)
    {
        cublasHandle_t cublasH = nullptr;
        CHECK_CUBLAS(cublasCreate(&cublasH));

        cusolverDnHandle_t cusolverH = nullptr;
        CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

        int m = std::min(N, config.max_krylov_dim);

        double *d_V = nullptr, *d_w = nullptr;
        CHECK_CUDA(cudaMalloc(&d_V, static_cast<size_t>(N) * m * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_w, static_cast<size_t>(N) * sizeof(double)));

        std::vector<double> h_alpha(m, 0.0);
        std::vector<double> h_beta(m, 0.0);

        std::mt19937_64 rng(42);
        std::uniform_real_distribution<double> dist(-0.5, 0.5);
        std::vector<double> h_v0(N);
        for (int i = 0; i < N; ++i) h_v0[i] = dist(rng);

        CHECK_CUDA(cudaMemcpy(d_V, h_v0.data(), N * sizeof(double), cudaMemcpyHostToDevice));

        double v0_norm = 0.0;
        CHECK_CUBLAS(cublasDnrm2(cublasH, N, d_V, 1, &v0_norm));
        double inv_norm = 1.0 / v0_norm;
        CHECK_CUBLAS(cublasDscal(cublasH, N, &inv_norm, d_V, 1));

        int m_actual = m;

        for (int j = 0; j < m; ++j) {
            double* d_vj = d_V + static_cast<size_t>(j) * N;

            matvec(d_vj, d_w);

            if (j > 0) {
                double neg_beta = -h_beta[j - 1];
                CHECK_CUBLAS(cublasDaxpy(cublasH, N, &neg_beta, d_V + static_cast<size_t>(j - 1) * N, 1, d_w, 1));
            }

            CHECK_CUBLAS(cublasDdot(cublasH, N, d_vj, 1, d_w, 1, &h_alpha[j]));

            double neg_alpha = -h_alpha[j];
            CHECK_CUBLAS(cublasDaxpy(cublasH, N, &neg_alpha, d_vj, 1, d_w, 1));

            if (j < m - 1) {
                CHECK_CUBLAS(cublasDnrm2(cublasH, N, d_w, 1, &h_beta[j]));

                if (std::abs(h_beta[j]) < config.tolerance) {
                    m_actual = j + 1;
                    break;
                }

                double* d_vnext = d_V + static_cast<size_t>(j + 1) * N;
                CHECK_CUDA(cudaMemcpy(d_vnext, d_w, N * sizeof(double), cudaMemcpyDeviceToDevice));
                double inv_beta = 1.0 / h_beta[j];
                CHECK_CUBLAS(cublasDscal(cublasH, N, &inv_beta, d_vnext, 1));
            }
        }

        std::vector<double> h_T(m_actual * m_actual, 0.0);
        for (int i = 0; i < m_actual; ++i) {
            h_T[i * m_actual + i] = h_alpha[i];
            if (i < m_actual - 1) {
                h_T[i * m_actual + (i + 1)] = h_beta[i];
                h_T[(i + 1) * m_actual + i] = h_beta[i];
            }
        }

        double *d_T = nullptr, *d_W = nullptr;
        CHECK_CUDA(cudaMalloc(&d_T, m_actual * m_actual * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_W, m_actual * sizeof(double)));
        CHECK_CUDA(cudaMemcpy(d_T, h_T.data(), m_actual * m_actual * sizeof(double), cudaMemcpyHostToDevice));

        int lwork = 0;
        CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(
            cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m_actual, d_T, m_actual, d_W, &lwork));

        double* d_work = nullptr;
        int* d_info = nullptr;
        CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

        CHECK_CUSOLVER(cusolverDnDsyevd(
            cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m_actual, d_T, m_actual, d_W, d_work, lwork, d_info));

        int h_info = 0;
        CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            throw std::runtime_error("cuSOLVER syevd failed with info=" + std::to_string(h_info));
        }

        std::vector<double> h_W(m_actual);
        CHECK_CUDA(cudaMemcpy(h_W.data(), d_W, m_actual * sizeof(double), cudaMemcpyDeviceToHost));

        int k_pairs = std::min(config.num_eigenpairs, m_actual);
        Eigensolution sol;
        sol.pairs.resize(k_pairs);

        double* d_x = nullptr;
        CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(double)));

        for (int k = 0; k < k_pairs; ++k) {
            sol.pairs[k].energy = h_W[k];
            sol.pairs[k].state_vector.resize(N);

            double alpha_gemv = 1.0;
            double beta_gemv = 0.0;
            CHECK_CUBLAS(cublasDgemv(
                cublasH, CUBLAS_OP_N, N, m_actual,
                &alpha_gemv, d_V, N,
                d_T + static_cast<size_t>(k) * m_actual, 1,
                &beta_gemv, d_x, 1
            ));

            CHECK_CUDA(cudaMemcpy(sol.pairs[k].state_vector.data(), d_x, N * sizeof(double), cudaMemcpyDeviceToHost));
        }

        cudaFree(d_x);
        cudaFree(d_V);
        cudaFree(d_w);
        cudaFree(d_T);
        cudaFree(d_W);
        cudaFree(d_work);
        cudaFree(d_info);
        cusolverDnDestroy(cusolverH);
        cublasDestroy(cublasH);

        return sol;
    }
#endif // HAS_CUDA_LANCZOS
};
