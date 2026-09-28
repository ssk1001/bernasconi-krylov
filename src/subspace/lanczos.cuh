#pragma once
#include "../labs_energy/labs_energy.hpp"

#include <vector>
#include <cstddef>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// CUDA / cuBLAS / cuSOLVER
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

// ---------------------------------------------------------
// CUDA Error Checking Macros
// ---------------------------------------------------------
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in ") + __FILE__ + ":" + std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
    } \
}
#define CHECK_CUBLAS(call) { if(call != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS error"); }
#define CHECK_CUSOLVER(call) { if(call != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER error"); }

// ---------------------------------------------------------
// Type Traits 
// ---------------------------------------------------------
template <typename T> struct CudaTraits;

template <> struct CudaTraits<double> {
    static cublasStatus_t dot(cublasHandle_t h, int n, const double *x, int incx, const double *y, int incy, double *res) { return cublasDdot(h, n, x, incx, y, incy, res); }
    static cublasStatus_t axpy(cublasHandle_t h, int n, const double *alpha, const double *x, int incx, double *y, int incy) { return cublasDaxpy(h, n, alpha, x, incx, y, incy); }
    static cublasStatus_t scal(cublasHandle_t h, int n, const double *alpha, double *x, int incx) { return cublasDscal(h, n, alpha, x, incx); }
    static cublasStatus_t nrm2(cublasHandle_t h, int n, const double *x, int incx, double *res) { return cublasDnrm2(h, n, x, incx, res); }
    static cublasStatus_t gemv(cublasHandle_t h, cublasOperation_t trans, int m, int n, const double *alpha, const double *A, int lda, const double *x, int incx, const double *beta, double *y, int incy) { return cublasDgemv(h, trans, m, n, alpha, A, lda, x, incx, beta, y, incy); }
    
    static cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const double *A, int lda, const double *W, int *lwork) { return cusolverDnDsyevd_bufferSize(h, jobz, uplo, n, A, lda, W, lwork); }
    static cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, double *A, int lda, double *W, double *work, int lwork, int *info) { return cusolverDnDsyevd(h, jobz, uplo, n, A, lda, W, work, lwork, info); }
};

template <> struct CudaTraits<float> {
    static cublasStatus_t dot(cublasHandle_t h, int n, const float *x, int incx, const float *y, int incy, float *res) { return cublasSdot(h, n, x, incx, y, incy, res); }
    static cublasStatus_t axpy(cublasHandle_t h, int n, const float *alpha, const float *x, int incx, float *y, int incy) { return cublasSaxpy(h, n, alpha, x, incx, y, incy); }
    static cublasStatus_t scal(cublasHandle_t h, int n, const float *alpha, float *x, int incx) { return cublasSscal(h, n, alpha, x, incx); }
    static cublasStatus_t nrm2(cublasHandle_t h, int n, const float *x, int incx, float *res) { return cublasSnrm2(h, n, x, incx, res); }
    static cublasStatus_t gemv(cublasHandle_t h, cublasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy) { return cublasSgemv(h, trans, m, n, alpha, A, lda, x, incx, beta, y, incy); }
    
    static cusolverStatus_t syevd_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, const float *A, int lda, const float *W, int *lwork) { return cusolverDnSsyevd_bufferSize(h, jobz, uplo, n, A, lda, W, lwork); }
    static cusolverStatus_t syevd(cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, float *A, int lda, float *W, float *work, int lwork, int *info) { return cusolverDnSsyevd(h, jobz, uplo, n, A, lda, W, work, lwork, info); }
};

// ---------------------------------------------------------
// GPU Kernels
// ---------------------------------------------------------

template <typename ScalarType>
__global__ void fullspace_matvec_kernel(int N, int num_bits, ScalarType g, const ScalarType* diag, const ScalarType* x, ScalarType* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        ScalarType yi = diag[i] * x[i]; 
        for (int j = 0; j < num_bits; ++j) {
            int nxt = i ^ (1 << j);
            yi -= g * x[nxt];
        }
        y[i] = yi;
    }
}

template <typename ScalarType>
__global__ void subspace_matvec_kernel(int N, int num_bits, ScalarType g, const uint64_t* basis_bits, const ScalarType* diag, const ScalarType* x, ScalarType* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        uint64_t my_bits = basis_bits[i];
        ScalarType yi = diag[i] * x[i];
        
        for (int j = 0; j < num_bits; ++j) {
            uint64_t flipped = my_bits ^ (1ULL << j);
            
            // In-kernel binary search 
            int left = 0, right = N - 1;
            while (left <= right) {
                int mid = left + (right - left) / 2;
                uint64_t mid_val = basis_bits[mid];
                
                if (mid_val == flipped) {
                    yi -= g * x[mid];
                    break;
                } else if (mid_val < flipped) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        y[i] = yi;
    }
}

// ---------------------------------------------------------
// Main Solver Class
// ---------------------------------------------------------
template <typename StateType = SpinState, typename ScalarType = double>
class SpinEigensolver {
public:
    struct Eigenpair {
        ScalarType energy;
        std::vector<ScalarType> state_vector;
    };

    static Eigenpair compute_ground_state(const std::vector<StateType>& basis, ScalarType g) {
        return solve_subspace_internal(basis, g);
    }

    static Eigenpair compute_ground_state(size_t num_bits, ScalarType g) {
        return solve_fullspace_internal(num_bits, g);
    }

private:
    // Helper to extract bitmask representation for sorting and binary searching
    // Adjust based on how SpinState internally stores its bits.
    static uint64_t extract_bits(const StateType& state) {
        // Assuming your SpinState struct contains a std::vector<uint64_t> or similar structure 
        // based on your initialization pattern {ctx.num_bits, {val}}.
        // Example: return state.bits[0];
        
        // If SpinState has a method to get raw bits:
        // return state.to_uint64(); 
        
        // Temporary placeholder - adapt to actual SpinState struct fields
        return state.bits[0]; 
    }

    struct SubspaceSortData {
        uint64_t bits;
        ScalarType energy;
        int orig_index;
        
        bool operator<(const SubspaceSortData& other) const {
            return bits < other.bits;
        }
    };

    static Eigenpair solve_subspace_internal(const std::vector<StateType>& basis, ScalarType g) {
        int N = basis.size();
        if (N == 0) return {0.0, {}};
        int num_bits = basis[0].num_bits;
        SpinCache sc;

        // 1. Precompute energies and sort basis on CPU for O(log N) GPU binary search
        std::vector<SubspaceSortData> host_data(N);
        for (int i = 0; i < N; ++i) {
            host_data[i].bits = extract_bits(basis[i]);
            host_data[i].energy = static_cast<ScalarType>(SpinLib::get_energy(sc, basis[i])) / static_cast<ScalarType>(num_bits);
            host_data[i].orig_index = i;
        }

        std::sort(host_data.begin(), host_data.end());

        // Extract contiguous arrays for device transfer
        std::vector<uint64_t> h_basis_bits(N);
        std::vector<ScalarType> h_diag(N);
        std::vector<int> h_orig_idx(N);

        for (int i = 0; i < N; ++i) {
            h_basis_bits[i] = host_data[i].bits;
            h_diag[i] = host_data[i].energy;
            h_orig_idx[i] = host_data[i].orig_index;
        }

        // 2. Setup Device Memory
        uint64_t* d_basis_bits;
        ScalarType* d_diag;
        CHECK_CUDA(cudaMalloc(&d_basis_bits, N * sizeof(uint64_t)));
        CHECK_CUDA(cudaMalloc(&d_diag, N * sizeof(ScalarType)));
        
        CHECK_CUDA(cudaMemcpy(d_basis_bits, h_basis_bits.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_diag, h_diag.data(), N * sizeof(ScalarType), cudaMemcpyHostToDevice));

        auto matvec = [&](const ScalarType* d_x, ScalarType* d_y) {
            int threads = 256;
            int blocks = (N + threads - 1) / threads;
            subspace_matvec_kernel<<<blocks, threads>>>(N, num_bits, g, d_basis_bits, d_diag, d_x, d_y);
            CHECK_CUDA(cudaDeviceSynchronize());
        };

        Eigenpair raw_result = run_cuda_lanczos(N, matvec);

        CHECK_CUDA(cudaFree(d_basis_bits));
        CHECK_CUDA(cudaFree(d_diag));

        // 3. Un-shuffle the eigenvector back into the user's original basis order
        Eigenpair final_result;
        final_result.energy = raw_result.energy;
        final_result.state_vector.resize(N);
        
        for (int i = 0; i < N; ++i) {
            final_result.state_vector[h_orig_idx[i]] = raw_result.state_vector[i];
        }

        return final_result;
    }

    static Eigenpair solve_fullspace_internal(size_t num_bits, ScalarType g) {
        int N = 1 << num_bits;
        SpinCache sc;

        std::vector<ScalarType> h_diag(N);
        for (int i = 0; i < N; ++i) {
            h_diag[i] = static_cast<ScalarType>(SpinLib::get_energy(sc, {static_cast<uint32_t>(num_bits), {static_cast<uint64_t>(i)}})) 
                      / static_cast<ScalarType>(num_bits);
        }
        std::cout << "[DEBUG] FullSpace h_diag[0] = " << h_diag[0] 
          << ", h_diag[N-1] = " << h_diag[N-1] << "\n";
        ScalarType* d_diag;
        CHECK_CUDA(cudaMalloc(&d_diag, N * sizeof(ScalarType)));
        CHECK_CUDA(cudaMemcpy(d_diag, h_diag.data(), N * sizeof(ScalarType), cudaMemcpyHostToDevice));

        auto matvec = [&](const ScalarType* d_x, ScalarType* d_y) {
            int threads = 256;
            int blocks = (N + threads - 1) / threads;
            fullspace_matvec_kernel<<<blocks, threads>>>(N, num_bits, g, d_diag, d_x, d_y);
            CHECK_CUDA(cudaGetLastError()); 
            CHECK_CUDA(cudaDeviceSynchronize());
        };

        Eigenpair result = run_cuda_lanczos(N, matvec);
        CHECK_CUDA(cudaFree(d_diag));
        return result;
    }

    // ---------------------------------------------------------
    // Shared GPU Lanczos Routine
    // ---------------------------------------------------------
    template <typename MatVecFunc>
    static Eigenpair run_cuda_lanczos(int N, MatVecFunc matvec, int max_krylov_dim = 100) {
        cublasHandle_t cublasH;
        CHECK_CUBLAS(cublasCreate(&cublasH));

        int m = std::min(N, max_krylov_dim);
        
        ScalarType *d_V, *d_w;
        CHECK_CUDA(cudaMalloc(&d_V, N * m * sizeof(ScalarType)));
        CHECK_CUDA(cudaMalloc(&d_w, N * sizeof(ScalarType)));

        std::vector<ScalarType> h_alpha(m, 0.0);
        std::vector<ScalarType> h_beta(m, 0.0);

        // Ensure rand() is seeded (requires <cstdlib>)
        srand(42); 
        std::vector<ScalarType> h_v0(N);
        for (int i = 0; i < N; ++i) h_v0[i] = ((ScalarType)rand() / RAND_MAX) - 0.5;
        CHECK_CUDA(cudaMemcpy(d_V, h_v0.data(), N * sizeof(ScalarType), cudaMemcpyHostToDevice));

        ScalarType v0_norm = 0;
        CHECK_CUBLAS(CudaTraits<ScalarType>::nrm2(cublasH, N, d_V, 1, &v0_norm));
        
        std::cout << "[DEBUG] v0_norm = " << v0_norm << "\n"; // DEBUG LINE

        ScalarType inv_norm = 1.0 / v0_norm;
        CHECK_CUBLAS(CudaTraits<ScalarType>::scal(cublasH, N, &inv_norm, d_V, 1));

        int m_actual = m;
        for (int j = 0; j < m; ++j) {
            ScalarType* d_vj = d_V + j * N;
            
            matvec(d_vj, d_w);

            if (j > 0) {
                ScalarType neg_beta = -h_beta[j - 1];
                CHECK_CUBLAS(CudaTraits<ScalarType>::axpy(cublasH, N, &neg_beta, d_V + (j - 1) * N, 1, d_w, 1));
            }

            CHECK_CUBLAS(CudaTraits<ScalarType>::dot(cublasH, N, d_vj, 1, d_w, 1, &h_alpha[j]));

            ScalarType neg_alpha = -h_alpha[j];
            CHECK_CUBLAS(CudaTraits<ScalarType>::axpy(cublasH, N, &neg_alpha, d_vj, 1, d_w, 1));

            if (j < m - 1) {
                CHECK_CUBLAS(CudaTraits<ScalarType>::nrm2(cublasH, N, d_w, 1, &h_beta[j]));
                
                std::cout << "[DEBUG] Iter " << j << " | alpha: " << h_alpha[j] << " | beta: " << h_beta[j] << "\n"; // DEBUG LINE

                if (std::abs(h_beta[j]) < 1e-9) { 
                    std::cout << "[DEBUG] Krylov space exhausted at iteration " << j << "\n";
                    m_actual = j + 1; 
                    break; 
                }

                ScalarType* d_vnext = d_V + (j + 1) * N;
                CHECK_CUDA(cudaMemcpy(d_vnext, d_w, N * sizeof(ScalarType), cudaMemcpyDeviceToDevice));
                ScalarType inv_beta = 1.0 / h_beta[j];
                CHECK_CUBLAS(CudaTraits<ScalarType>::scal(cublasH, N, &inv_beta, d_vnext, 1));
            }
        }

        std::vector<ScalarType> h_T(m_actual * m_actual, 0.0);
        for (int i = 0; i < m_actual; ++i) {
            h_T[i * m_actual + i] = h_alpha[i];
            if (i < m_actual - 1) {
                h_T[i * m_actual + (i + 1)] = h_beta[i];
                h_T[(i + 1) * m_actual + i] = h_beta[i];
            }
        }

        cusolverDnHandle_t cusolverH = NULL;
        CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

        ScalarType *d_T, *d_W;
        CHECK_CUDA(cudaMalloc(&d_T, m_actual * m_actual * sizeof(ScalarType)));
        CHECK_CUDA(cudaMalloc(&d_W, m_actual * sizeof(ScalarType)));
        CHECK_CUDA(cudaMemcpy(d_T, h_T.data(), m_actual * m_actual * sizeof(ScalarType), cudaMemcpyHostToDevice));

        int lwork = 0;
        CHECK_CUSOLVER(CudaTraits<ScalarType>::syevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m_actual, d_T, m_actual, d_W, &lwork));
        
        ScalarType* d_work;
        CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(ScalarType)));
        
        int* d_info;
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

        CHECK_CUSOLVER(CudaTraits<ScalarType>::syevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, m_actual, d_T, m_actual, d_W, d_work, lwork, d_info));

        // Fetch cuSOLVER info to check for convergence failure
        int h_info = 0;
        CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        std::cout << "[DEBUG] cuSOLVER info = " << h_info << "\n"; // DEBUG LINE

        std::vector<ScalarType> h_W(m_actual);
        CHECK_CUDA(cudaMemcpy(h_W.data(), d_W, m_actual * sizeof(ScalarType), cudaMemcpyDeviceToHost));
        
        int min_idx = std::distance(h_W.begin(), std::min_element(h_W.begin(), h_W.end()));
        ScalarType min_energy = h_W[min_idx];

        ScalarType* d_x;
        CHECK_CUDA(cudaMalloc(&d_x, N * sizeof(ScalarType)));
        
        ScalarType alpha_gemv = 1.0, beta_gemv = 0.0;
        CHECK_CUBLAS(CudaTraits<ScalarType>::gemv(cublasH, CUBLAS_OP_N, N, m_actual, &alpha_gemv, d_V, N, 
                                                  d_T + min_idx * m_actual, 1, &beta_gemv, d_x, 1));

        Eigenpair result;
        result.energy = min_energy;
        result.state_vector.resize(N);
        CHECK_CUDA(cudaMemcpy(result.state_vector.data(), d_x, N * sizeof(ScalarType), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_V));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_T));
        CHECK_CUDA(cudaFree(d_W));
        CHECK_CUDA(cudaFree(d_work));
        CHECK_CUDA(cudaFree(d_info));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
        CHECK_CUBLAS(cublasDestroy(cublasH));

        return result;
    }
};