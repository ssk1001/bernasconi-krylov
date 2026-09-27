#pragma once
#include "../labs_energy/labs_energy.hpp"

#include <slepceps.h>
#include <vector>
#include <cstddef>
#include <unordered_map>

template <typename StateType = SpinState, typename ScalarType = double>
class SpinEigensolver {
public:
    // Eigenpair return structure containing both energy and state vector
    struct Eigenpair {
        ScalarType energy;
        std::vector<ScalarType> state_vector;
    };

    // Endpoint 1: Explicit Subspace Basis
    static Eigenpair compute_ground_state(const std::vector<StateType>& basis, ScalarType g) {
        return solve_subspace_internal(basis, g);
    }

    // Endpoint 2: Full N-bit Space
    static Eigenpair compute_ground_state(size_t num_bits, ScalarType g) {
        return solve_fullspace_internal(num_bits, g);
    }

private:
    struct SubspaceContext {
        const std::vector<StateType>& basis;
        ScalarType g;
        SpinCache sc;
        std::unordered_map<SpinState, int, SpinStateHash> state_to_index;
        Vec x_seq;
        VecScatter scat;
        PetscInt rstart, rend;
    };

    struct FullSpaceContext {
        size_t num_bits;
        ScalarType g;
        SpinCache sc;
        Vec x_seq;
        VecScatter scat;
        PetscInt rstart, rend;
    };

    static PetscErrorCode MatMultSubspaceBridge(Mat A, Vec x, Vec y) {
        SubspaceContext* ctx = nullptr;
        MatShellGetContext(A, &ctx);

        // Gather distributed vector x into local sequential vector x_seq
        VecScatterBegin(ctx->scat, x, ctx->x_seq, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(ctx->scat, x, ctx->x_seq, INSERT_VALUES, SCATTER_FORWARD);

        const PetscScalar *x_arr; 
        PetscScalar *y_arr;
        VecGetArrayRead(ctx->x_seq, &x_arr); // Global indexing available here
        VecGetArray(y, &y_arr);              // Only our local chunk

        int num_bits = ctx->basis[0].num_bits;

        // Only compute for the rows assigned to this specific MPI process
        for (PetscInt i = ctx->rstart; i < ctx->rend; ++i) {
            PetscInt local_i = i - ctx->rstart;
            y_arr[local_i] = 0.0;

            PetscScalar energy = static_cast<PetscScalar>(SpinLib::get_energy(ctx->sc, ctx->basis[i])) 
                               / static_cast<PetscScalar>(num_bits);
            y_arr[local_i] += energy * x_arr[i];

            // Off-diagonal flip terms using pre-built lookup table
            for (int b = 0; b < num_bits; ++b) {
                SpinState flipped = ctx->basis[i].flip_bit(b);

                auto it = ctx->state_to_index.find(flipped);
                if (it != ctx->state_to_index.end()) {
                    y_arr[local_i] -= ctx->g * x_arr[it->second];
                }
            }
        }

        VecRestoreArrayRead(ctx->x_seq, &x_arr); 
        VecRestoreArray(y, &y_arr);
        return 0;
    }

    static PetscErrorCode MatMultFullSpaceBridge(Mat A, Vec x, Vec y) {
        FullSpaceContext* ctx = nullptr;
        MatShellGetContext(A, &ctx);

        // Gather distributed vector x into local sequential vector x_seq
        VecScatterBegin(ctx->scat, x, ctx->x_seq, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(ctx->scat, x, ctx->x_seq, INSERT_VALUES, SCATTER_FORWARD);

        const PetscScalar *x_arr; 
        PetscScalar *y_arr;
        VecGetArrayRead(ctx->x_seq, &x_arr); 
        VecGetArray(y, &y_arr);

        // Only compute for the rows assigned to this specific MPI process
        for (PetscInt i = ctx->rstart; i < ctx->rend; ++i) {
            PetscInt local_i = i - ctx->rstart;
            y_arr[local_i] = 0.0;

            // Off-diagonal transverse field term (X_j spin flips)
            for (size_t j = 0; j < ctx->num_bits; ++j) {
                uint64_t nxt = static_cast<uint64_t>(i) ^ (1ULL << j);
                y_arr[local_i] -= ctx->g * x_arr[nxt];
            }

            // Diagonal interaction term
            PetscScalar energy = static_cast<PetscScalar>(SpinLib::get_energy(ctx->sc, {ctx->num_bits, {static_cast<u_int64_t>(i)}})) 
                               / static_cast<PetscScalar>(ctx->num_bits);
            y_arr[local_i] += energy * x_arr[i];
        }

        VecRestoreArrayRead(ctx->x_seq, &x_arr); 
        VecRestoreArray(y, &y_arr);
        return 0;
    }

    static Eigenpair solve_subspace_internal(const std::vector<StateType>& basis, ScalarType g) {
        SubspaceContext ctx{basis, g, SpinCache{}, {}, NULL, NULL, 0, 0};
        ctx.state_to_index.reserve(basis.size());
        for (int i = 0; i < static_cast<int>(basis.size()); ++i) {
            ctx.state_to_index[basis[i]] = i;
        }

        PetscInt N = static_cast<PetscInt>(basis.size());

        // Create distributed matrix A
        Mat A;
        MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, &ctx, &A);
        MatGetOwnershipRange(A, &ctx.rstart, &ctx.rend);

        // Setup Scatter context for MatMult
        Vec x_dist;
        MatCreateVecs(A, &x_dist, NULL);
        VecCreateSeq(PETSC_COMM_SELF, N, &ctx.x_seq);
        VecScatterCreateToAll(x_dist, &ctx.scat, &ctx.x_seq);
        VecDestroy(&x_dist);

        Eigenpair result = run_slepc_solver(N, MatMultSubspaceBridge, A);

        VecScatterDestroy(&ctx.scat);
        VecDestroy(&ctx.x_seq);
        return result;
    }   

    static Eigenpair solve_fullspace_internal(size_t num_bits, ScalarType g) {
        FullSpaceContext ctx{num_bits, g, SpinCache{}, NULL, NULL, 0, 0};
        PetscInt N = static_cast<PetscInt>(1ULL << num_bits);

        // Create distributed matrix A
        Mat A;
        MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, &ctx, &A);
        MatGetOwnershipRange(A, &ctx.rstart, &ctx.rend);

        // Setup Scatter context for MatMult
        Vec x_dist;
        MatCreateVecs(A, &x_dist, NULL);
        VecCreateSeq(PETSC_COMM_SELF, N, &ctx.x_seq);
        VecScatterCreateToAll(x_dist, &ctx.scat, &ctx.x_seq);
        VecDestroy(&x_dist);

        Eigenpair result = run_slepc_solver(N, MatMultFullSpaceBridge, A);

        VecScatterDestroy(&ctx.scat);
        VecDestroy(&ctx.x_seq);
        return result;
    }

    // Shared SLEPc Execution Pipeline extracting Eigenvalue + Eigenvector
    static Eigenpair run_slepc_solver(PetscInt N, PetscErrorCode (*matmul_bridge)(Mat, Vec, Vec), Mat A) {
        
        MatShellSetOperation(A, MATOP_MULT, (void(*)(void))matmul_bridge);

        EPS eps;
        EPSCreate(PETSC_COMM_WORLD, &eps);
        EPSSetOperators(eps, A, NULL);
        EPSSetProblemType(eps, EPS_HEP);
        EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
        EPSSetType(eps, EPSKRYLOVSCHUR);

        EPSSolve(eps);
        
        // Vr is distributed across MPI nodes
        Vec Vr, Vi;
        MatCreateVecs(A, &Vr, &Vi);

        PetscScalar kr, ki;
        EPSGetEigenpair(eps, 0, &kr, &ki, Vr, Vi);

        // Re-assemble the distributed eigenvector into a complete sequential copy on all nodes
        Vec Vr_seq;
        VecCreateSeq(PETSC_COMM_SELF, N, &Vr_seq);
        VecScatter scat_res;
        VecScatterCreateToAll(Vr, &scat_res, &Vr_seq);
        VecScatterBegin(scat_res, Vr, Vr_seq, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(scat_res, Vr, Vr_seq, INSERT_VALUES, SCATTER_FORWARD);

        const PetscScalar *v_arr = nullptr;
        VecGetArrayRead(Vr_seq, &v_arr);

        Eigenpair result;
        result.energy = PetscRealPart(kr);

        // Explicitly copy all N elements safely
        result.state_vector.resize(N);
        for (PetscInt i = 0; i < N; ++i) {
            result.state_vector[i] = v_arr[i];
        }

        VecRestoreArrayRead(Vr_seq, &v_arr);

        // Clean up PETSc objects
        VecDestroy(&Vr_seq);
        VecScatterDestroy(&scat_res);
        VecDestroy(&Vr);
        VecDestroy(&Vi);
        MatDestroy(&A);
        EPSDestroy(&eps);

        return result;
    }
};