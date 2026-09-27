#include <vector>
#include <stdexcept>
#include <Eigen/Dense>
#include "../labs_energy/labs_energy.hpp" // Assumes your provided SpinState and SpinLib are here

struct EigensystemSolution {
    double ground_energy;
    Eigen::VectorXd ground_state;
};

class ExactDiagonalizer  {
public:
    static EigensystemSolution solve_full_space(size_t N, double g) {
        size_t dim = 1ULL << N;
        std::vector<SpinState> full_basis(dim);
        
        // Compactly initialize the 2^N space
        for (uint64_t i = 0; i < dim; ++i) {
            full_basis[i] = {N, {i}}; 
        }
        
        return solve_subspace(full_basis, g);
    }

    static EigensystemSolution solve_subspace(const std::vector<SpinState>& basis, double g) {
        size_t dim = basis.size();
        if (dim == 0) return {0.0, Eigen::VectorXd()};
        
        size_t N = basis[0].num_bits;
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
        SpinCache cache; // Instantiate local cache to leverage O(N) neighbor updates during matrix build
        
        for (size_t a = 0; a < dim; ++a) {
            // Diagonal elements: E(s_a) / N retrieved or computed via SpinLib cache[cite: 1, 2]
            H(a, a) = static_cast<double>(SpinLib::get_energy(cache, basis[a])) / N;
            
            // Off-diagonal elements: -g coupling applied exclusively for 1-flip neighbors[cite: 2]
            for (size_t b = a + 1; b < dim; ++b) {
                if (hamming_distance(basis[a], basis[b]) == 1) {
                    H(a, b) = H(b, a) = -g; 
                }
            }
        }
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(H);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error("Eigenvalue decomposition failed");
        }
        
        return {solver.eigenvalues()(0), solver.eigenvectors().col(0)};
    }

private:
    static int hamming_distance(const SpinState& a, const SpinState& b) {
        int dist = 0;
        for (size_t i = 0; i < a.bits.size(); ++i) {
            dist += __builtin_popcountll(a.bits[i] ^ b.bits[i]);
        }
        return dist;
    }
};