#include <bits/stdc++.h>
#include <Eigen/Dense>
#include "link.hpp"
using namespace std;
using namespace Eigen;

int main()
{
    cout << "Enter number of bits: ";
    cin >> n;
    // long double g;
    // cout<<"Enter flipping factor: ";
    // cin>>g;
    cout << "Enter max eigenvalues: ";
    int maxegv;
    cin >> maxegv;
    ofstream outFile("data_raw/spectrum/samplerun.txt");
    ofstream eigenData("data_raw/spectrum/sample_eigen.txt");
    vector<vector<int>> symGroups = symmetryFamilies(n);
    for (int i = 0; i <= 20; i++)
    {
        long double g = (long double)(i) * 1.0;
        MatrixXd A(1 << n, 1 << n);
        for (int k = 0; k < (1 << n); k++)
        {
            for (int j = 0; j < (1 << n); j++)
            {
                if (k == j)
                    A(k, j) = (long double)calculateEnergy(k) / (long double)(n);
                else if (__builtin_popcount(k ^ j) == 1)
                    A(k, j) = (-g);
            }
        }
        SelfAdjointEigenSolver<MatrixXd> solver(A);
        if (solver.info() != Eigen::Success)
        {
            std::cerr << "Eigenpair calculation failed to converge!" << std::endl;
            return 1;
        }
        VectorXd eigenvalues = solver.eigenvalues();
        // for(int j = 0;j < eigenvalues.size();j++) outFile<<g<<" "<<eigenvalues(j)<<endl;
        EigenPair ep = lanczos_calc(g, n, maxegv);
        for (auto x : ep.eigenvalues)
            outFile << g << " " << x << endl;
        VectorXd loweststate = solver.eigenvectors().col(0);
        long double ans = 0;
        cout<<"g = "<<g<<endl;
        cout<<"Eigenstate : "<<endl;
        for(int j = 0;j < loweststate.size();j++) cout<<loweststate(j)/loweststate(0)<<endl;
        for(int famnumber = 0;famnumber < symGroups.size();famnumber++){
            int energy = calculateEnergy(symGroups[famnumber][0]);
            long double coeff = 0;
            for(auto x : symGroups[famnumber]) coeff += (ep.eigenvectors[0][x] * ep.eigenvectors[0][x]);
            cout<<"Symmetry rep = "<<symGroups[famnumber][0]<<endl;
            cout<<"Contribution = "<<coeff<<" Energy = "<<energy<<endl;
            eigenData<<g<<" "<<energy<<" "<<coeff<<endl;
        }
        cout<<"Residual of lowest eigenstate = "<<endl;
        long double res = residual(ep.eigenvectors[0], g, n, ep.eigenvalues[0]);
        cout<<res<<endl;
        for(int j = 0;j < loweststate.size();j++) ans += solver.eigenvectors().col(0)(j) * ep.eigenvectors[0][j];
        cout<<"Overlap with ground state = "<<ans<<endl;
    }
    return 0;
}