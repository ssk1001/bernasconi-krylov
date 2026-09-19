#include <bits/stdc++.h>
#include "link.hpp"
#include "lambda_lanczos/lambda_lanczos.hpp"

using namespace std;
using lambda_lanczos::LambdaLanczos;
const long double STEP_SIZE = 0.1;



long double residual(vector<double> &in, long double g, int bits, 
long double eigenvalue){
    n = bits;
    vector<double> ans(1 << n, 0.0);
    for(int i = 0;i < (1 << n);i++){
        ans[i] += (in[i] * calculateEnergy(i)) / (double)(bits);
        for(int j = 0;j < n;j++){
            int nxt = i ^ (1 << j);
            ans[nxt] += (in[i] * (-g));
        }
    }
    for(int i = 0;i < (1 << n);i++){
        ans[i] -= (eigenvalue * in[i]);
    }
    long double ret = 0;
    for(int i = 0;i < (1 << n);i++){
        ret += ans[i] * ans[i];
        }
    return ret;
}



vector<vector<int>> symmetryFamilies(int bits)
{
    vector<bool> vis(1 << bits, false);
    vector<vector<int>> ans;
    int msk = (1 << bits) - 1;
    int msk1 = 0, msk2 = 0;
    for (int i = 0; i < bits; i++)
    {
        if (i % 2)
            msk1 = (msk1 | (1 << i));
        else
            msk2 = (msk2 | (1 << i));
    }
    for (int i = 0; i < (1 << bits); i++)
    {
        if (!vis[i])
        {
            vector<int> cov;
            queue<int> nodes;
            nodes.push(i);
            vis[i] = true;
            while (!nodes.empty())
            {
                int nxt = nodes.front();
                cov.push_back(nxt);
                nodes.pop();
                if (!vis[msk ^ nxt])
                {
                    vis[msk ^ nxt] = true;
                    nodes.push(msk ^ nxt);
                }
                if (!vis[msk1 ^ nxt])
                {
                    vis[msk1 ^ nxt] = true;
                    nodes.push(msk1 ^ nxt);
                }
                if (!vis[msk2 ^ nxt])
                {
                    vis[msk2 ^ nxt] = true;
                    nodes.push(msk2 ^ nxt);
                }
                int candidate = 0;
                for (int j = 0; j < bits; j++)
                {
                    if (nxt & (1 << j))
                        candidate = (candidate | (1 << (bits - j - 1)));
                }
                if (!vis[candidate])
                {
                    vis[candidate] = true;
                    nodes.push(candidate);
                }
            }
            cout << "Family : ";
            for (auto x : cov)
                cout << x << " ";
            cout << endl;
            ans.push_back(cov);
        }
    }
    return ans;
}

EigenPair lanczos_calc(long double g, int bits, int numeigenvals)
{
    auto mv_mul = [&](const vector<double> &in, vector<double> &out)
    {
        for (u_int64_t i = 0; i < (1ULL << bits); i++)
        {
            out[i] += ((long double)calculateEnergy(i) / (long double)(bits)) * (in[i]);
            for (int j = 0; j < bits; j++)
            {
                u_int64_t nxt = i ^ (1ULL << j);
                out[nxt] += (-g) * (in[i]);
            }
            // for(int j = 0;j < (1ULL << bits);j++){
            //     if(i == j) out[i] += ((long double)calculateEnergy(i) / (long double)(bits)) * (in[i]);
            //     if(__builtin_popcount(i ^ j) == 1) out[j] += (in[i] * (-g));
            // }
        }
    };
    LambdaLanczos<double> engine(mv_mul, 1 << bits, 10, false, numeigenvals);
    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;
    engine.run(eigenvalues, eigenvectors);
    // cout << "For g = " << g << endl;
    // for (int i = 0; i < eigenvalues.size(); i++)
    // {
    //     vector<pair<double, int>> eigs;
    //     for (int j = 0; j < eigenvectors[i].size(); j++)
    //         eigs.push_back({abs(eigenvectors[i][j]), j});
    //     sort(eigs.begin(), eigs.end());
    //     cout << "Eigenvalue = " << eigenvalues[i] << endl;
    //     cout << "Eigenvector" << endl;
    //     for (int j = 0; j < eigenvectors[i].size(); j++)
    //     {
    //         // if(abs(eigenvectors[i][j]) > 1e-10)
    //         cout << eigenvectors[i][j] << endl;
    //     }
    //     cout << endl;
    // }
    EigenPair ans;
    ans.eigenvalues = eigenvalues;
    ans.eigenvectors = eigenvectors;
    return ans;
}

// int main()
// {
//     int bits;
//     cout << "Enter number of bits: ";
//     cin >> bits;
//     n = bits;
//     for (int i = 1; i <= 20; i++)
//     {
//         long double g = (long double)(i)*STEP_SIZE;
//         lanczos_calc(g, bits);
//     }
//     return 0;
// }