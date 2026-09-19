#pragma once
#include<bits/stdc++.h>
using namespace std;

extern unordered_map<u_int64_t, vector<int>> cache;
extern int n;


struct EigenPair
{
    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;
};


int calculateEnergy(u_int64_t bitstring);
vector<vector<int>> symmetryFamilies(int bits);
EigenPair lanczos_calc(long double g, int bits, int maxeigenvals);
long double residual(vector<double> &in, long double g, int bits, long double eigenvalue);