#include "link.hpp"
#include <bits/stdc++.h>
using namespace std;

const int MAX_CACHE_SIZE = 1e6 + 1;
unordered_map<u_int64_t, vector<int>> cache;
int n;

void cacheSearch(u_int64_t bitstring, vector<int> *&prevCk, u_int64_t &prevmask)
{
    if (cache.find(bitstring) != cache.end())
    {
        prevmask = bitstring;
        prevCk = &(cache.find(bitstring)->second);
        return;
    }
    for (int i = 0; i < n; i++)
    {
        u_int64_t nxt = (bitstring ^ (1ULL << i));
        if (cache.find(nxt) != cache.end())
        {
            prevmask = nxt;
            prevCk = &(cache.find(nxt)->second);
            return;
        }
    }
}

int calculateEnergy(u_int64_t bitstring)
{
    u_int64_t prevmask;
    vector<int> *prevCk = nullptr;
    cacheSearch(bitstring, prevCk, prevmask);
    int ans = 0;
    if (prevCk != nullptr)
    {
        if (prevmask == bitstring)
            for (int i = 0; i < n - 1; i++)
                ans = (ans + ((*prevCk)[i] * (*prevCk)[i]));
        else
        {
            vector<int> Ck(n - 1);
            for (int i = 0; i < n - 1; i++)
                Ck[i] = (*prevCk)[i];
            int pos = -1;
            for (int i = 0; i < n; i++)
                if (((1ULL << i) & (prevmask)) != ((1ULL << i) & bitstring))
                {
                    pos = i;
                    break;
                }
            assert(pos != -1);
            for (int i = 0; i < n; i++)
            {
                if (i == pos)
                    continue;
                bool first = (1ULL << i) & bitstring;
                bool second = (1ULL << pos) & bitstring;
                if (first == second)
                    Ck[abs(i - pos) - 1] += 2;
                else
                    Ck[abs(i - pos) - 1] -= 2;
            }
            for (int i = 0; i < n - 1; i++)
                ans += Ck[i] * Ck[i];
            cache.emplace(bitstring, move(Ck));
        }
    }
    else
    {
        vector<int> newCk(n - 1);
        for (int i = 0; i < n; i++)
        {
            for (int j = i - 1; j >= 0; j--)
            {
                bool first = (1ULL << i) & bitstring;
                bool second = (1ULL << j) & bitstring;
                newCk[abs(i - j) - 1] += (first == second) ? 1 : -1;
            }
        }
        for (int i = 0; i < n - 1; i++)
            ans += (newCk[i] * newCk[i]);
        cache.emplace(bitstring, move(newCk));
    }
    if (cache.size() > MAX_CACHE_SIZE)
        cache.erase(cache.begin());
    return ans;
}

int testEnergies(int bits)
{
    n = bits;
    int ans = INT_MAX;
    for (u_int64_t i = 0; i < (1ULL << bits); i++)
    {
        int t = calculateEnergy(i);
        // cout<<t<<" ";
        ans = min(ans, t);
    }
    cache.clear();
    return ans;
}

// int main(){
//     //Tests : minimum energies in exhaustive manner

//     cout<<"Test n = 3: Minimum Energy = "<<testEnergies(3)<<((testEnergies(3) == 1) ? " PASSED" : " FAILED")<<endl;
//     cout<<"Test n = 6: Minimum Energy = "<<testEnergies(6)<<((testEnergies(6) == 7) ? " PASSED" : " FAILED")<<endl;
//     cout<<"Test n = 7: Minimum Energy = "<<testEnergies(7)<<((testEnergies(7) == 3) ? " PASSED" : " FAILED")<<endl;
//     cout<<"Test n = 8: Minimum Energy = "<<testEnergies(8)<<((testEnergies(8) == 8) ? " PASSED" : " FAILED")<<endl;
//     cout<<"Test n = 9: Minimum Energy = "<<testEnergies(9)<<((testEnergies(9) == 12) ? " PASSED" : " FAILED")<<endl;
//     cout<<"Enter n: ";
//     cin>>n;
//     for(int i = 0;i < (1 << n);i++){
//         cout<<i<<" "<<calculateEnergy(i)<<endl;
//     }
//     return 0;
// }