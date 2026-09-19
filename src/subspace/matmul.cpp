#include<bits/stdc++.h>
#include"cache.hpp"
using namespace std;








int main(){
    // int n;
    cout<<"Enter n: ";
    cin>>n;
    long double g;
    cout<<"Enter g :";
    cin>>g;
    long double scale;
    cout<<"Enter scaling factor : ";
    cin>>scale;
    cout<<"Enter vector: (format a0, a1.... a(1 << n - 1))";
    vector<long double> a(1 << n);
    for(int i = 0;i < 1 << n;i++) cin>>a[i];
    vector<long double> ans(1 << n);
    for(int i = 0;i < (1 << n);i++){
        for(int j = 0;j < (1 << n);j++){
            if(i == j) ans[j] += ((long double)calculateEnergy(i) * a[i]) / (long double)(n);
            else if(__builtin_popcount(i ^ j) == 1) ans[j] += a[i] * (-g);
        }
    }
    for(auto &x : ans) x /= scale;
    cout<<"Vector diff:"<<endl;
    for(int i = 0;i < (1 << n);i++) cout<<(a[i] - ans[i])<<endl;
    return 0;
}