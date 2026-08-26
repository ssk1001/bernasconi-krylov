#include<bits/stdc++.h>
#include "rapidcsv.h"
#include<filesystem>
#include<chrono>
using namespace std;
using namespace rapidcsv;



int main(){
    auto start = std::chrono::high_resolution_clock::now();
    string filename = "";
    int n;
    cin>>n;
    filename.append("data_raw/classical/exhaustive/");
    filename.append(to_string(n));
    filename.append(".csv");
    if(filesystem::exists(filename)){
        Document doc(filename, LabelParams(0, -1));
        vector<long long> energies = doc.GetColumn<long long>("E(s)");
        int mask = (min_element(energies.begin(), energies.end())) - energies.begin();
        cout<<"Minimum energy config mask: ";
        cout<<mask<<endl;
        cout<<"Minimum energy = "<<energies[mask]<<endl;
        // doc.Save(filename);
    }else{
        Document doc("", LabelParams(0, -1));
        vector<long long> seqs(1 << n);
        iota(seqs.begin(), seqs.end(), 0);
        doc.InsertColumn(0, seqs, "Sequence");
        doc.RemoveColumn("");
        fill(seqs.begin(), seqs.end(), 0);
        for(int i = 1;i < n;i++){
            string temp = "C";
            temp.append(to_string(i));
            doc.InsertColumn(doc.GetColumnCount(), seqs, temp);
        }
        doc.InsertColumn(doc.GetColumnCount(), seqs, "E(s)");
        long long E = 0;
        for(int i = 1;i < n;i++){
            string temp = "C";
            temp.append(to_string(i));
            doc.SetCell(temp, 0, n - i);
            E += (long long)(n - i) * (long long)(n - i);
        }
        doc.SetCell("E(s)", 0, E);
        for(int msk = 1;msk < (1 << n);msk++){
            E = 0;
            int pos = -1;
            for(int j = 0;j < n;j++){
                if(msk & (1 << j)){
                    pos = j;
                    break;
                }
            }
            if(pos == -1){
                cout<<"No set bits."<<endl;
                exit(1);
            }
            int base = msk ^ (1 << pos);
            vector<long long> base_data = doc.GetRow<long long>(base);
            vector<long long> new_data = base_data;
            new_data[0] = msk;
            for(int j = 0;j < n;j++){
                if(j == pos) continue;
                if((1 << j) & msk) new_data[abs(pos - j)] += 2ll;
                else new_data[abs(pos - j)] -= 2ll;
            }
            for(int j = 1;j < n;j++) E += new_data[j] * new_data[j];
            new_data[new_data.size() - 1] = E;
            doc.SetRow(msk, new_data);
        }
        vector<long long> energies = doc.GetColumn<long long>("E(s)");
        int mask = (min_element(energies.begin(), energies.end())) - energies.begin();
        cout<<"Minimum energy config mask: ";
        cout<<mask<<endl;
        cout<<"Minimum energy = "<<energies[mask]<<endl;
        doc.Save(filename);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    return 0;
}