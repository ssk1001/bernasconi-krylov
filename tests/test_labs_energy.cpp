#include "../src/labs_energy/labs_energy.hpp"
#include<stdio.h>

using namespace SpinLib;



int main(){
    SpinCache sc;
    u_int64_t mask;
    printf("Testing LABS energy with configs bits = n, mask = n\n");    
    int energies[] = {0, 0, 1, 1, 2, 6, 7, 35, 32, 24, 45};
    for(int i = 2;i <= 10;i++){
        SpinState temp;
        temp.num_bits = i;
        temp.bits = {(uint64_t)i};
        printf("For bits = %d, config = %d, energy = %ld, actual = %d\n", i, i, get_energy(sc, temp), energies[i]);
        sc.clear();
    }
    return 0;
}