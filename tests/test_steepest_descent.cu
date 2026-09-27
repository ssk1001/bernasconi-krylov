#include <iostream>
#include <chrono>
#include <iomanip>

// Include your header file containing SpinState, SpinCache, SpinLib, 
// and the multi_start_steepest_descent implementation
#include "../src/classical/steepest_descent.cuh" 

int main() {
    // 1. Configuration Parameters
    size_t num_bits;      // Total number of spins (N)
    size_t num_starts;     // Number of random restarts to escape local minima

    std::cout<<"Enter number of bits : ";
    std::cin>>num_bits;
    std::cout<<"Enter number of start states : ";
    std::cin>>num_starts;
    std::cout << "=========================================\n";
    std::cout << " Multi-Start Steepest Descent Optimizer  \n";
    std::cout << "=========================================\n";
    std::cout << "System Size (N) : " << num_bits << " spins\n";
    std::cout << "Start States    : " << num_starts << "\n\n";

    // 2. Initialize Cache
    // Assuming SpinCache is default constructible as defined in your headers
    SpinCache cache; 

    // 3. Execution and Timing
    auto start_time = std::chrono::high_resolution_clock::now();
    
    DescentResult result = multi_start_steepest_descent(num_bits, num_starts, cache);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = end_time - start_time;

    // 4. Summary Output
    std::cout << "\n=========================================\n";
    std::cout << " Optimization Summary\n";
    std::cout << "=========================================\n";
    std::cout << "Total Wall Time : " << std::fixed << std::setprecision(4) 
              << total_elapsed.count() << " seconds\n";
    std::cout << "Global Best     : " << result.best_energy << "\n";
    if(num_bits <= 64) 
        std::cout << "Global Best config = " << result.best_state.bits[0] << "\n";
    return 0;   
}