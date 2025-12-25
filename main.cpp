#include "mtx_to_csr.h"
#include "find_submatrix.h"
#include <omp.h>
#include <thread>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[]) {
    // Set OpenMP thread count: use 24 or max available cores, whichever is smaller
    const int requested_threads = 24;
    const int max_available = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int num_threads = std::min(requested_threads, max_available);
    omp_set_num_threads(num_threads);
    
    std::cout << "Using " << num_threads << " OpenMP threads (requested: " 
              << requested_threads << ", available: " << max_available << ")\n";
    
    const std::string mtx_file_str = argv[1];
    
    // Check if the .mtx file exists
    if (!std::filesystem::exists(mtx_file_str)) {
        std::cerr << "Error: Matrix file does not exist: " << mtx_file_str << "\n";
        return 1;
    }
    
    std::string matrix_name = std::filesystem::path(mtx_file_str).stem().string();
    
    std::cout << "\n========================================\n";
    std::cout << "Processing matrix: " << matrix_name << "\n";
    std::cout << "File: " << mtx_file_str << "\n";
    std::cout << "========================================\n";
    
    try {
        SpMatrixCOO<float> A = read_matrix<float>(mtx_file_str);
        
        // Check if matrix was read successfully
        if (A.num_rows == 0 && A.num_cols == 0) {
            std::cerr << "Warning: Failed to read matrix " << matrix_name << "\n";
            return 1;
        }
        
        SpMatrixCSR<float> A_CSR = A.to_csr();
        SpMatrixCSC<float> A_CSC = A.to_csc();
        // print(A_CSR);
        // print(A_CSC);
        BestSubmatrix best;

        dense_pass_generic(A_CSR, best);  // CSR pass (height ≥ 2500)
        dense_pass_generic(A_CSC, best);  // CSC pass (width ≥ 2500)

        if (best.area == 0) {
            general_rectangle_pass(A_CSR, best);
        }

        // Create results directory if it doesn't exist
        std::filesystem::create_directories("results");
        
        // Write results to file
        std::string results_file = "results/" + matrix_name + ".info";
        std::ofstream out_file(results_file);
        
        if (best.area > 0) {
            out_file << "Best submatrix found:\n";
            out_file << "  Rows: [" << best.r0 << ", " << best.r1 << ")\n";
            out_file << "  Cols: [" << best.c0 << ", " << best.c1 << ")\n";
            out_file << "  Dimensions: "
                      << (best.r1 - best.r0) << " x "
                      << (best.c1 - best.c0) << "\n";
            out_file << "  Area: " << best.area << "\n";
        } else {
            out_file << "No dense submatrix found (area > 0.5 density, span >= 2500)\n";
        }
        
        out_file.close();
    } catch (const std::exception& e) {
        std::cerr << "Error processing matrix " << matrix_name << ": " << e.what() << "\n";
        return 1;
    }

    return 0;
}