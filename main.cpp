#include "mtx_to_csr.h"
#include "find_submatrix.h"
#include <omp.h>
#include <thread>
#include <filesystem>
#include <algorithm>

int main() {
    // Set OpenMP thread count: use 24 or max available cores, whichever is smaller
    const int requested_threads = 24;
    const int max_available = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int num_threads = std::min(requested_threads, max_available);
    omp_set_num_threads(num_threads);
    
    std::cout << "Using " << num_threads << " OpenMP threads (requested: " 
              << requested_threads << ", available: " << max_available << ")\n";
    
    const std::string suitesparse_dir_str = "/local/scratch/a/das160/partition-matrix/Suitesparse";
    
    // Iterate over all subdirectories in Suitesparse
    for (const auto& entry : std::filesystem::directory_iterator(suitesparse_dir_str)) {
        if (!entry.is_directory()) {
            continue;  // Skip files, only process directories
        }
        
        std::string matrix_name = entry.path().filename().string();
        std::string matrix_path_str = entry.path().string();
        std::string mtx_file_str = matrix_path_str + "/" + matrix_name + ".mtx";
        
        // Check if the .mtx file exists
        if (!std::filesystem::exists(mtx_file_str)) {
            continue;  // Skip if .mtx file doesn't exist
        }
        
        std::cout << "\n========================================\n";
        std::cout << "Processing matrix: " << matrix_name << "\n";
        std::cout << "File: " << mtx_file_str << "\n";
        std::cout << "========================================\n";
        
        try {
            SpMatrixCOO<float> A = read_matrix<float>(mtx_file_str);
            
            // Check if matrix was read successfully
            if (A.num_rows == 0 && A.num_cols == 0) {
                std::cerr << "Warning: Failed to read matrix " << matrix_name << ", skipping...\n";
                continue;
            }
            
            SpMatrixCSR<float> A_CSR = A.to_csr();
            SpMatrixCSC<float> A_CSC = A.to_csc();
            // print(A_CSR);
            // print(A_CSC);
            BestSubmatrix best;

            dense_pass_generic(A_CSR, best);  // CSR pass (height ≥ 2500)
            dense_pass_generic(A_CSC, best);  // CSC pass (width ≥ 2500)

            if (best.area > 0) {
                std::cout << "Best submatrix found:\n";
                std::cout << "  Rows: [" << best.r0 << ", " << best.r1 << ")\n";
                std::cout << "  Cols: [" << best.c0 << ", " << best.c1 << ")\n";
                std::cout << "  Dimensions: "
                          << (best.r1 - best.r0) << " x "
                          << (best.c1 - best.c0) << "\n";
                std::cout << "  Area: " << best.area << "\n";
            } else {
                std::cout << "No dense submatrix found (area > 0.5 density, span >= 2500)\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing matrix " << matrix_name << ": " << e.what() << "\n";
            continue;
        }
    }

    return 0;
}