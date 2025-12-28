#include "mtx_to_csr.h"
#include "find_submatrix.h"
#include <omp.h>
#include <thread>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>

int main(int /* argc */, char* argv[]) {
    // Set OpenMP thread count: use 24 or max available cores, whichever is smaller
    const int requested_threads = 20;
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

        std::vector<BestSubmatrix> blocks;
        std::vector<bool> row_used(A.num_rows, false);
        std::vector<bool> col_used(A.num_cols, false);
        Region full_region{0, A.num_rows, 0, A.num_cols};

        decompose_region(
            A_CSR,
            A_CSC,
            full_region,
            blocks,
            row_used,
            col_used
        );

        // Create results directory if it doesn't exist
        std::filesystem::create_directories("results");
        
        // Write results to file
        std::string results_file = "results/" + matrix_name + ".info";
        std::ofstream out_file(results_file);
        
        if (blocks.empty()) {
            out_file << "No dense submatrix found (area > 0.5 density, span >= 2500)\n";
        } else {
            out_file << "Found " << blocks.size() << " dense submatrix(es):\n\n";
            for (size_t i = 0; i < blocks.size(); ++i) {
                const auto& block = blocks[i];
                out_file << "Block " << (i + 1) << ":\n";
                out_file << "  Rows: [" << block.r0 << ", " << block.r1 << ")\n";
                out_file << "  Cols: [" << block.c0 << ", " << block.c1 << ")\n";
                out_file << "  Dimensions: "
                          << (block.r1 - block.r0) << " x "
                          << (block.c1 - block.c0) << "\n";
                out_file << "  Area: " << block.area << "\n";
                out_file << "  Density: " << block.density << "\n";
                if (i < blocks.size() - 1) {
                    out_file << "\n";
                }
            }
        }
        
        out_file.close();
    } catch (const std::exception& e) {
        std::cerr << "Error processing matrix " << matrix_name << ": " << e.what() << "\n";
        return 1;
    }

    return 0;
}