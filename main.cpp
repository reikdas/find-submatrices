#include "mtx_to_csr.h"
#include "find_submatrix.h"
#include <omp.h>
#include <thread>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <functional>

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

        // Set up 4-hour timeout
        const auto timeout_duration = std::chrono::hours(4);
        const auto start_time = std::chrono::steady_clock::now();
        bool timeout_reached = false;
        
        auto timeout_check = [&]() -> bool {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed >= timeout_duration) {
                if (!timeout_reached) {
                    timeout_reached = true;
                    std::cerr << "\nWarning: 4-hour timeout reached for matrix " 
                              << matrix_name << ". Saving partial results...\n";
                }
                return true;
            }
            return false;
        };

        decompose_region(
            A_CSR,
            A_CSC,
            full_region,
            blocks,
            row_used,
            col_used,
            timeout_check
        );

        // Create results directory if it doesn't exist
        std::filesystem::create_directories("results");
        
        // Write results to YAML file
        std::string results_file = "results/" + matrix_name + ".yaml";
        std::ofstream out_file(results_file);
        
        out_file << "timeout: " << (timeout_reached ? "true" : "false") << "\n";
        
        if (blocks.empty()) {
            out_file << "blocks: []\n";
            out_file << "message: \"No dense submatrix found (area > 0.5 density, span >= 2500)\"\n";
        } else {
            out_file << "blocks:\n";
            for (size_t i = 0; i < blocks.size(); ++i) {
                const auto& block = blocks[i];
                out_file << "  - rows: [" << block.r0 << ", " << block.r1 << ")\n";
                out_file << "    cols: [" << block.c0 << ", " << block.c1 << ")\n";
                out_file << "    dimensions: [" << (block.r1 - block.r0) << ", " << (block.c1 - block.c0) << "]\n";
                out_file << "    area: " << block.area << "\n";
                out_file << "    density: " << block.density << "\n";
            }
        }
        
        out_file.close();
    } catch (const std::exception& e) {
        std::cerr << "Error processing matrix " << matrix_name << ": " << e.what() << "\n";
        return 1;
    }

    return 0;
}