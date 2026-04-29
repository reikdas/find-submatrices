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
#include <stdexcept>

namespace {

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program << " <matrix.mtx> [options]\n"
        << "Options:\n"
        << "  --min-density <float>       Minimum accepted block density (default 0.5)\n"
        << "  --min-area <int>            Minimum accepted block area (default 2500)\n"
        << "  --gamma <float>             Density-score exponent (default 1.5)\n"
        << "  --threads <int>             Requested OpenMP threads (default 20)\n"
        << "  --timeout-seconds <float>   Timeout in seconds (default 14400)\n"
        << "  --output <path>             YAML output path (default results/<matrix>.yaml)\n";
}

std::string require_value(int& i, int argc, char* argv[], const std::string& flag) {
    if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
    }
    return argv[++i];
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    PartitionConfig config;
    int requested_threads = 20;
    double timeout_seconds = 4.0 * 60.0 * 60.0;
    std::string output_path;
    const std::string mtx_file_str = argv[1];

    try {
        for (int i = 2; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--min-density") {
                config.min_density = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--min-area") {
                config.min_area = std::stoi(require_value(i, argc, argv, arg));
            } else if (arg == "--gamma") {
                config.gamma = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--threads") {
                requested_threads = std::stoi(require_value(i, argc, argv, arg));
            } else if (arg == "--timeout-seconds") {
                timeout_seconds = std::stod(require_value(i, argc, argv, arg));
            } else if (arg == "--output") {
                output_path = require_value(i, argc, argv, arg);
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("Unknown option: " + arg);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    // Set OpenMP thread count: use 24 or max available cores, whichever is smaller
    const int max_available = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    const int num_threads = std::min(std::max(1, requested_threads), max_available);
    omp_set_num_threads(num_threads);
    
    std::cout << "Using " << num_threads << " OpenMP threads (requested: " 
              << requested_threads << ", available: " << max_available << ")\n";
    
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

        // Set up timeout
        const auto timeout_duration = std::chrono::duration<double>(timeout_seconds);
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
            config,
            timeout_check
        );

        std::filesystem::path results_file =
            output_path.empty()
                ? std::filesystem::path("results") / (matrix_name + ".yaml")
                : std::filesystem::path(output_path);
        std::filesystem::path parent = results_file.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
        std::ofstream out_file(results_file);
        
        out_file << "timeout: " << (timeout_reached ? "true" : "false") << "\n";
        
        if (blocks.empty()) {
            out_file << "blocks: []\n";
            out_file << "message: \"No dense submatrix found (density >= "
                     << config.min_density << ", area >= " << config.min_area << ")\"\n";
        } else {
            out_file << "blocks:\n";
            for (size_t i = 0; i < blocks.size(); ++i) {
                const auto& block = blocks[i];
            out_file << "  - rows: [" << block.r0 << ", " << block.r1 << "]\n";
            out_file << "    cols: [" << block.c0 << ", " << block.c1 << "]\n";
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
