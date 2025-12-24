#include "mtx_to_csr.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>

// Helper function to read raw Matrix Market file and extract all entries
template<typename T>
std::map<std::pair<int, int>, T> read_raw_matrix_market(const std::string& filename) {
    std::map<std::pair<int, int>, T> entries;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return entries;
    }
    
    std::string line;
    bool banner_found = false;
    MatrixSymmetry symmetry = MatrixSymmetry::GENERAL;
    
    // Read until we find the banner
    while (std::getline(file, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty()) continue;
        
        if (trimmed[0] == '%' && trimmed[1] == '%') {
            parse_banner(trimmed, symmetry);
            banner_found = true;
            break;
        }
    }
    
    if (!banner_found) {
        std::cerr << "Error: Matrix Market banner not found" << std::endl;
        file.close();
        return entries;
    }
    
    // Skip comments and find dimensions line
    int num_rows, num_cols, num_nonzeros;
    while (std::getline(file, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty()) continue;
        if (trimmed[0] == '%') continue;
        
        std::istringstream iss(trimmed);
        if (iss >> num_rows >> num_cols >> num_nonzeros) {
            break;
        }
    }
    
    // Read matrix entries
    int row, col;
    T value;
    int entries_read = 0;
    
    while (std::getline(file, line) && entries_read < num_nonzeros) {
        std::string trimmed = trim(line);
        if (trimmed.empty()) continue;
        if (trimmed[0] == '%') continue;
        
        std::istringstream iss(trimmed);
        if (!(iss >> row >> col)) {
            continue;
        }
        
        // Matrix Market uses 1-based indexing, convert to 0-based
        row--;
        col--;
        
        // Read value
        if (!(iss >> value)) {
            value = T(1.0);
        }
        
        // Store entry
        entries[{row, col}] = value;
        entries_read++;
        
        // Handle symmetric matrices: add lower triangle entry
        if (symmetry == MatrixSymmetry::SYMMETRIC && row != col) {
            entries[{col, row}] = value;
        }
        // Handle skew-symmetric matrices: add negative lower triangle entry
        else if (symmetry == MatrixSymmetry::SKEW_SYMMETRIC && row != col) {
            entries[{col, row}] = -value;
        }
        // Handle hermitian matrices: add conjugate lower triangle entry
        else if (symmetry == MatrixSymmetry::HERMITIAN && row != col) {
            entries[{col, row}] = value;  // For real matrices, conjugate is same
        }
    }
    
    file.close();
    return entries;
}

// Helper function to check if two floating point values are approximately equal
template<typename T>
bool approx_equal(T a, T b, T tolerance = T(1e-6)) {
    return std::abs(a - b) < tolerance;
}

// Test that CSR indices are sorted within each row
template<typename T>
bool test_csr_indices_sorted(const SpMatrixCompressed<T>& csr) {
    if (csr.direction != CompressionDirection::ROW) {
        std::cerr << "ERROR: Expected ROW direction for CSR" << std::endl;
        return false;
    }
    for (int i = 0; i < csr.num_rows; i++) {
        int start = csr.indptr[i];
        int end = csr.indptr[i+1];
        for (int j = start + 1; j < end; j++) {
            if (csr.indices[j] < csr.indices[j-1]) {
                std::cerr << "ERROR: CSR indices not sorted in row " << i 
                          << " at position " << j << std::endl;
                std::cerr << "  indices[" << (j-1) << "] = " << csr.indices[j-1] << std::endl;
                std::cerr << "  indices[" << j << "] = " << csr.indices[j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Test that CSC indices are sorted within each column
template<typename T>
bool test_csc_indices_sorted(const SpMatrixCompressed<T>& csc) {
    if (csc.direction != CompressionDirection::COLUMN) {
        std::cerr << "ERROR: Expected COLUMN direction for CSC" << std::endl;
        return false;
    }
    for (int i = 0; i < csc.num_cols; i++) {
        int start = csc.indptr[i];
        int end = csc.indptr[i+1];
        for (int j = start + 1; j < end; j++) {
            if (csc.indices[j] < csc.indices[j-1]) {
                std::cerr << "ERROR: CSC indices not sorted in column " << i 
                          << " at position " << j << std::endl;
                std::cerr << "  indices[" << (j-1) << "] = " << csc.indices[j-1] << std::endl;
                std::cerr << "  indices[" << j << "] = " << csc.indices[j] << std::endl;
                return false;
            }
        }
    }
    return true;
}

// Test CSR format
template<typename T>
bool test_csr(const SpMatrixCompressed<T>& csr, const std::map<std::pair<int, int>, T>& expected_entries) {
    std::cout << "Testing CSR format..." << std::endl;
    
    if (csr.direction != CompressionDirection::ROW) {
        std::cerr << "ERROR: Expected ROW direction for CSR" << std::endl;
        return false;
    }
    
    // Check dimensions
    if (csr.num_rows <= 0 || csr.num_cols <= 0) {
        std::cerr << "ERROR: Invalid dimensions" << std::endl;
        return false;
    }
    
    // Check indptr size
    if (csr.indptr.size() != static_cast<size_t>(csr.num_rows + 1)) {
        std::cerr << "ERROR: indptr size mismatch. Expected " << (csr.num_rows + 1) 
                  << ", got " << csr.indptr.size() << std::endl;
        return false;
    }
    
    // Check that indptr is non-decreasing
    for (int i = 0; i < csr.num_rows; i++) {
        if (csr.indptr[i] > csr.indptr[i+1]) {
            std::cerr << "ERROR: indptr is not non-decreasing at row " << i << std::endl;
            return false;
        }
    }
    
    // Check that last indptr entry equals num_nonzeros
    if (csr.indptr[csr.num_rows] != csr.num_nonzeros) {
        std::cerr << "ERROR: indptr[" << csr.num_rows << "] = " << csr.indptr[csr.num_rows]
                  << ", expected " << csr.num_nonzeros << std::endl;
        return false;
    }
    
    // Build a map from CSR data
    std::map<std::pair<int, int>, T> csr_entries;
    for (int i = 0; i < csr.num_rows; i++) {
        for (int j = csr.indptr[i]; j < csr.indptr[i+1]; j++) {
            int col = csr.indices[j];
            T val = csr.val[j];
            csr_entries[{i, col}] = val;
        }
    }
    
    // Compare with expected entries
    if (csr_entries.size() != expected_entries.size()) {
        std::cerr << "ERROR: Number of entries mismatch. Expected " << expected_entries.size()
                  << ", got " << csr_entries.size() << std::endl;
        return false;
    }
    
    // Check all expected entries are present with correct values
    for (const auto& entry : expected_entries) {
        auto it = csr_entries.find(entry.first);
        if (it == csr_entries.end()) {
            std::cerr << "ERROR: Missing entry at (" << entry.first.first << ", " 
                      << entry.first.second << ")" << std::endl;
            return false;
        }
        if (!approx_equal(it->second, entry.second)) {
            std::cerr << "ERROR: Value mismatch at (" << entry.first.first << ", " 
                      << entry.first.second << "). Expected " << entry.second 
                      << ", got " << it->second << std::endl;
            return false;
        }
    }
    
    // Check no extra entries
    for (const auto& entry : csr_entries) {
        if (expected_entries.find(entry.first) == expected_entries.end()) {
            std::cerr << "ERROR: Extra entry at (" << entry.first.first << ", " 
                      << entry.first.second << ") = " << entry.second << std::endl;
            return false;
        }
    }
    
    std::cout << "CSR test passed! (" << csr.num_rows << "x" << csr.num_cols 
              << ", " << csr.num_nonzeros << " nonzeros)" << std::endl;
    return true;
}

// Test CSC format
template<typename T>
bool test_csc(const SpMatrixCompressed<T>& csc, const std::map<std::pair<int, int>, T>& expected_entries) {
    std::cout << "Testing CSC format..." << std::endl;
    
    if (csc.direction != CompressionDirection::COLUMN) {
        std::cerr << "ERROR: Expected COLUMN direction for CSC" << std::endl;
        return false;
    }
    
    // Check dimensions
    if (csc.num_rows <= 0 || csc.num_cols <= 0) {
        std::cerr << "ERROR: Invalid dimensions" << std::endl;
        return false;
    }
    
    // Check indptr size
    if (csc.indptr.size() != static_cast<size_t>(csc.num_cols + 1)) {
        std::cerr << "ERROR: indptr size mismatch. Expected " << (csc.num_cols + 1) 
                  << ", got " << csc.indptr.size() << std::endl;
        return false;
    }
    
    // Check that indptr is non-decreasing
    for (int i = 0; i < csc.num_cols; i++) {
        if (csc.indptr[i] > csc.indptr[i+1]) {
            std::cerr << "ERROR: indptr is not non-decreasing at col " << i << std::endl;
            return false;
        }
    }
    
    // Check that last indptr entry equals num_nonzeros
    if (csc.indptr[csc.num_cols] != csc.num_nonzeros) {
        std::cerr << "ERROR: indptr[" << csc.num_cols << "] = " << csc.indptr[csc.num_cols]
                  << ", expected " << csc.num_nonzeros << std::endl;
        return false;
    }
    
    // Build a map from CSC data
    std::map<std::pair<int, int>, T> csc_entries;
    for (int i = 0; i < csc.num_cols; i++) {
        for (int j = csc.indptr[i]; j < csc.indptr[i+1]; j++) {
            int row = csc.indices[j];
            T val = csc.val[j];
            csc_entries[{row, i}] = val;
        }
    }
    
    // Compare with expected entries
    if (csc_entries.size() != expected_entries.size()) {
        std::cerr << "ERROR: Number of entries mismatch. Expected " << expected_entries.size()
                  << ", got " << csc_entries.size() << std::endl;
        return false;
    }
    
    // Check all expected entries are present with correct values
    for (const auto& entry : expected_entries) {
        auto it = csc_entries.find(entry.first);
        if (it == csc_entries.end()) {
            std::cerr << "ERROR: Missing entry at (" << entry.first.first << ", " 
                      << entry.first.second << ")" << std::endl;
            return false;
        }
        if (!approx_equal(it->second, entry.second)) {
            std::cerr << "ERROR: Value mismatch at (" << entry.first.first << ", " 
                      << entry.first.second << "). Expected " << entry.second 
                      << ", got " << it->second << std::endl;
            return false;
        }
    }
    
    // Check no extra entries
    for (const auto& entry : csc_entries) {
        if (expected_entries.find(entry.first) == expected_entries.end()) {
            std::cerr << "ERROR: Extra entry at (" << entry.first.first << ", " 
                      << entry.first.second << ") = " << entry.second << std::endl;
            return false;
        }
    }
    
    std::cout << "CSC test passed! (" << csc.num_rows << "x" << csc.num_cols 
              << ", " << csc.num_nonzeros << " nonzeros)" << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    std::string filename = "lp_wood1p/lp_wood1p.mtx";
    
    if (argc > 1) {
        filename = argv[1];
    }
    
    std::cout << "Reading raw Matrix Market file: " << filename << std::endl;
    auto expected_entries = read_raw_matrix_market<float>(filename);
    std::cout << "Read " << expected_entries.size() << " entries from raw file" << std::endl;
    
    std::cout << "\nReading matrix using read_matrix function..." << std::endl;
    SpMatrixCOO<float> matrix = read_matrix<float>(filename);
    std::cout << "Matrix dimensions: " << matrix.num_rows << "x" << matrix.num_cols 
              << ", " << matrix.num_nonzeros << " nonzeros" << std::endl;
    
    std::cout << "\nConverting to CSR format..." << std::endl;
    SpMatrixCompressed<float> csr = matrix.to_csr();
    
    std::cout << "\nConverting to CSC format..." << std::endl;
    SpMatrixCompressed<float> csc = matrix.to_csc();
    
    std::cout << "\n";
    bool csr_passed = test_csr(csr, expected_entries);
    std::cout << "\n";
    bool csc_passed = test_csc(csc, expected_entries);
    
    std::cout << "\nTesting CSR indices are sorted..." << std::endl;
    bool csr_sorted = test_csr_indices_sorted(csr);
    if (csr_sorted) {
        std::cout << "CSR indices sorting test passed!" << std::endl;
    } else {
        std::cerr << "CSR indices sorting test FAILED!" << std::endl;
    }
    
    std::cout << "\nTesting CSC indices are sorted..." << std::endl;
    bool csc_sorted = test_csc_indices_sorted(csc);
    if (csc_sorted) {
        std::cout << "CSC indices sorting test passed!" << std::endl;
    } else {
        std::cerr << "CSC indices sorting test FAILED!" << std::endl;
    }
    
    if (csr_passed && csc_passed && csr_sorted && csc_sorted) {
        std::cout << "\n=== All tests passed! ===" << std::endl;
        return 0;
    } else {
        std::cout << "\n=== Some tests failed! ===" << std::endl;
        return 1;
    }
}

