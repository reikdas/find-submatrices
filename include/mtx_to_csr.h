#ifndef MTX_TO_CSR_H
#define MTX_TO_CSR_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <tuple>

// Matrix Market symmetry types
enum class MatrixSymmetry {
    GENERAL,
    SYMMETRIC,
    SKEW_SYMMETRIC,
    HERMITIAN
};

// Compression direction for SpMatrixCompressed
enum class CompressionDirection {
    ROW,    // Row-compressed (CSR format)
    COLUMN  // Column-compressed (CSC format)
};

// Unified compressed sparse matrix format
// Uses indptr (index pointers) and indices to abstract CSR and CSC
template<typename T>
struct SpMatrixCompressed {
    std::vector<T> val;           // Non-zero values
    std::vector<int> indices;     // Column indices (for CSR) or row indices (for CSC)
    std::vector<int> indptr;      // Row pointers (for CSR) or column pointers (for CSC)
    int num_rows;
    int num_cols;
    int num_nonzeros;
    CompressionDirection direction;  // ROW for CSR, COLUMN for CSC
    
    SpMatrixCompressed() 
        : num_rows(0), num_cols(0), num_nonzeros(0), direction(CompressionDirection::ROW) {}
};

// Type aliases for backward compatibility
// CSR and CSC are now just SpMatrixCompressed with different directions
template<typename T>
using SpMatrixCSR = SpMatrixCompressed<T>;

template<typename T>
using SpMatrixCSC = SpMatrixCompressed<T>;

// COO (Coordinate) format - stores matrix as (row, col, val) triplets
template<typename T>
struct SpMatrixCOO {
    std::vector<int> row;
    std::vector<int> col;
    std::vector<T> val;
    int num_rows;
    int num_cols;
    int num_nonzeros;
    MatrixSymmetry symmetry;
    
    SpMatrixCOO() : num_rows(0), num_cols(0), num_nonzeros(0), symmetry(MatrixSymmetry::GENERAL) {}
    
    // Convert to CSR format (row-compressed)
    SpMatrixCompressed<T> to_csr() const {
        return convert_to_csr(*this);
    }
    
    // Convert to CSC format (column-compressed)
    SpMatrixCompressed<T> to_csc() const {
        return convert_to_csc(*this);
    }
};

// Helper function to trim whitespace
inline std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
}

// Parse Matrix Market banner line
inline bool parse_banner(const std::string& line, MatrixSymmetry& symmetry) {
    std::istringstream iss(line);
    std::string token;
    
    // Check for %%MatrixMarket
    if (!(iss >> token) || token != "%%MatrixMarket") {
        return false;
    }
    
    // Skip object, format, field
    iss >> token >> token >> token;
    
    // Read symmetry
    if (!(iss >> token)) {
        return false;
    }
    
    // Convert to lowercase for comparison
    std::transform(token.begin(), token.end(), token.begin(), ::tolower);
    
    if (token == "general") {
        symmetry = MatrixSymmetry::GENERAL;
    } else if (token == "symmetric") {
        symmetry = MatrixSymmetry::SYMMETRIC;
    } else if (token == "skew-symmetric") {
        symmetry = MatrixSymmetry::SKEW_SYMMETRIC;
    } else if (token == "hermitian") {
        symmetry = MatrixSymmetry::HERMITIAN;
    } else {
        return false;
    }
    
    return true;
}

// Read Matrix Market file
template<typename T>
SpMatrixCOO<T> read_matrix(const std::string& filename) {
    SpMatrixCOO<T> matrix;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return matrix;
    }
    
    std::string line;
    bool banner_found = false;
    
    // Read until we find the banner
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;
        
        if (line[0] == '%' && line[1] == '%') {
            if (!parse_banner(line, matrix.symmetry)) {
                std::cerr << "Error: Invalid Matrix Market banner" << std::endl;
                file.close();
                return matrix;
            }
            banner_found = true;
            break;
        }
    }
    
    if (!banner_found) {
        std::cerr << "Error: Matrix Market banner not found" << std::endl;
        file.close();
        return matrix;
    }
    
    // Skip comments and find dimensions line
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '%') continue;  // Skip comments
        
        // Parse dimensions: rows cols nonzeros
        std::istringstream iss(line);
        if (!(iss >> matrix.num_rows >> matrix.num_cols >> matrix.num_nonzeros)) {
            std::cerr << "Error: Cannot parse dimensions" << std::endl;
            file.close();
            return matrix;
        }
        break;
    }
    
    // Read matrix entries
    int row, col;
    T value;
    int entries_read = 0;
    
    while (std::getline(file, line) && entries_read < matrix.num_nonzeros) {
        line = trim(line);
        if (line.empty()) continue;
        if (line[0] == '%') continue;  // Skip comments
        
        std::istringstream iss(line);
        if (!(iss >> row >> col)) {
            continue;
        }
        
        // Matrix Market uses 1-based indexing, convert to 0-based
        row--;
        col--;
        
        // Read value (for real/complex matrices)
        if (!(iss >> value)) {
            value = T(1.0);  // Pattern matrices have implicit value 1.0
        }
        
        matrix.row.push_back(row);
        matrix.col.push_back(col);
        matrix.val.push_back(value);
        entries_read++;
        
        // Handle symmetric matrices: add lower triangle entry
        if (matrix.symmetry == MatrixSymmetry::SYMMETRIC && row != col) {
            matrix.row.push_back(col);
            matrix.col.push_back(row);
            matrix.val.push_back(value);
        }
        // Handle skew-symmetric matrices: add negative lower triangle entry
        else if (matrix.symmetry == MatrixSymmetry::SKEW_SYMMETRIC && row != col) {
            matrix.row.push_back(col);
            matrix.col.push_back(row);
            matrix.val.push_back(-value);
        }
        // Handle hermitian matrices: add conjugate lower triangle entry
        // For real matrices, this is the same as symmetric
        else if (matrix.symmetry == MatrixSymmetry::HERMITIAN && row != col) {
            matrix.row.push_back(col);
            matrix.col.push_back(row);
            matrix.val.push_back(value);  // For real matrices, conjugate is same
        }
    }
    
    // Update actual number of nonzeros after expansion
    matrix.num_nonzeros = matrix.val.size();
    
    file.close();
    return matrix;
}

// Internal helper function to convert SpMatrixCOO to compressed format
// Note: This function handles unsorted input from Matrix Market files by sorting entries.
// The resulting indices are guaranteed to be sorted within each compressed dimension.
template<typename T>
SpMatrixCompressed<T> convert_to_compressed(const SpMatrixCOO<T>& matrix, CompressionDirection direction) {
    SpMatrixCompressed<T> compressed;
    compressed.num_rows = matrix.num_rows;
    compressed.num_cols = matrix.num_cols;
    compressed.num_nonzeros = matrix.num_nonzeros;
    compressed.direction = direction;
    
    // Determine which dimension to compress and set indptr size
    int compressed_dim = (direction == CompressionDirection::ROW) ? matrix.num_rows : matrix.num_cols;
    
    if (matrix.num_nonzeros == 0) {
        compressed.indptr.resize(compressed_dim + 1, 0);
        return compressed;
    }
    
    // Create vector of (row, col, val) tuples for sorting
    // This handles unsorted input from Matrix Market files
    std::vector<std::tuple<int, int, T>> entries;
    entries.reserve(matrix.num_nonzeros);
    
    for (size_t i = 0; i < matrix.row.size(); i++) {
        entries.emplace_back(matrix.row[i], matrix.col[i], matrix.val[i]);
    }
    
    // Sort based on compression direction
    // CSR: sort by row, then by column (indices = col_idx sorted within each row)
    // CSC: sort by column, then by row (indices = row_idx sorted within each column)
    if (direction == CompressionDirection::ROW) {
        std::sort(entries.begin(), entries.end(), 
            [](const std::tuple<int, int, T>& a, const std::tuple<int, int, T>& b) {
                if (std::get<0>(a) != std::get<0>(b)) {
                    return std::get<0>(a) < std::get<0>(b);
                }
                return std::get<1>(a) < std::get<1>(b);
            });
    } else {
        std::sort(entries.begin(), entries.end(), 
            [](const std::tuple<int, int, T>& a, const std::tuple<int, int, T>& b) {
                if (std::get<1>(a) != std::get<1>(b)) {
                    return std::get<1>(a) < std::get<1>(b);
                }
                return std::get<0>(a) < std::get<0>(b);
            });
    }
    
    // Build compressed structure using indptr and indices
    compressed.indptr.resize(compressed_dim + 1, 0);
    compressed.indices.reserve(compressed.num_nonzeros);
    compressed.val.reserve(compressed.num_nonzeros);
    
    int current_dim = -1;
    for (const auto& entry : entries) {
        int row = std::get<0>(entry);
        int col = std::get<1>(entry);
        T val = std::get<2>(entry);
        
        // Determine which coordinate to use for indptr and which for indices
        int dim_coord = (direction == CompressionDirection::ROW) ? row : col;
        int index_coord = (direction == CompressionDirection::ROW) ? col : row;
        
        // Fill indptr for skipped dimensions
        while (current_dim < dim_coord) {
            current_dim++;
            compressed.indptr[current_dim] = compressed.val.size();
        }
        
        compressed.indices.push_back(index_coord);
        compressed.val.push_back(val);
    }
    
    // Fill remaining indptr entries
    while (current_dim < compressed_dim) {
        current_dim++;
        compressed.indptr[current_dim] = compressed.val.size();
    }
    
    compressed.num_nonzeros = compressed.val.size();
    return compressed;
}

// Convert SpMatrixCOO to CSR format (row-compressed)
// Note: This function handles unsorted input from Matrix Market files by sorting entries.
// The resulting indices are guaranteed to be sorted within each row.
template<typename T>
SpMatrixCompressed<T> convert_to_csr(const SpMatrixCOO<T>& matrix) {
    return convert_to_compressed(matrix, CompressionDirection::ROW);
}

// Convert SpMatrixCOO to CSC format (column-compressed)
// Note: This function handles unsorted input from Matrix Market files by sorting entries.
// The resulting indices are guaranteed to be sorted within each column.
template<typename T>
SpMatrixCompressed<T> convert_to_csc(const SpMatrixCOO<T>& matrix) {
    return convert_to_compressed(matrix, CompressionDirection::COLUMN);
}

// Print compressed matrix (works for both CSR and CSC)
template<typename T>
void print(const SpMatrixCompressed<T>& A) {
    if (A.direction == CompressionDirection::ROW) {
        // Print as CSR (row-compressed)
        for (int i = 0; i < A.num_rows; i++) {
            for (int j = A.indptr[i]; j < A.indptr[i+1]; j++) {
                std::cout << A.val[j] << " ";
            }
            std::cout << std::endl;
        }
    } else {
        // Print as CSC (column-compressed)
        for (int i = 0; i < A.num_cols; i++) {
            for (int j = A.indptr[i]; j < A.indptr[i+1]; j++) {
                std::cout << A.val[j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

#endif // MTX_TO_CSR_H

