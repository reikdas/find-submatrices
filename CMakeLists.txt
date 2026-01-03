cmake_minimum_required(VERSION 3.10)
project(PartitionMatrix VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type to Release by default if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
endif()

# Find OpenMP
# On macOS with Homebrew, help CMake find libomp
if(APPLE)
    # Try to find libomp via Homebrew prefix
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(HOMEBREW_LIBOMP_PREFIX)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        set(OpenMP_CXX_INCLUDE_DIR "${HOMEBREW_LIBOMP_PREFIX}/include")
        message(STATUS "Found libomp at: ${HOMEBREW_LIBOMP_PREFIX}")
    endif()
endif()

find_package(OpenMP REQUIRED)

# Main executable
add_executable(partition_matrix
    main.cpp
)

# Add include directory for partition_matrix target
target_include_directories(partition_matrix PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link OpenMP to executables
target_link_libraries(partition_matrix PUBLIC OpenMP::OpenMP_CXX)

# Test executable
add_executable(test_csr_csc
    tests/test_csr_csc.cpp
)

# Add include directory for test_csr_csc target
target_include_directories(test_csr_csc PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link OpenMP to test executable
target_link_libraries(test_csr_csc PUBLIC OpenMP::OpenMP_CXX)

# Enable testing
enable_testing()

# Add test
add_test(NAME CSR_CSC_Test COMMAND test_csr_csc)

# Optional: Add custom target to run tests with verbose output
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS test_csr_csc
    COMMENT "Running all tests"
)

# Print configuration summary
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

