"""
Script to download matrices from SuiteSparse and analyze their density patterns.
"""

import os
import shutil
import subprocess
from multiprocessing import Pool

import ssgetpy
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix
from ssgetpy import fetch, search


def check_condition1_test(csr_indptr, csr_indices, cols, window_size=2500) -> bool:
    rows = len(csr_indptr) - 1

    for i in range(rows):
        start = csr_indptr[i]
        end = csr_indptr[i + 1]
        row_indices = csr_indices[start:end] 

        # Two-pointer sliding window over nonzeros
        left = 0
        for right in range(len(row_indices)):
            # Shrink window if width exceeds window_size
            while row_indices[right] - row_indices[left] + 1 > window_size:
                left += 1

            nnz_in_window = right - left + 1
            density = nnz_in_window / window_size

            if density >= 0.5:
                return True

        # Handle case where window extends beyond last column
        if len(row_indices) > 0:
            nnz = len(row_indices)
            effective_window = min(window_size, cols - row_indices[0])
            if nnz / effective_window >= 0.5:
                return True

    return False

def check_condition2_test(csr) -> bool:
    rows, cols = csr.shape
    indptr = csr.indptr

    row_nnz = indptr[1:] - indptr[:-1]

    # Necessary condition: some row band must have enough nnz
    # Conservative: check any window of up to 100 rows
    MAX_ROWS = 100
    MIN_NNZ = 3500

    prefix = row_nnz.cumsum()

    for i in range(rows):
        j = min(rows, i + MAX_ROWS)
        nnz_band = prefix[j - 1] - (prefix[i - 1] if i > 0 else 0)
        if nnz_band >= MIN_NNZ:
            return True

    return False


def analyze_matrix(csr: csr_matrix, csc: csc_matrix) -> (bool, str):
    res1 =  check_condition1_test(csr.indptr, csr.indices, csc.indptr[-1])
    if res1:
        return True, "Condition 1: Row"

    res2 = check_condition1_test(csc.indptr, csc.indices, csr.indptr[-1])
    if res2:
        return True, "Condition 1: Column"

    return check_condition2_test(csr), "Condition 2"

def get_matrix_info(matrix_name):
    """Search for a matrix by name and return its info object, or None if not found uniquely."""
    found = search(name_or_id=matrix_name)
    filtered_found = [m for m in found if m.name == matrix_name]
    if len(filtered_found) != 1:
        print(f"Skipping {matrix_name}: found {len(filtered_found)} matches")
        return None
    return filtered_found[0]


def get_matrix_paths(matrix_info):
    """Extract paths for a matrix: tar_path, tar_dir, matrix_subdir, and matrix_path."""
    localpath_info = matrix_info.localpath()
    tar_path = localpath_info[0] if isinstance(localpath_info, tuple) else localpath_info
    tar_dir = os.path.dirname(tar_path)
    matrix_subdir = os.path.join(tar_dir, matrix_info.name)
    matrix_path = os.path.join(matrix_subdir, f"{matrix_info.name}.mtx")
    return tar_path, tar_dir, matrix_subdir, matrix_path


def cleanup_matrix_files(tar_path, matrix_subdir):
    """Delete the tar file and extracted directory for a matrix."""
    if os.path.exists(tar_path):
        os.remove(tar_path)
    if os.path.exists(matrix_subdir):
        shutil.rmtree(matrix_subdir)


def download_matrix(matrix_name: str) -> tuple[str, object]:
    """
    Download a matrix from SuiteSparse and return the path to the .mtx file and matrix_info.
    
    Args:
        matrix_name: Name of the matrix to download
    
    Returns:
        Tuple of (matrix_path, matrix_info) where matrix_path is the path to the .mtx file
        and matrix_info is the matrix info object. Returns (None, None) if download fails.
    """
    matrix_info = get_matrix_info(matrix_name)
    if matrix_info is None:
        return None, None
    
    # Download the matrix
    fetch(matrix_name)
    
    _, _, _, matrix_path = get_matrix_paths(matrix_info)
    
    if not os.path.exists(matrix_path):
        print(f"Error: Matrix file not found at {matrix_path}")
        return None, None
    
    return matrix_path, matrix_info


def process_single_matrix(matrix_info) -> (bool, str):
    """Process a single matrix and return analysis results."""
    _, _, _, matrix_path = get_matrix_paths(matrix_info)
    print(matrix_path)

    # Check if the matrix file exists
    if not os.path.exists(matrix_path):
        raise FileNotFoundError(f"Matrix file not found at {matrix_path}. Make sure fetch() was called to download the matrix.")

    # Load the matrix
    matrix = mmread(matrix_path)

    # Convert to CSR and CSC
    csr = csr_matrix(matrix)
    csc = csc_matrix(matrix)

    # Analyze the matrix
    return analyze_matrix(csr, csc)

def pre_filter():
    """Pre-filter matrices by searching SuiteSparse and analyzing them."""
    # Search for real and binary matrices separately, then combine
    real_matrices = ssgetpy.search(nzbounds=(80_000, 20_000_000), dtype='real', limit=100000)
    binary_matrices = ssgetpy.search(nzbounds=(80_000, 20_000_000), dtype='binary', limit=100000)
    # Combine results, using id as key to avoid duplicates
    matrices_dict = {m.id: m for m in real_matrices}
    matrices_dict.update({m.id: m for m in binary_matrices})
    matrices = list(matrices_dict.values())

    matrix_names = [m.name for m in matrices]

    def process_matrix_worker(matrix_name):
        # Download the matrix
        matrix_path, matrix_info = download_matrix(matrix_name)
        if matrix_info is None:
            return None
        
        result, condition = process_single_matrix(matrix_info)

        tar_path, _, matrix_subdir, _ = get_matrix_paths(matrix_info)
        cleanup_matrix_files(tar_path, matrix_subdir)

        return result

    with Pool(processes=24) as pool:
        pool.map(process_matrix_worker, matrix_names)

def find_blocks():
    """Find blocks in a predefined set of evaluation matrices."""
    eval_matrices = [
        "eris1176",
        "std1_Jac3",
        "lp_wood1p",
        "jendrec1",
        "lowThrust_5",
        "hangGlider_4",
        "brainpc2",
        "hangGlider_3",
        "lowThrust_7",
        "lowThrust_11",
        "lowThrust_3",
        "lowThrust_6",
        "lowThrust_12",
        "hangGlider_5",
        "bloweybl",
        "heart1",
        "TSOPF_FS_b9_c6",
        "Sieber",
        "case9",
        "c-30",
        "c-32",
        "freeFlyingRobot_10",
        "freeFlyingRobot_11",
        "freeFlyingRobot_12",
        "lowThrust_10",
        "lowThrust_13",
        "lowThrust_4",
        "lowThrust_8",
        "lowThrust_9",
        "lp_fit2p",
        "nd12k",
        "std1_Jac2",
        "vsp_c-30_data_data"
    ]
    
    for matrix_name in eval_matrices:
        print(f"Processing {matrix_name}")
        
        # Download the matrix
        matrix_path, matrix_info = download_matrix(matrix_name)
        if matrix_info is None:
            continue

        tar_path, _, matrix_subdir, _ = get_matrix_paths(matrix_info)
        subprocess.run(["./build/partition_matrix", matrix_path], check=True)
        
        cleanup_matrix_files(tar_path, matrix_subdir)
        print(f"Completed {matrix_name}")

if __name__ == "__main__":
    # pre_filter()
    find_blocks()
