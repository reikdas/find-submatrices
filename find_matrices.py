"""
Script to download matrices from SuiteSparse and analyze their density patterns.
"""

import os
import shutil
import subprocess
import zlib
import multiprocessing
from multiprocessing import Pool

# Use 'spawn' instead of 'fork' to avoid SQLite threading issues
# (ssgetpy uses SQLite, which cannot share connections across forked processes)
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Already set

import ssgetpy
from scipy.io import mmread
from scipy.sparse import csc_matrix, csr_matrix
from ssgetpy import fetch, search




def check_2d_block_potential(csr, csc, min_dim=50, min_density=0.5, row_span_multiplier=2) -> bool:
    """
    Stricter pre-filter that checks for 2D block potential.
    
    For a valid block of min_dim x min_dim with density min_density:
    - Need min_dim rows, each with some minimum nnz
    - Need min_dim cols, each with some minimum nnz
    - These rows/cols must be clustered (not scattered across the matrix)
    - The cluster must have enough total nnz to form a valid block
    
    Uses relaxed thresholds to avoid false negatives while catching false positives.
    """
    rows, cols = csr.shape
    csr_indptr = csr.indptr
    csc_indptr = csc.indptr
    
    # Minimum nnz required for a valid block
    min_cluster_nnz = int(min_dim * min_dim * min_density)  # 1250 for 50x50 @ 0.5
    
    # Per-row/col threshold: 0.6x the average needed
    # For 50x50 block with 50% density, each row needs ~25 nnz on avg
    # We use 15 as threshold (still safe since 25 > 15)
    min_nnz_threshold = int(min_dim * min_density * 0.6)
    
    # Find rows with enough nonzeros, storing (row_idx, nnz)
    qualifying_rows = []
    for i in range(rows):
        row_nnz = csr_indptr[i + 1] - csr_indptr[i]
        if row_nnz >= min_nnz_threshold:
            qualifying_rows.append((i, row_nnz))
    
    if len(qualifying_rows) < min_dim:
        return False
    
    # Find columns with enough nonzeros, storing (col_idx, nnz)
    qualifying_cols = []
    for j in range(cols):
        col_nnz = csc_indptr[j + 1] - csc_indptr[j]
        if col_nnz >= min_nnz_threshold:
            qualifying_cols.append((j, col_nnz))
    
    if len(qualifying_cols) < min_dim:
        return False
    
    # Check if there's a cluster of min_dim qualifying rows within a reasonable span
    # AND the cluster has enough total nnz
    max_span = min_dim * row_span_multiplier
    
    # Use sliding window sum for efficiency
    found_row_cluster = False
    if len(qualifying_rows) >= min_dim:
        window_nnz = sum(r[1] for r in qualifying_rows[:min_dim])
        for i in range(len(qualifying_rows) - min_dim + 1):
            if i > 0:
                window_nnz = window_nnz - qualifying_rows[i - 1][1] + qualifying_rows[i + min_dim - 1][1]
            # Check both span AND total nnz
            if qualifying_rows[i + min_dim - 1][0] - qualifying_rows[i][0] < max_span:
                if window_nnz >= min_cluster_nnz:
                    found_row_cluster = True
                    break
    
    if not found_row_cluster:
        return False
    
    # Check if there's a cluster of min_dim qualifying columns with enough total nnz
    found_col_cluster = False
    if len(qualifying_cols) >= min_dim:
        window_nnz = sum(c[1] for c in qualifying_cols[:min_dim])
        for j in range(len(qualifying_cols) - min_dim + 1):
            if j > 0:
                window_nnz = window_nnz - qualifying_cols[j - 1][1] + qualifying_cols[j + min_dim - 1][1]
            # Check both span AND total nnz
            if qualifying_cols[j + min_dim - 1][0] - qualifying_cols[j][0] < max_span:
                if window_nnz >= min_cluster_nnz:
                    found_col_cluster = True
                    break
    
    return found_col_cluster




def analyze_matrix(csr: csr_matrix, csc: csc_matrix) -> (bool, str):
    # Check for 2D block potential - requires dense structure in BOTH dimensions
    # This is sufficient to avoid false negatives while filtering out matrices
    # that cannot possibly contain a valid block
    result = check_2d_block_potential(csr, csc, min_dim=50, min_density=0.5)
    return result, "2D block potential check"

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

def process_matrix_worker(matrix_name):
    """Worker function for processing a single matrix in parallel."""
    try:
        matrix_info = get_matrix_info(matrix_name)
        if matrix_info is None:
            return (matrix_name, None)
        
        # Download the matrix before processing (fetch expects name/ID, not object)
        fetch(matrix_name)
        result, condition = process_single_matrix(matrix_info)

        tar_path, _, matrix_subdir, _ = get_matrix_paths(matrix_info)
        cleanup_matrix_files(tar_path, matrix_subdir)

        return (matrix_name, result)
    except (zlib.error, Exception) as e:
        # Handle decompression errors and other exceptions
        print(f"Error processing {matrix_name}: {e}")
        # Clean up any partial files if matrix_info was created
        try:
            if 'matrix_info' in locals() and matrix_info is not None:
                tar_path, _, matrix_subdir, _ = get_matrix_paths(matrix_info)
                cleanup_matrix_files(tar_path, matrix_subdir)
        except:
            pass
        return (matrix_name, False)

def pre_filter():
    """Pre-filter matrices by searching SuiteSparse and analyzing them."""
    # Search for real and binary matrices separately, then combine
    real_matrices = ssgetpy.search(nzbounds=(80_000, 20_000_000), dtype='real', limit=100000)
    binary_matrices = ssgetpy.search(nzbounds=(80_000, 20_000_000), dtype='binary', limit=100000)
    # Combine results, using id as key to avoid duplicates
    matrices_dict = {m.id: m for m in real_matrices}
    matrices_dict.update({m.id: m for m in binary_matrices})
    matrices = list(matrices_dict.values())

    ignore_matrices = [
    ]

    # Also ignore all later_eval matrices
    ignore_matrices = ignore_matrices

    matrix_names = [m.name for m in matrices if m.name not in ignore_matrices]
    
    with Pool(processes=24) as pool:
        with open('matrices.txt', 'w') as f:
            # Use imap_unordered to process results as they complete
            for matrix_name, result in pool.imap_unordered(process_matrix_worker, matrix_names):
                if result is not None:
                    f.write(f"{matrix_name}: {result}\n")
                    f.flush()
                else:
                    # Handle case where matrix_info was None
                    f.write(f"{matrix_name}: False\n")
                    f.flush()

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
