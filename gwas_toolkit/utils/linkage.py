import numpy as np
import numba
import gc
from tqdm import tqdm
from scipy.sparse import coo_matrix, issparse
from sklearn.random_projection import GaussianRandomProjection

@numba.jit(nopython=True)
def find_common_sorted(a, b):
    """Numba-compatible sorted array intersection"""
    common = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            common.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return np.array(common)

@numba.jit(nopython=True)
def compute_ld_for_snp(i, indptr, indices, data, n_snps, window_size, threshold=0.2):
    """
    Compute LD for one SNP i against SNPs j (i+1 <= j < i+window_size) using low-level optimized calculations
    
    Returns lists (as Python lists) for row, col and r2 values.
    """
    rows_i = []
    cols_i = []
    values_i = []
    
    # Get data for SNP i
    i_start = indptr[i]
    i_end = indptr[i+1]
    # Slicing the raw arrays
    i_idx = indices[i_start:i_end]
    i_vals = data[i_start:i_end]
    
    # Manually filter valid genotypes (0,1,2) for SNP i:
    valid_i_list = []
    valid_i_vals_list = []
    for k in range(len(i_vals)):
        if i_vals[k] != -127:
            valid_i_list.append(i_idx[k])
            valid_i_vals_list.append(i_vals[k])
    i_idx_valid = np.array(valid_i_list)
    i_vals_valid = np.array(valid_i_vals_list)
    
    # Loop over SNP j in a local window
    for j in range(i+1, min(i+window_size, n_snps)):
        j_start = indptr[j]
        j_end = indptr[j+1]
        j_idx = indices[j_start:j_end]
        j_vals = data[j_start:j_end]
        
        # Filter valid genotypes for SNP j
        valid_j_list = []
        valid_j_vals_list = []
        for k in range(len(j_vals)):
            if j_vals[k] != -127:
                valid_j_list.append(j_idx[k])
                valid_j_vals_list.append(j_vals[k])
        j_idx_valid = np.array(valid_j_list)
        j_vals_valid = np.array(valid_j_vals_list)
        
        # Find common samples
        common = find_common_sorted(i_idx_valid, j_idx_valid)
        if len(common) < 10:
            continue
        
        # Extract genotypes for common samples
        # (Because arrays are sorted we can step through them in order)
        g_i = []
        g_j = []
        ii = 0
        ji = 0
        for t in range(len(common)):
            sample = common[t]
            while ii < len(i_idx_valid) and i_idx_valid[ii] < sample:
                ii += 1
            while ji < len(j_idx_valid) and j_idx_valid[ji] < sample:
                ji += 1
            if ii < len(i_idx_valid) and i_idx_valid[ii] == sample:
                g_i.append(i_vals_valid[ii])
                ii += 1
            if ji < len(j_idx_valid) and j_idx_valid[ji] == sample:
                g_j.append(j_vals_valid[ji])
                ji += 1
        
        count_i = len(g_i)
        if count_i == 0:
            continue

        # Calculate allele frequencies for SNP i
        sum_i = 0.0
        for x in range(len(g_i)):
            sum_i += g_i[x]
        p_i = sum_i / (2 * count_i)
        
        # Calculate allele frequencies for SNP j
        count_j = len(g_j)
        sum_j = 0.0
        for x in range(len(g_j)):
            sum_j += g_j[x]
        p_j = sum_j / (2 * count_j)
        
        # Skip monomorphic SNPs
        if (p_i == 0 or p_i == 1) or (p_j == 0 or p_j == 1):
            continue
        
        # Calculate covariance
        cov_sum = 0.0
        for k in range(len(g_i)):
            cov_sum += g_i[k] * g_j[k]
        cov = (cov_sum / count_i) - (2 * p_i) * (2 * p_j)
        
        # Calculate variances
        var_i = 2 * p_i * (1 - p_i)
        var_j = 2 * p_j * (1 - p_j)
        
        # Compute r²
        r2 = 0.0
        if (var_i * var_j) > 0:
            r2 = (cov * cov) / (var_i * var_j)
        if r2 > threshold:
            rows_i.append(i)
            cols_i.append(j)
            values_i.append(r2)
    
    return rows_i, cols_i, values_i

def compute_ld_csc_with_progress(indptr, indices, data, window_size=200, threshold=0.2):
    """
    Computes LD over the entire CSC matrix while updating a progress bar.
    
    This function loops over SNPs in Python (with tqdm) and calls the JIT-compiled
    compute_ld_for_snp for each SNP i.
    """
    n_snps = len(indptr) - 1
    total_rows = []
    total_cols = []
    total_values = []
    
    for i in tqdm(range(n_snps), desc="Computing LD for SNPs"):
        rows_i, cols_i, values_i = compute_ld_for_snp(i, indptr, indices, data, n_snps, window_size, threshold)
        # Append the partial results
        total_rows.extend(rows_i)
        total_cols.extend(cols_i)
        total_values.extend(values_i)
    
    return total_rows, total_cols, total_values

# ======================== Bridge Functions for MemmapManager Integration ========================

def compute_ld_with_numba(manager, input_name, output_name, window_size=200, threshold=0.2):
    """
    Compute LD matrix using Numba-accelerated functions and store result in MemmapManager.
    This function bridges between MemmapManager and Numba-optimized LD computation.
    """
    # Get data from manager
    data_csc = manager.get_memmap(input_name)
    
    if not issparse(data_csc):
        raise ValueError("Input must be a sparse CSC matrix for Numba-accelerated LD calculation")
    
    # Extract components needed for Numba function
    indptr = data_csc.indptr.astype(np.int64)
    indices = data_csc.indices.astype(np.int64)
    data = data_csc.data.astype(np.int8)
    
    # Call Numba-optimized function
    rows, cols, values = compute_ld_csc_with_progress(indptr, indices, data, window_size, threshold)
    
    # Create symmetric LD matrix
    ld_matrix = coo_matrix(
        (values + values, (rows + cols, cols + rows)),
        shape=(data_csc.shape[1], data_csc.shape[1]),
        dtype=np.float32
    ).tocsc()
    
    # Store results back to manager
    manager.create_memmap(output_name, ld_matrix, temp=False)
    
    return None

# ======================== Main LD Pipeline Functions ========================

def approximate_pairwise_r_squared(manager, memmap_name, output_name, chunk_size = 10000, 
                                   hash_size=64, approx_threshold=0.2,
                                   window_size=200):
    """
    Complete LD pipeline: never pre-allocates large arrays or matrices.
    Uses LSH projection for approximate screening followed by exact refinement.
    
    Note: chunk_size is now determined dynamically based on memory constraints
    """
    # Phase 1: Approximate r² using LSH projection with dynamic chunk sizing
    triplet_info = approximate_r_squared_core(
        manager, memmap_name, chunk_size, output_name+"_approx",
        hash_size, approx_threshold
    )
    
    # Phase 2: Build sparse matrix from triplets
    status = build_sparse_from_triplets(manager, output_name, triplet_info)
    if status is not None:
        print('Did not find any significant linkage... not dataframe will be created.')
        return None
    
    # Phase 3: Exact windowed refinement using Numba-optimized functions
    exact_windowed_r_squared_numba(
        manager, memmap_name, output_name, window_size, approx_threshold
    )
    
    # Safe cleanup of temporary files
    try:
        manager.delete_memmap(triplet_info["rows"], True)
        manager.delete_memmap(triplet_info["cols"], True)
        manager.delete_memmap(triplet_info["vals"], True)
    except FileNotFoundError:
        pass  # Safely handle case where files don't exist
    
    return None

def approximate_r_squared_core(manager, memmap_name, chunk_size, temp_out_name,
                               hash_size, threshold):
    """
    Phase 1: Vectorized and memory-efficient approximate r² computation
    Returns a dict with triplet memmap names and triplet_count.
    Exits early if no significant correlations are found.
    """
    meta = manager.metadata[memmap_name]
    n_samples, n_cols = meta['shape']
    projector = GaussianRandomProjection(n_components=hash_size, random_state=42)
    temp_name_rows = temp_out_name + "_rows"
    temp_name_cols = temp_out_name + "_cols"
    temp_name_vals = temp_out_name + "_vals"
    triplet_count = 0
    first_batch = True
    
    # Create a flag to track if we found any significant correlations
    found_significant = False

    for i in tqdm(range(0, n_cols, chunk_size), desc="Phase 1: Approximate r²"):
        i_end = min(i + chunk_size, n_cols)
        chunk = manager.fetch_data(memmap_name, columns=slice(i, i_end))
        if issparse(chunk):
            chunk = chunk.astype(np.float32).tocsr()
            chunk.data[chunk.data == -127] = 0
            projector.fit(chunk)
            projected = chunk.dot(projector.components_.T)
        else:
            chunk = chunk.astype(np.float32)
            chunk[chunk == -127] = 0
            projected = projector.fit_transform(chunk)

        # Vectorized r² calculation
        X = projected
        Xm = np.mean(X, axis=0, keepdims=True)
        X_centered = X - Xm
        # Fix: Use X.shape[0] instead of X.shape for denominator
        cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
        std_X = np.std(X, axis=0, ddof=1, keepdims=True)
        denom = std_X.T @ std_X
        # Avoid division by zero
        corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom != 0)
        # Square to get r²
        r2 = np.nan_to_num(corr ** 2, nan=0.0)

        # Only keep upper triangular values that exceed threshold
        mask = np.triu(r2, k=1) > threshold
        sub_rows, sub_cols = np.nonzero(mask)
        
        if sub_rows.size > 0:
            found_significant = True
            global_rows = (sub_rows + i).astype(np.uint32)
            global_cols = (sub_cols + i).astype(np.uint32)
            values = r2[sub_rows, sub_cols].astype(np.float32)
            
            if first_batch:
                manager.create_memmap(temp_name_rows, global_rows, temp=True)
                manager.create_memmap(temp_name_cols, global_cols, temp=True)
                manager.create_memmap(temp_name_vals, values, temp=True)
                first_batch = False
            else:
                manager.extend_memmap(temp_name_rows, global_rows, temp=True)
                manager.extend_memmap(temp_name_cols, global_cols, temp=True)
                manager.extend_memmap(temp_name_vals, values, temp=True)
            
            triplet_count += global_rows.size
        
        # Clean up to free memory
        del chunk, projected, X, X_centered, cov, std_X, denom, corr, r2, mask
        manager.close_handles()
        gc.collect()
    
    # Check if we found any significant correlations
    if not found_significant:
        print("No significant correlations found in the approximate phase. Ending computation.")
        return {
            "rows": None,
            "cols": None,
            "vals": None,
            "triplet_count": 0,
            "n_cols": n_cols,
            "empty": True
        }
        
    return {
        "rows": temp_name_rows,
        "cols": temp_name_cols,
        "vals": temp_name_vals,
        "triplet_count": triplet_count,
        "n_cols": n_cols,
        "empty": False
    }

def build_sparse_from_triplets(manager, output_name, triplet_info):
    """
    Phase 2: Efficiently build sparse matrix from triplets using COO format,
    converting to CSC only at the final step to avoid efficiency warnings.
    Exits early if no significant correlations were found.
    """
    # Check if the data is empty
    if triplet_info.get("empty", False) or triplet_info["triplet_count"] == 0:
        return 2
        
    n_cols = triplet_info["n_cols"]
    triplet_count = triplet_info["triplet_count"]
    temp_name_rows = triplet_info["rows"]
    temp_name_cols = triplet_info["cols"]
    temp_name_vals = triplet_info["vals"]

    # Calculate safe batch size based on memory constraints
    batch_size = min(1000000, manager._calc_sparse_chunk((n_cols, n_cols)))
    
    # Process in batches to avoid memory issues
    for start in tqdm(range(0, triplet_count, batch_size), desc="Phase 2: Build sparse dataframes from approximaton pointers"):
        end = min(start + batch_size, triplet_count)
        
        # Fetch data for this batch
        rows = manager.get_memmap(temp_name_rows)[start:end]
        cols = manager.get_memmap(temp_name_cols)[start:end]
        vals = manager.get_memmap(temp_name_vals)[start:end]
        
        # Build batch as COO (efficient for construction)
        batch_coo = coo_matrix(
            (vals, (rows, cols)), 
            shape=(n_cols, n_cols), 
            dtype=np.float32
        )
        
        # Convert to CSC only once per batch
        batch_csc = batch_coo.tocsc()
        
        # Create or update the output matrix
        if start == 0:
            # First batch: create new matrix
            manager.create_memmap(output_name, batch_csc, temp=False)
        else:
            # Subsequent batches: update existing matrix using manager methods
            batch_name = f"{output_name}_batch_temp"
            manager.create_memmap(batch_name, batch_csc, temp=True)
            
            # Get existing matrix and new batch
            existing = manager.get_memmap(output_name)
            batch = manager.get_memmap(batch_name)
            
            # Add matrices (avoiding in-place CSC modifications)
            result = existing + batch
            
            # Store temporary result
            result_name = f"{output_name}_result_temp"
            manager.create_memmap(result_name, result, temp=True)
            
            # Replace original with result
            result_matrix = manager.get_memmap(result_name)
            manager.create_memmap(output_name, result_matrix, temp=False)
            
            # Clean up temporary matrices
            manager.delete_memmap(batch_name, True)
            manager.delete_memmap(result_name, True)
        
        # Clean up memory
        del rows, cols, vals, batch_coo, batch_csc
        manager.close_handles()
        gc.collect()



def exact_windowed_r_squared_numba(manager, input_name, output_name, window_size, threshold):
    """
    Phase 3: Exact windowed refinement using Numba-accelerated LD computation.
    This function processes significant columns from the approximate matrix
    and refines them using the Numba-optimized LD calculation.
    Exits early if no significant columns are found.
    """
    # Get data from manager
    data_csc = manager.get_memmap(input_name)
    approx_ld = manager.get_memmap(output_name)
    
    # Get significant columns that need refinement
    sig_cols = np.unique(approx_ld.nonzero()[1])
    n_cols = approx_ld.shape[1]
    
    # If no significant columns, exit early
    if len(sig_cols) == 0:
        print("No significant columns found for refinement. Skipping exact calculation.")
        return None
    
    # Extract components needed for Numba function
    indptr = data_csc.indptr.astype(np.int64)
    indices = data_csc.indices.astype(np.int64)
    data = data_csc.data.astype(np.int8)
    
    # Process in batches to avoid memory issues
    batch_size = 100  # Process this many significant columns at once
    
    for batch_start in tqdm(range(0, len(sig_cols), batch_size), 
                           desc="Phase 3: Processing windows for exact LD refinement"):
        batch_end = min(batch_start + batch_size, len(sig_cols))
        batch_sig_cols = sig_cols[batch_start:batch_end]
        
        total_rows, total_cols, total_values = [], [], []
        
        for col in batch_sig_cols:
            # Compute LD for this column and its window
            max_window = min(window_size, n_cols - col)
            rows_i, cols_i, values_i = compute_ld_for_snp(
                col, indptr, indices, data, n_cols, max_window, threshold
            )
            
            # Only keep values that improve on the approximate matrix
            for j in range(len(rows_i)):
                i, j_col = rows_i[j], cols_i[j]
                if values_i[j] > approx_ld[i, j_col]:
                    total_rows.append(i)
                    total_cols.append(j_col)
                    total_values.append(values_i[j])
        
        # Apply updates if any found
        if total_rows:

            # Convert lists to arrays
            rows_arr = np.array(total_rows, dtype=np.uint32)
            cols_arr = np.array(total_cols, dtype=np.uint32)
            vals_arr = np.array(total_values, dtype=np.float32)
            
            # Create COO matrix for updates (efficient for construction)
            updates = coo_matrix((vals_arr, (rows_arr, cols_arr)), shape=approx_ld.shape)
            
            # Update using manager methods
            temp_update_name = f"{output_name}_update_temp"
            manager.create_memmap(temp_update_name, updates.tocsc(), temp=True)
            
            # Get existing matrix and update
            existing = manager.get_memmap(output_name)
            update_matrix = manager.get_memmap(temp_update_name)
            
            # Add matrices (avoiding in-place CSC modifications)
            result = existing + update_matrix
            
            # Store result
            temp_result_name = f"{output_name}_result_temp"
            manager.create_memmap(temp_result_name, result, temp=True)
            
            # Replace original with result
            result_matrix = manager.get_memmap(temp_result_name)
            manager.create_memmap(output_name, result_matrix, temp=False)
            
            # Clean up temporary matrices
            manager.delete_memmap(temp_update_name, True)
            manager.delete_memmap(temp_result_name, True)
        
        # Clean up memory
        manager.close_handles()
        gc.collect()
