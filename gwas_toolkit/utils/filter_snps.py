import numpy as np
import gc
from tqdm import tqdm

def filter_snps_memmap(manager, memmap_name, output_name, threshold=0.2, 
                       missing_value=-127, chunk_size=1000):
    """Filters SNPs based on -127 missingness
    Args:
        manager: the memmap manager class
        memmap_name: the name of the memmap for the data
        output_name: the desired name for the output memmap
        threshold: the threshold for missingness that is tolerated
        missing_value: the code for missing value
        chunk_size: the size of the chunk desired
    Returns:
        None
    """
    meta = manager.metadata[memmap_name]
    is_sparse = meta['sparse']
    n_cols = meta['shape'][1]
    n_rows = meta['shape'][0]
    valid_columns = []

    if is_sparse:
        chunk_size = manager._calc_sparse_chunk((n_rows, n_cols))
        
        with tqdm(total=n_cols, desc="Filtering sparse SNPs") as pbar:
            for i in range(0, n_cols, chunk_size):
                cols = slice(i, min(i + chunk_size, n_cols))
                chunk = manager.fetch_data(memmap_name, columns=cols)
                
                # Count -127 in each column's data
                missing_counts = np.zeros(chunk.shape[1], dtype=int)
                for col_idx in range(chunk.shape[1]):
                    start = chunk.indptr[col_idx]
                    end = chunk.indptr[col_idx+1]
                    missing_counts[col_idx] = np.sum(chunk.data[start:end] == missing_value)
                
                valid_local = np.where(missing_counts/n_rows < threshold)[0]
                valid_columns.extend(valid_local + i)
                pbar.update(chunk.shape[1])
                
    else:
        chunk_size = manager._calc_dense_chunk((n_rows, n_cols))
        
        with tqdm(total=n_cols, desc="Filtering dense SNPs") as pbar:
            for i in range(0, n_cols, chunk_size):
                chunk = manager.fetch_data(memmap_name, columns=slice(i, i+chunk_size))
                missing_frac = np.mean(chunk == missing_value, axis=0)
                valid_local = np.where(missing_frac < threshold)[0]
                valid_columns.extend(valid_local + i)
                pbar.update(chunk.shape[1])
                del chunk
                gc.collect()

    filtered = manager.fetch_data(memmap_name, columns=valid_columns)
    manager.create_memmap(output_name, filtered, temp=False)
    return None