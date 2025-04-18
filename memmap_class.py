import os
import numpy as np
import gc
from weakref import WeakSet
from datetime import datetime
from functools import wraps
import pyarrow as pa
from scipy.sparse import issparse, csc_matrix
from functools import wraps

class MemmapManager:
    """Complete memory-managed storage with Arrow-optimized sparse matrices and dense arrays
    Args:
        temp_dir: directory to store memmaps
        max_memory: the maximum memory utilization in GB
    """

    def __init__(self, temp_dir='temp', max_memory = 2):
        self.temp_dir = os.path.abspath(temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        self.storage = {
            'dense': {'temp': {}, 'final': {}},
            'sparse': {'temp': {}, 'final': {}}
        }
        self.metadata = {}
        self.open_handles = list()
        self.max_memory = max_memory * 1024**3
        self.chunk_factor = max_memory * 0.8

    # ======================== Core API Methods ========================
    
    def create_memmap(self, name, data=None, shape=None, dtype=np.int8, temp=False):
        """Create memory map with automatic format detection"""
        if data is not None:
            if issparse(data):
                return self._create_sparse(name, data, temp)
            return self._create_dense(name, data, temp)
        if shape and dtype:
            return self._create_sparse(name, csc_matrix(shape, dtype=dtype), temp)
        raise ValueError("Must provide data or shape+dtype")

    def get_memmap(self, name, mode='r', columns=None):
        """Metadata-validated retrieval"""
        if name not in self.metadata:
            raise KeyError(f"Memmap '{name}' not found in metadata registry")
        
        meta = self.metadata[name]
        if not os.path.exists(meta['path']):
            raise FileNotFoundError(f"Data file missing for {name}")
        
        # Column subset handling
        if columns is not None:
            return self._get_column_subset(name, columns, meta)
        
        # Full retrieval
        if meta['sparse']:
            return self._get_sparse(name, meta)
        return self._get_dense(name, meta, mode)
    
    def _get_column_subset(self, name, columns, meta):
        """Column-aware retrieval with metadata propagation and handle tracking"""
        if meta['sparse']:
            subset = self._get_sparse_columns(name, columns)
        else:
            subset = self._get_dense_columns(name, columns)
        temp_name = f"temp_{name}_subset"
        self.create_memmap(temp_name, subset, temp=True)
        # The following get_memmap will track the handle via _get_dense or _get_sparse
        return self.get_memmap(temp_name, mode='r')
    
    def _get_sparse_info(self, name):
        """Retrieve metadata for a sparse matrix entry"""
        if name in self.storage['sparse']['temp']:
            return self.storage['sparse']['temp'][name]
        elif name in self.storage['sparse']['final']:
            return self.storage['sparse']['final'][name]
        raise KeyError(f"Sparse matrix '{name}' not found in any storage")


    # ======================== Memory Enforcement ========================
    
    def _memory_guard(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if func.__name__ in ['_get_sparse', '_get_dense']:
                name = args[0]
                if 'sparse' in func.__name__:
                    info = self.storage['sparse']['final'].get(name) or {}
                    est_mem = info.get('shape', (0,))[0] * 4  # indptr size
                else:
                    meta = self.metadata.get(name)
                    if meta is None:
                        shape, dtype = (0, 0), np.int8
                    else:
                        shape = meta['shape']
                        dtype = meta['dtype']
                    est_mem = np.prod(shape) * np.dtype(dtype).itemsize
                if est_mem > self.max_memory * self.chunk_factor:
                    raise MemoryError(f"Operation would use {est_mem/1e9:.2f}GB")
            return func(self, *args, **kwargs)
        return wrapper


    # ======================== Sparse Matrix Handling ========================
    
    @_memory_guard
    def _create_sparse(self, name, data, temp):
        """Arrow-optimized CSC storage with Zstd compression"""
        csc_data = data.tocsc().astype(data.dtype)
        path = self._get_path(name, temp, sparse=True)
        
        schema = pa.schema([
            ('indptr', pa.uint32()),
            ('indices', pa.uint32()),
            ('data', pa.from_numpy_dtype(csc_data.dtype))
        ])

        with pa.OSFile(path, 'wb') as sink:
            opts = pa.ipc.IpcWriteOptions(compression='zstd')
            with pa.ipc.new_file(sink, schema, options=opts) as writer:
                # Initial batch with indptr
                writer.write(pa.RecordBatch.from_arrays([
                    pa.array(csc_data.indptr, type=pa.uint32()),
                    pa.array(np.zeros_like(csc_data.indptr), type=pa.uint32()),
                    pa.array(np.zeros(len(csc_data.indptr), dtype=csc_data.dtype))
                ], schema=schema))

                # Write data in memory-safe chunks
                chunk_size = self._calc_sparse_chunk(csc_data)
                for i in range(0, csc_data.nnz, chunk_size):
                    end = min(i + chunk_size, csc_data.nnz)
                    chunk_len = end - i
                    writer.write(pa.RecordBatch.from_arrays([
                        pa.array(np.zeros(chunk_len, dtype=np.uint32)),
                        pa.array(csc_data.indices[i:end], type=pa.uint32()),
                        pa.array(csc_data.data[i:end])
                    ], schema=schema))

        self._store_metadata(name, path, csc_data.shape, csc_data.dtype, temp, sparse=True)
        return None

# ======================== Sparse Matrix Handling ========================
    @_memory_guard
    def _get_sparse(self, name, meta):
        """Full sparse matrix retrieval using provided metadata"""
        with pa.memory_map(meta['path'], 'r') as mmap:
            reader = pa.ipc.open_file(mmap)
            indptr = reader.get_batch(0).column('indptr').to_numpy()
            indices, data = [], []
            
            # Process batches using metadata shape for memory limits
            chunk_size = self._calc_sparse_chunk(meta['shape'])
            for i in range(1, reader.num_record_batches):
                batch = reader.get_batch(i)
                indices.append(batch.column('indices').to_numpy())
                data.append(batch.column('data').to_numpy())
                
                # Memory-safe batch processing
                if len(indices) * indices[0].itemsize > self.max_memory * self.chunk_factor:
                    gc.collect()
            
            return csc_matrix(
                (np.concatenate(data) if data else [],
                np.concatenate(indices) if indices else [],
                indptr),
                shape=meta['shape']
            )

    def _get_sparse_subset(self, name, rows, columns):
        """Column/row subsetting with chunked processing"""
        col_subset = self._get_sparse_columns(name, columns) if columns else self._get_sparse(name)
        if rows is not None:
            return col_subset[rows, :].tocsc()
        return col_subset

    def _get_sparse_columns(self, name, columns):
        """Memory-safe column retrieval"""
        info = self._get_sparse_info(name)
        cols = np.asarray(columns)
        
        with pa.memory_map(info['path'], 'r') as mmap:
            reader = pa.ipc.open_file(mmap)
            indptr = reader.get_batch(0).column('indptr').to_numpy()
            
            starts = indptr[cols]
            ends = indptr[cols+1]
            min_idx = starts.min() if len(starts) > 0 else 0
            max_idx = ends.max() if len(ends) > 0 else 0
            
            indices, data = [], []
            batch_sizes = [reader.get_batch(i).num_rows for i in range(1, reader.num_record_batches)]
            cum_sizes = np.cumsum([0] + batch_sizes)
            
            first_batch = np.searchsorted(cum_sizes, min_idx, side='right') - 1
            last_batch = np.searchsorted(cum_sizes, max_idx, side='right') - 1
            
            for i in range(first_batch, last_batch+1):
                batch = reader.get_batch(i+1)
                batch_start = cum_sizes[i]
                slice_start = max(min_idx - batch_start, 0)
                slice_end = min(max_idx - batch_start, batch_sizes[i])
                
                indices.append(batch.column('indices').to_numpy()[slice_start:slice_end])
                data.append(batch.column('data').to_numpy()[slice_start:slice_end])

            return csc_matrix(
                (np.concatenate(data) if data else [],
                 (np.concatenate(indices) - min_idx) if indices else [],
                 indptr[cols] - indptr[cols[0]]),
                shape=(info['shape'][0], len(cols))
            )

    # ======================== Dense Matrix Handling ========================
    
    @_memory_guard
    def _create_dense(self, name, data, temp):
        """Chunked dense storage with memory limits and handle tracking"""
        path = self._get_path(name, temp, sparse=False)
        if len(data.shape) == 1:
            mmap = np.memmap(path, dtype=data.dtype, mode='w+', shape=data.shape)
            self.open_handles.append(mmap)
            mmap[:] = data[:]
        else:
            chunk_size = self._calc_dense_chunk(data)
            mmap = np.memmap(path, dtype=data.dtype, mode='w+', shape=data.shape)
            self.open_handles.append(mmap)
            for i in range(0, data.shape[1], chunk_size):
                end = min(i + chunk_size, data.shape[1])
                mmap[:, i:end] = data[:, i:end]
        mmap.flush()
        del mmap
        self._store_metadata(name, path, data.shape, data.dtype, temp, sparse=False)
        return None

    # ======================== Dense Matrix Handling ========================
    @_memory_guard
    def _get_dense(self, name, meta, mode):
        """Metadata-driven dense array access with handle tracking"""
        mmap = np.memmap(
            meta['path'],
            dtype=meta['dtype'],
            mode=mode,
            shape=meta['shape']
        )
        self.open_handles.append(mmap)
        return mmap


    def _get_dense_subset(self, name, rows, columns):
        """Chunked dense subsetting with handle tracking"""
        path = self._get_dense_path(name)
        shape, dtype = self.metadata[name]['shape'], self.metadata[name]['dtype']
        temp_name = f"temp_{name}_subset"
        temp_path = os.path.join(self.temp_dir, f"{temp_name}.dat")
        output_shape = (
            len(rows) if rows is not None else shape[0],
            len(columns) if columns is not None else shape[1]
        )
        output = np.memmap(temp_path, dtype=dtype, mode='w+', shape=output_shape)
        self.open_handles.append(output)
        chunk_size = self._calc_dense_chunk(output)
        src = np.memmap(path, dtype=dtype, mode='r', shape=shape)
        self.open_handles.append(src)
        if columns is not None:
            for i in range(0, len(columns), chunk_size):
                cols = columns[i:i+chunk_size]
                output[:, i:i+len(cols)] = src[rows, cols] if rows is not None else src[:, cols]
        else:
            output[:] = src[rows, :] if rows is not None else src[:]
        output.flush()
        del output, src
        self._store_metadata(temp_name, temp_path, output_shape, dtype, True, False)
        return self.get_memmap(temp_name, mode='r')


    # ======================== Utility Methods ========================
    
    def _get_path(self, name, temp, sparse):
        prefix = 'temp' if temp else 'final'
        ext = 'arrow' if sparse else 'dat'
        path = os.path.join(self.temp_dir, f"{prefix}_{name}.{ext}")
        if os.path.exists(path):
            try: os.remove(path)
            except: pass
        return path

    
    def _store_metadata(self, name, path, shape, dtype, temp, sparse):
        """Enhanced metadata storage with full context"""
        key_store = 'temp' if temp else 'final'
        
        metadata = {
            'path': path,
            'shape': shape,
            'dtype': dtype,
            'sparse': sparse,
            'temp': temp,
            'created': datetime.now().isoformat()
        }
        
        if sparse:
            self.storage['sparse'][key_store][name] = metadata
        else:
            self.storage['dense'][key_store][name] = metadata
        
        # Maintain separate metadata registry
        self.metadata[name] = metadata
    
    def _calc_sparse_chunk(self, data):
        """Safe chunk size calculation with division guard"""
        if isinstance(data, tuple):  # Shape-based estimate
            rows, cols = data
            if cols == 0:
                return 100  # Default safe chunk size when no columns
            bytes_per_col = rows * 4  # uint32 (4 bytes) per indptr entry
            return int(self.max_memory * self.chunk_factor // bytes_per_col)
        return int((self.max_memory * self.chunk_factor) // 
                (data.indices.itemsize + data.data.itemsize))

    
    def _calc_dense_chunk(self, data):
        """Calculate a safe chunk size for a dense array.
        If data is a shape tuple and has fewer than two dimensions, return a small default.
        """
        if isinstance(data, tuple):
            if len(data) < 2 or data[1] == 0:
                return 100
            bytes_per_col = data[0] * np.dtype(np.int8).itemsize
        else:
            if len(data.shape) < 2 or data.shape[1] == 0:
                return 100
            bytes_per_col = data.nbytes // data.shape[1]
        if bytes_per_col == 0:
            return 100
        return int((self.max_memory * self.chunk_factor) // bytes_per_col)



    # ======================== Management Methods ========================
    
    def delete_memmap(self, name, is_temp):
        """Unified deletion with explicit handle closing"""
        self.close_handles()
        gc.collect()
        key_store = 'temp' if is_temp else 'final'
        for fmt in ['sparse', 'dense']:
            if name in self.storage[fmt][key_store]:
                path = self.storage[fmt][key_store].pop(name)
                if fmt == 'sparse': path = path['path']
                try: os.remove(path)
                except: pass
                if name in self.metadata:
                    del self.metadata[name]
                return
        raise FileNotFoundError(f"Memmap '{name}' not found")

    def close_handles(self):
        """Close all open memory maps"""
        for handle in list(self.open_handles):
            try:
                if hasattr(handle, '_mmap') and handle._mmap is not None:
                    handle._mmap.close()
                elif hasattr(handle, 'close'):
                    handle.close()
                elif hasattr(handle, 'flush'):
                    handle.flush()
            except Exception:
                pass
        self.open_handles.clear()

    
    def __del__(self):
        self.close_handles()
        self.delete_all_temp()
    
    def delete_all_temp(self):
        """Clean temporary files"""
        for fmt in ['sparse', 'dense']:
            for name in list(self.storage[fmt]['temp'].keys()):
                self.delete_memmap(name, True)
    
    def delete_all(self):
        """Full cleanup"""
        self.delete_all_temp()
        for fmt in ['sparse', 'dense']:
            for name in list(self.storage[fmt]['final'].keys()):
                self.delete_memmap(name, False)
    
    def list_files(self):
        """List all managed files"""
        return {
            'sparse': {
                'temp': list(self.storage['sparse']['temp'].keys()),
                'final': list(self.storage['sparse']['final'].keys())
            },
            'dense': {
                'temp': list(self.storage['dense']['temp'].keys()),
                'final': list(self.storage['dense']['final'].keys())
            }
        }
    
    def get_file_path(self, name):
        """Get path for either format"""
        for fmt in ['sparse', 'dense']:
            for store in ['temp', 'final']:
                if name in self.storage[fmt][store]:
                    if fmt == 'sparse':
                        return self.storage[fmt][store][name]['path']
                    return self.storage[fmt][store][name]
        raise FileNotFoundError(f"File '{name}' not found")

    # ======================== Modification Methods ========================
    
    def modify_batch(self, name, rows, cols, values, temp=False):
        """
        Efficient batch modification that avoids SparseEfficiencyWarning
        by using COO for construction and addition operations.
        
        Parameters:
        -----------
        name : str
            Name of the memmap to modify
        rows : array-like
            Row indices for modifications
        cols : array-like
            Column indices for modifications
        values : array-like
            Values to set at the specified indices
        temp : bool, optional
            Whether to store the result as a temporary memmap
        """
        from scipy.sparse import coo_matrix
        
        # Get existing memmap
        original = self.get_memmap(name)
        
        # Determine storage name
        result_name = f"temp_{name}_mod" if temp else name
        
        if issparse(original):
            # Create updates as COO matrix (efficient for construction)
            updates = coo_matrix(
                (values, (rows, cols)), 
                shape=original.shape,
                dtype=values.dtype
            )
            
            # Store updates in a temporary memmap
            update_name = f"temp_{name}_update"
            self._create_sparse(update_name, updates.tocsc(), temp=True)
            
            # Get the temporary matrix (handle tracked)
            temp_updates = self.get_memmap(update_name)
            
            # Add updates to original (avoiding in-place CSC modifications)
            updated = original + temp_updates
            
            # Store the result
            self._create_sparse(result_name, updated, temp=temp)
            
            # Clean up temporary matrix
            self.delete_memmap(update_name, True)
        else:
            # Dense arrays can be modified directly
            updated = original.copy()
            updated[rows, cols] = values
            self._create_dense(result_name, updated, temp=temp)
        
        # Preserve original metadata if not temp
        if not temp:
            self.metadata[name] = self.metadata[result_name]
        
        # Explicitly close handles
        self.close_handles()
        gc.collect()
        
        return None
    
    def _modify_sparse(self, name, rows, cols, values, temp):
        """
        Optimized sparse batch modification using COO format for efficient
        construction and avoiding in-place CSC matrix modification.
        
        Parameters:
        -----------
        name : str
            Name of the memmap to modify
        rows : array-like
            Row indices for modifications
        cols : array-like
            Column indices for modifications
        values : array-like
            Values to set at the specified indices
        temp : bool
            Whether to store the result as a temporary memmap
        """
        from scipy.sparse import coo_matrix
        
        # Get existing matrix
        original = self.get_memmap(name)
        
        # Create COO matrix for updates (efficient for construction)
        updates = coo_matrix((values, (rows, cols)), shape=original.shape)
        
        # Store updates in a temporary memmap
        update_name = f"temp_{name}_update"
        self._create_sparse(update_name, updates.tocsc(), temp=True)
        
        # Get the temporary matrix
        temp_updates = self.get_memmap(update_name)
        
        # Add updates to original
        updated = original + temp_updates
        
        # Store the result
        result_name = name if not temp else f"temp_{name}"
        self._create_sparse(result_name, updated, temp=temp)
        
        # Clean up temporary matrix
        self.delete_memmap(update_name, True)
        
        # Explicitly close handles
        self.close_handles()
        gc.collect()


    
    def _modify_dense(self, name, rows, cols, values, temp):
        """Dense batch modification"""
        memmap = self.get_memmap(name, 'r+')
        memmap[rows, cols] = values
        memmap.flush()
    
    def modify_element(self, name, row, col, value):
        """Single-element modification"""
        self.modify_batch(name, [row], [col], [value])
    
    def wrap_in_memmap(self, func, name, **kwargs):
        """Auto-convert to int8 and enforce chunked processing"""
        data = func(**kwargs).astype(np.int8)
        if issparse(data):
            return self._create_sparse(name, data.tocsc(), kwargs.get('temp', False))
        return self._create_dense(name, data, kwargs.get('temp', False))

    # ======================== Subset Handling ========================
    
    def fetch_data(self, name, rows=None, columns=None):
        """Memory-safe data retrieval with metadata inheritance and handle tracking"""
        original = self.get_memmap(name)
        original_meta = self.metadata[name]
        if issparse(original):
            subset = original[rows, :] if rows is not None else original
            subset = subset[:, columns] if columns is not None else subset
        else:
            subset = original[rows, columns] if rows is not None and columns is not None else original[rows]
        temp_name = f"temp_{name}_fetch"
        temp_path = os.path.join(self.temp_dir, f"{temp_name}.{'arrow' if issparse(subset) else 'dat'}")
        new_shape = subset.shape
        new_dtype = original_meta.get('dtype', np.int8)
        is_sparse = issparse(subset)
        self._store_metadata(temp_name, temp_path, new_shape, new_dtype, True, is_sparse)
        if is_sparse:
            self._create_sparse(temp_name, subset.tocsc(), temp=True)
        else:
            self._create_dense(temp_name, subset.astype(new_dtype), temp=True)
        # The following get_memmap will track the handle via _get_dense or _get_sparse
        return self.get_memmap(temp_name, mode='r')

    def create_sparse_builder(self, name, shape, dtype=np.float32, temp=False):
        """Initialize sparse matrix construction with Arrow streaming"""
        path = self._get_path(name, temp, sparse=True)
        schema = pa.schema([
            ('row', pa.uint32()),
            ('col', pa.uint32()),
            ('data', pa.from_numpy_dtype(dtype))
        ])
        
        self._store_metadata(name, path, shape, dtype, temp, sparse=True)
        return pa.OSFile(path, 'wb'), schema

    def append_sparse_batch(self, file_handle, schema, rows, cols, data):
        """Append a batch of sparse matrix entries"""
        batch = pa.RecordBatch.from_arrays([
            pa.array(rows, type=pa.uint32()),
            pa.array(cols, type=pa.uint32()),
            pa.array(data)
        ], schema=schema)
        
        writer = pa.ipc.new_file(file_handle, schema)
        writer.write(batch)
        writer.close()

    def finalize_sparse(self, name, shape, temp=False):
        """Convert streamed Arrow data to final CSC format"""
        path = self.get_file_path(name)
        
        # Build CSC matrix in memory-safe chunks
        chunk_size = self._calc_sparse_chunk(shape)
        indptr = np.zeros(shape[1]+1, dtype=np.uint32)
        indices = []
        data = []
        
        with pa.memory_map(path, 'r') as mmap:
            reader = pa.ipc.open_file(mmap)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                cols = batch.column('col').to_numpy()
                
                # Sort columns for CSC format
                order = np.argsort(cols)
                indices.append(batch.column('row').to_numpy()[order])
                data.append(batch.column('data').to_numpy()[order])
                
                # Update indptr in chunks
                unique_cols, counts = np.unique(cols, return_counts=True)
                indptr[unique_cols+1] += counts
                
                # Memory cleanup
                if len(indices) * indices[0].itemsize > self.max_memory * self.chunk_factor:
                    self._flush_sparse_chunks(name, indptr, indices, data, temp)
                    indices.clear()
                    data.clear()
                    gc.collect()
        
        # Final flush
        self._flush_sparse_chunks(name, indptr, indices, data, temp)
        return None

    def _flush_sparse_chunks(self, name, indptr, indices, data, temp):
        """Write accumulated sparse data to disk"""
        if not indices:
            return
            
        # Convert to CSC chunks
        indptr = np.cumsum(indptr, out=indptr)
        csc_data = csc_matrix(
            (np.concatenate(data), np.concatenate(indices), indptr),
            shape=self.metadata[name]['shape']
        )
        
        # Update storage
        if self.exists(name):
            existing = self.get_memmap(name)
            csc_data += existing
        self.create_memmap(name, csc_data, temp=temp)

    def extend_memmap(self, name, new_data, temp=False):
        """
        Extend the memmapped array stored under 'name' by appending new_data along axis 0.
        This method uses file-backed operations so that the entire array is never loaded into memory.
        """
        existing = self.get_memmap(name, mode='r+')
        self.open_handles.append(existing)
        if len(existing.shape) < 2:
            old_length = existing.shape[0]
            new_length = old_length + new_data.shape[0]
            new_arr = np.empty(new_length, dtype=existing.dtype)
            chunk_size = 1024  # default for 1D
            for i in range(0, old_length, chunk_size):
                end = min(i + chunk_size, old_length)
                new_arr[i:end] = existing[i:end]
            new_arr[old_length:] = new_data
        else:
            old_shape = existing.shape
            new_shape = (old_shape[0] + new_data.shape[0],) + old_shape[1:]
            new_arr = np.empty(new_shape, dtype=existing.dtype)
            chunk_size = self._calc_dense_chunk(existing)
            for i in range(0, old_shape[0], chunk_size):
                end = min(i + chunk_size, old_shape[0])
                new_arr[i:end] = existing[i:end]
            new_arr[old_shape[0]:] = new_data
        del existing
        self.close_handles()
        gc.collect()
        self.delete_memmap(name, temp)
        self.create_memmap(name, new_arr, temp=temp)
        return None

    def _apply_sparse_updates(manager, matrix_name, rows, cols, values):
        """
        Apply updates to sparse matrix efficiently using COO format
        """
        from scipy.sparse import coo_matrix
        
        # Get existing matrix and its shape
        existing = manager.get_memmap(matrix_name)
        shape = existing.shape
        
        # Convert lists to arrays
        rows = np.array(rows, dtype=np.uint32)
        cols = np.array(cols, dtype=np.uint32)
        values = np.array(values, dtype=np.float32)
        
        # Create COO matrix for updates (efficient for construction)
        updates = coo_matrix((values, (rows, cols)), shape=shape)
        
        # Create a temporary name for the update
        temp_name = f"{matrix_name}_update_temp"
        
        # Convert COO to CSC only once before storage
        manager.create_memmap(temp_name, updates.tocsc(), temp=True)
        
        # Add to existing matrix (avoiding in-place CSC modifications)
        # For pairs that already exist, the addition will update the values
        updated = existing + manager.get_memmap(temp_name)
        
        # Store the updated matrix
        manager.create_memmap(matrix_name, updated, temp=False)
        
        # Clean up temporary matrix
        manager.delete_memmap(temp_name, True)

    def _apply_batch_updates(self, matrix_name, rows, cols, values):
        """
        Apply batch updates to a sparse matrix efficiently using COO format
        without modifying the sparse matrix in-place.
        
        Parameters:
        -----------
        matrix_name : str
            Name of the matrix to update
        rows : list or array
            Row indices for updates
        cols : list or array
            Column indices for updates
        values : list or array
            Values to set at the specified indices
        """
        from scipy.sparse import coo_matrix
        
        # Get existing matrix
        existing = self.get_memmap(matrix_name)
        shape = existing.shape
        
        # Convert lists to arrays if necessary
        if isinstance(rows, list):
            rows = np.array(rows, dtype=np.uint32)
            cols = np.array(cols, dtype=np.uint32)
            values = np.array(values, dtype=np.float32)
        
        # Create COO matrix for updates (efficient for construction)
        updates = coo_matrix((values, (rows, cols)), shape=shape)
        
        # Store as temporary CSC
        temp_name = f"{matrix_name}_update_temp"
        self._create_sparse(temp_name, updates.tocsc(), temp=True)
        
        # Get the temporary matrix
        temp_matrix = self.get_memmap(temp_name)
        
        # Add to existing matrix
        result = existing + temp_matrix
        
        # Store the result
        result_name = f"{matrix_name}_result_temp"
        self._create_sparse(result_name, result, temp=True)
        
        # Replace the original matrix with the result
        result_matrix = self.get_memmap(result_name)
        self._create_sparse(matrix_name, result_matrix, temp=False)
        
        # Clean up temporary matrices
        self.delete_memmap(temp_name, True)
        self.delete_memmap(result_name, True)
        
        # Close handles and collect garbage
        self.close_handles()
        gc.collect()

    def _apply_sparse_updates(self, target_name, source_name):
        """
        Safely add source sparse matrix to target sparse matrix,
        with proper handle tracking and memory management.
        
        Parameters:
        -----------
        target_name : str
            Name of target matrix to update
        source_name : str
            Name of source matrix to add
        """
        # Get matrices (manager tracks handles)
        target = self.get_memmap(target_name)
        source = self.get_memmap(source_name)
        
        # Add matrices (avoiding in-place modification of CSC)
        result = target + source
        
        # Store result with a temporary name
        temp_name = f"{target_name}_update_temp"
        self._create_sparse(temp_name, result, temp=True)
        
        # Replace target with result
        temp_result = self.get_memmap(temp_name)
        self._create_sparse(target_name, temp_result, temp=False)
        
        # Clean up temporary matrix
        self.delete_memmap(temp_name, True)
        
        # Explicitly close handles and collect garbage
        self.close_handles()
        gc.collect()










