import numpy as np
import multiprocessing
from bed_reader import open_bed

def load_bed_data(manager, bed_file, output_name="bed_data", cores_to_use=None, 
                  batch_size=10000, dtype='int8'):
    """
    Load data from a PLINK BED file into memory-mapped sparse storage
    
    Args:
        manager: MemmapManager instance
        bed_file: Path to the .bed file
        output_name: Name for the stored data in the manager
        cores_to_use: Number of CPU cores to use (defaults to CPU count - 2)
        batch_size: Size of batches for processing
        dtype: Data type for storage
        
    Returns:
        Reference to the stored data
    """
    # Set cores if not specified
    if cores_to_use is None:
        cores_to_use = max(1, multiprocessing.cpu_count() - 2)
    
    # Open the bed file
    bed = open_bed(bed_file)
    
    # Create a function that will be called by wrap_in_memmap
    def read_sparse_data(batch_size=batch_size, num_threads=cores_to_use):
        sparse_data = bed.read_sparse(dtype=dtype, batch_size=batch_size, format='csc')
        # Convert to int8 with -127 for missing values
        sparse_data.data[np.isnan(sparse_data.data)] = -127
        return sparse_data
    
    # Store the data using the manager
    manager.wrap_in_memmap(
        read_sparse_data,
        output_name,
        dtype=dtype,
        batch_size=batch_size,
        num_threads=cores_to_use
    )
    
    return manager.get_memmap(output_name)
