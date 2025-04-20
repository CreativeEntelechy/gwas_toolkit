import multiprocessing
from bed_reader import open_bed

def load_bed_data(manager, bed_path, output_name, batch_size=10000, cores_to_use = None, dtype='int8'):
    '''A wrapper function used to read in data and convert it to a memory-mapped file'''
    if cores_to_use is None:
        cores_to_use = max(1, multiprocessing.cpu_count() - 2)
    with open_bed(bed_path) as bed:
        manager.wrap_in_memmap(
        bed.read_sparse,  # the function itself
        output_name,    # identifier (used to name the memmap file)
        dtype=dtype,     # all named arguments pass through
        batch_size=batch_size,
        num_threads=cores_to_use
    )
    return None

