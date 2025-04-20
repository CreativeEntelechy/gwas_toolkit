from .utils.filter_snps import filter_snps_memmap
from .utils.linkage import (
    find_common_sorted, 
    compute_ld_for_snp, 
    compute_ld_csc_with_progress, 
    compute_ld_with_numba, 
    approximate_pairwise_r_squared, 
    approximate_r_squared_core,
    build_sparse_from_triplets, 
    exact_windowed_r_squared_numba
)
from .utils.data_loading import load_bed_data
from .memmap_class import MemmapManager

# Version information
__version__ = '0.0.1'