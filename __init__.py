from .memmap_class import MemmapManager
from .utils.filtering import filter_snps_memmap
from .utils.linkage import compute_ld_with_numba, approximate_pairwise_r_squared
from .utils.data_loading import load_bed_data

# Version information
__version__ = '0.0.1'