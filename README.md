# gwas_toolkit
A Python package for efficient, memory-managed Genome-Wide Association Study (GWAS) analysis with optimized sparse matrix operations.

## Overview
GWAS Analysis provides tools for loading large-scale genetic data, filtering SNPs, and calculating linkage disequilibrium (LD) using disk-backed sparse matrices. Typically GWAS packages require a ton of compute resources, but this package helps scale that down by chunking analysis. It also prevents OOM errors by using numpy memmaps alongside the efficient storage capacities of Arrow. Since GWAS datasets are inherently sparse (as 0 represents homozygosity to the reference genome), we can store data in sparse data representations to economize disk space. This is particularly helpful, as large LD calculations can take up terabytes, or even petabytes, of disk space.

To build this tool I used publicly available GWAS dataset for rice, which can be found here: `https://snp-seek.irri.org/`

I downloaded the 1M SNP dataset and was able to calculate an approximate LD matrix in 10 minutes! The challenge with linkage disequilibrium is that it is inherently expensive. To calculate LD is an $O(n^2)$ operation, which means that it scales quadratically with the number of SNPs. This means that even the faster computer in the world is not going to be able to calculate large GWAS datasets. What I choose to do instead is use Locality-Sensitive Hashing (LSH) to estimate approximate r-squared values. We take only those that are significant and then we calculate the exact values within a sliding window. The most likely misclassifications will occur around the boundaries and therefore the second pass will ensure accuracy, calculating exact r-squared values for regions with high correlation.

## Installation
Pre-requisite: have Conda or some package manager installed. See here: `https://www.anaconda.com/download`

Clone the repository and install:

```bash
git clone https://github.com/ItsReces/gwas_toolkit.git
cd gwas_toolkit
conda env create -f environment.yml
conda activate gwas_toolkit
```

## Usage Example
```python
from gwas_analysis import MemmapManager
from gwas_analysis.utils.data_loading import load_bed_data
from gwas_analysis.utils.filtering import filter_snps_memmap
from gwas_analysis.utils.linkage import approximate_pairwise_r_squared
import multiprocessing

# Set number of cores (leave 2 free for system)
cores_to_use = max(1, multiprocessing.cpu_count() - 2)

# Initialize the memory manager (handles disk-backed storage)
memmap_manager = MemmapManager(temp_dir="gwas_temp", max_memory = 2) # max memory usage is in GBs

# Load genetic data from a PLINK BED file as a sparse matrix
bed_file = "path/to/your/data.bed"
sparse_data = load_bed_data(
    memmap_manager,
    bed_file,
    output_name="sparse_data",
    cores_to_use=cores_to_use,
    batch_size=10000,
    dtype='int8'
)

# Filter SNPs based on missingness threshold (e.g., remove SNPs with >10% missing)
filter_snps_memmap(
    memmap_manager,
    "sparse_data",
    "filtered_data",
    threshold=0.1,
    missing_value=-127
)

# Calculate linkage disequilibrium (LD) using an approximate method
approximate_pairwise_r_squared(
    memmap_manager,
    "filtered_data",
    "ld_matrix",
    hash_size=64, # Size of random projection for initial screening
    approx_threshold=0.2, # Minimum r² value to keep
    window_size=200 # Max SNP distance for LD calculation
)

# Retrieve the resulting LD matrix
ld_matrix = memmap_manager.get_memmap("ld_matrix")
print(f"LD matrix shape: {ld_matrix.shape}")
print(f"Number of significant LD pairs: {ld_matrix.nnz}")

# Clean up temporary files when done
memmap_manager.delete_all_temp()
```

# Future Development Notes
In the future I plan to build out more exciting features. Among these are the following:
- Imputation methods
- Haplotype Analysis
- Calculating Polygenic Scores
- Causal inference
- Machine Learning approaches for haplotype analysis, casual inference, and Polygenic Scores
- Including methods for multithreading and GPU acceleration
- Building out unit tests

