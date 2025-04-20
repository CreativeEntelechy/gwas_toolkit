# gwas_toolkit
A Python package for efficient, memory-managed Genome-Wide Association Study (GWAS) analysis with optimized sparse matrix operations. Currently in development phase with new features soon to be released!

## Overview
This package provides tools for loading genetic data, filtering SNPs, and calculating linkage disequilibrium (LD) while managing memory and disk constraints. A typical GWAS dataset is massive, with millions of columns. Therefore, memory and disk requirements are exceptionally large and often cloud compute resources are necessary. This package minimizes storage constraints by efficiently storing memory-mapped matrices using Arrow, ZTSD compression, and sparse matrices. Memory is an even more expensive resource and this package minimizes memory usage by implementing out-of-core operations, batch processing, and built-in memory monitoring. 

A less technical explanation is in order. GWAS datasets are inherently sparse since 0's typically indicate that a locus is homozygous with the reference allele. This means that we only really need to store the non-zero values as well as their pointers. When we combine sparse representation with the compression techniques implemented in the `MemmapManager` class we are able to achieve a file size that is less than the binary file! 

Another bottleneck is attempting to process this data. Since the data, even in sparse representation, is massive, it is easy to get run out of memory. A computer's memory can only hold a fraction of what can be stored on disk. Therefore, the `MemmapManager` class helps us by allowing us to memory map our files onto disk and access them as if they were available in memory. As a result, it is possible to process massive files on your desktop computer without much hassle. 

This package also aims to provide a suite of methods to perform GWAS analysis efficiently. Currently filtering and linkage disequilibrium are implemented, but in the near future I am to implement a full suite of GWAS tools to manage the entire analysis pipeline. This will include imputation, quality control, haplotype analysis, polygenic score calculation, dimensionality reduction, and statistical analysis.

## Test Data
Clearly it is not possible to access human studies without research affiliation or being on an IRB, so non-human datasets are the best option.

A great resources to download GWAS data for rice can be found at the [Rice SNP-seek Database](https://snp-seek.irri.org/), which holds a wealth of publicly available genotype and phenotype data for a variety of rice species. 

For my initial testing I used the 1M SNP dataset which was created from the full dataset after several rounds of LD-pruning.

## Core Functionalities

### Setting up the MemmapManager class
The manager class will handle file processing for you as well as storing memory-mapped files. All you need to do is instantiate the class and, optionally, tell it where to store the temporary files and provide a memory limit. The memory limit should be based on the specifications of your device or server. For example, if you device has 8GB or RAM, you might consider setting the limit to be 1GB.

### Loading data 
Currently this pipeline accepts `.bed` files, which is a common format used for tools, such as Plink. You will need to instantiate the manager class included in this package to perform all core operations. All you need to do is provide the file path and possibly adjust the parameters if you wish to adjust the chunk size. You will use the `load_bed_data` function included in this package

### Filtering data
Once you instantiate the manager class you are then able to filter the data. You will use the `filter_snps_memmap` to accomplish this task. In this function you are able to indicate the missing value (often -127 in Plink files) as well as the threshold for missingness. You are also able to indicate your desired chunk size. It will return the filtered data as a separate memory-mapped file.

### Performing Linkage Disequilibrium
This is one of the coolest current features of the pipeline. Linkage Disequilibrium (LD) is one of the most computationally challenging aspects of GWAS analysis. This package uses Hill & Robinson's r-squared method. Calculating LD means finding the pairwise r-squared for all combinations of loci. This task has $O(n^2)$ time complexity, which means that it scales quadratically as the size of the dataset increases.

The approach that this package uses is to use Locality-Sensitive Hashing (LSH) to estimate r-squared values. LSH is a high-dimensional analysis technique that aims to find significant covariances without brute force calculation. The data is first reduced into lower-dimensional space by random projections. Then random hashes are generated and used to create a hash for each data point. Items that have a similar hash are put into a bucket. This will be repeated for all of the random hashes, the number of which is an argument to `approximate_pairwise_r_squared` function. This will be computed across chunks since it is far more likely for SNPs that are closer to each other to be in LD. We will keep all SNP pairs whose estimated r-squared is larger than the pre-set `threshold` argument.

Once this step is complete we will next look at all of the pairs that were found to be significant. Clearly our approximation from the last step will make some errors and therefore we use exact windowed calculations. The errors are far more likely to occur around the edges of the LD blocks. The `window_size` argument coordinates the search by computing an exact r-squared calculation on each pair and **any** pair within its window. The exact calculation is significant sped up by using Numba's Just-In-Time (JIT) compiler to speed up numerical calculations. Python code is inherently much slower than low-level code and so Numba allows us to calculate exact windows without as much overhead.

## Installation
Pre-requisite: you should have your favorite package manager installed. 

I will lead you through a Conda installation. All you need to do is download [Anaconda](`https://www.anaconda.com/download`)

You will clone the repostory on your local machine and install it within its own conda environment.

```bash
git clone https://github.com/ItsReces/gwas_toolkit.git
cd gwas_toolkit
conda env create -f environment.yml
conda activate gwas_toolkit
```

## Usage Example

You may find a complete pipeline for implementing the available functions below. The conda environment has Jupyter already installed, so you are welcome to open an interactive notebook and play around with it.

```python
# Install libraries
from gwas_toolkit import * # Or import only what you want to use
import os
import scipy.sparse

# Instantiate the memory manager (this will create the directory)
memmap_manager = MemmapManager(temp_dir="gwas_temp", max_memory = 2) # indicate the desired temp directory and max memory usage in GB

# Load genetic data from a PLINK BED file as a sparse matrix
# You can download sample data here: https://snp-seek.irri.org/
# In this test example it is assumed that the data is stored in data_raw in your working directory. Modified accordingly for your use case
base_dir = os.getcwd()
files_path = os.path.join(base_dir, 'data_raw')
bed_file = os.path.join(files_path, '<name of your file>')

# This will load your data. It returns nothing because the file is memory mapped.
load_bed_data(
    memmap_manager, # The manager class you instantiate
    bed_file, # the path to your file
    output_name="sparse_data", # the output memmap name you want
    cores_to_use = 1, # this tells the reader how many threads to use for reading the file
    batch_size=10000, # batch size (this will influence memory usage)
    dtype='int8' # data type (for GWAS it will be int8)
)

# Filter the data to include only SNPs with 10% or less missingness and save as filtered in temp folder
filter_snps_memmap(memmap_manager, "sparse_data", "filtered", threshold=0.1, 
                       missing_value=-127, chunk_size=10000)

# Approximate the r-squared matrix and save as ld_calc in temp folder
approximate_pairwise_r_squared(memmap_manager, 
    "filtered", 
    "ld_calc", 
    chunk_size = 10000,
    hash_size = 64,
    approx_threshold = 0.2,
    window_size = 200
)

# Load the ld matrix to disk and save it to disk permanently
ld_matrix = memmap_manager.get_memmap("ld_calc") # This will load the file to disk
scipy.sparse.save_npz("ld_calc.npz", ld_matrix) # If you wish to save your ld calculations

# Clean up all files
memmap_manager.delete_all_temp() # This will delete all temporary files
memmap_manager.delete_all() # This will delete all memmap files in your temp_dir
```

# Future Development Notes
In the future I plan to build out more exciting features. Among these are the following:
- Complete Quality Control suite
- Imputation methods
- Haplotype Analysis 
- Calculating Polygenic Scores
- Better dimensionality reduction methods than PLINK :wink:
- Causal inference
- Machine Learning enhancements for phenotype analysis
- Including methods for multithreading and GPU acceleration 
- Building out unit tests

