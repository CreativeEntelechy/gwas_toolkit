from setuptools import setup

setup(
    name="gwas_toolkit",
    version="0.0.1",
    description="GWAS analysis toolkit that is memory-aware and fast",
    packages=["gwas_analysis"],
    install_requires=[
        "numpy",
        "scipy",
        "numba",
        "tqdm",
        "scikit-learn",
        "pyarrow",
        "bed_reader"
    ],
)
